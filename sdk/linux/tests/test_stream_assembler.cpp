////////////////////////////////////////////////////////////////////////////////
//
// File:      tests/test_stream_assembler.cpp
// Purpose:   Hardware-free unit tests for StreamAssembler: the drop-until-IDR
//            startup gate, keyframe classification, and the diagnostics
//            tracking (frame-id / keyframe / stats) added for the stream-startup
//            investigation.
// Author:    Calvin Nguyen, Gianluca Bencomo
//
// Copyright (c) 2026, Remnant Robotics, Inc. All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
////////////////////////////////////////////////////////////////////////////////

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "internal/stream_assembler.hpp"

using namespace ef;
using namespace ef::internal;

namespace {

int g_failures = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                      \
        }                                                                      \
    } while (0)

// Test harness: a StreamAssembler that lets the test feed raw ef_stream packets
// (on_packet is protected) and provides the pure-virtual stop().
struct TestAssembler : public StreamAssembler {
    void stop() override {}
    void feed(const std::vector<uint8_t>& pkt) { on_packet(pkt.data(), (int)pkt.size()); }
};

void wr16(uint8_t* p, uint16_t v) { p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; }
void wr32(uint8_t* p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}
void wr64(uint8_t* p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }

// One single-fragment video packet (ef_stream: 8-B common + 36-B video hdr + payload).
std::vector<uint8_t> vpkt(uint32_t seq, uint32_t frame_id, uint64_t ts,
                          const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> b(44 + payload.size(), 0);
    b[0] = 0xEF;              // magic
    b[2] = 1;                 // type = video
    b[3] = 0x01 | 0x02;       // FragStart | FragEnd (whole frame in one packet)
    wr32(&b[4], seq);
    wr32(&b[8], frame_id);
    wr32(&b[12], 0);                          // offset
    wr32(&b[16], (uint32_t)payload.size());   // plen
    wr32(&b[20], (uint32_t)payload.size());   // fsize
    wr16(&b[24], 1920);
    wr16(&b[26], 1200);
    b[28] = 0;                                // pixfmt
    wr64(&b[32], ts);
    std::memcpy(&b[44], payload.data(), payload.size());
    return b;
}

// H.265 access units: a keyframe leads with a VPS NAL (type 32 -> byte 0x40),
// a P-frame with a TRAIL NAL (type 1 -> byte 0x02). is_keyframe() reads the first
// NAL after the 00 00 01 start code.
std::vector<uint8_t> h265_key() { return {0, 0, 1, 0x40, 0x01, 0xAA, 0xBB, 0xCC}; }
std::vector<uint8_t> h265_p()   { return {0, 0, 1, 0x02, 0x01, 0x11, 0x22, 0x33}; }

// ---- tests ----

// Drop-until-IDR startup gate: before the first keyframe, encoded frames are
// withheld (grab times out); the first keyframe opens the gate.
void test_startup_gate_holds_until_keyframe() {
    TestAssembler a;
    a.set_video_codec(2);   // H265

    a.feed(vpkt(/*seq*/1, /*id*/1, /*ts*/1000, h265_p()));      // P before any IDR
    CHECK(a.grab(20, false) == Status::TIMEOUT);                // withheld

    a.feed(vpkt(2, 2, 2000, h265_key()));                       // IDR opens the gate
    CHECK(a.grab(20, false) == Status::SUCCESS);
    StreamAssembler::RawFrame f;
    CHECK(a.current_video(&f) && f.frame_id == 2 && f.keyframe && f.complete);

    a.feed(vpkt(3, 3, 3000, h265_p()));                         // P after IDR flows
    CHECK(a.grab(20, false) == Status::SUCCESS);
    CHECK(a.current_video(&f) && f.frame_id == 3 && !f.keyframe);
}

// RAW/MJPEG (codec 0) is intra: never gated, every complete frame is a keyframe.
void test_raw_never_gated() {
    TestAssembler a;
    a.set_video_codec(0);   // RAW
    a.feed(vpkt(1, 10, 500, {1, 2, 3, 4, 5, 6}));
    CHECK(a.grab(20, false) == Status::SUCCESS);
    StreamAssembler::RawFrame f;
    CHECK(a.current_video(&f) && f.frame_id == 10 && f.keyframe && f.complete);
}

// Diagnostics tracking: last frame id, last keyframe id/ts, and total frame count
// advance across a keyframe -> P -> (frame-id gap) keyframe sequence.
void test_stats_track_frames_and_keyframes() {
    TestAssembler a;
    a.set_video_codec(2);
    a.set_debug(kDiagLevel, /*session*/7);   // exercise the diagnostics code paths

    a.feed(vpkt(1, 100, 100000, h265_key())); (void)a.grab(20, false);
    a.feed(vpkt(2, 101, 133000, h265_p()));   (void)a.grab(20, false);
    // Frame-id gap: jump from 101 to 158 (56 skipped), as a keyframe.
    a.feed(vpkt(3, 158, 200000, h265_key())); (void)a.grab(20, false);

    StreamAssembler::StreamStats st;
    a.get_stats(&st);
    CHECK(st.have_frame_id && st.last_frame_id == 158);
    CHECK(st.last_keyframe_id == 158);
    CHECK(st.last_keyframe_ts_ns == 200000);
    CHECK(st.total_frames == 3);
    CHECK(st.running == false);   // no transport thread in the harness
}

}  // namespace

int main() {
    // Silence the [ef.diag] lines the debug-path test emits (behavior under test
    // is the stats, not the log text); comment out to inspect the diagnostics.
    std::freopen("/dev/null", "w", stderr);

    test_startup_gate_holds_until_keyframe();
    test_raw_never_gated();
    test_stats_track_frames_and_keyframes();

    // stderr is /dev/null; report the result on stdout.
    std::printf("stream_assembler tests: %s (%d failure(s))\n",
                g_failures ? "FAILED" : "passed", g_failures);
    return g_failures ? 1 : 0;
}
