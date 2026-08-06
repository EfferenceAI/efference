////////////////////////////////////////////////////////////////////////////////
//
// File:      grab_stress_v3.cpp
// Purpose:   Session-start stress harness: repeat open -> grab -> close and
//            measure the startup frame cadence, to reproduce and characterize
//            the 1200p H.265 stream-startup pause.
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
//
// grab_stress_v3 repeatedly opens a fresh session, captures for a short window
// (long enough to cover the startup transient), and closes. For every cycle it
// records the open->first-frame latency and the largest inter-frame gap, then
// flags any cycle whose startup gap looks like the reported stall (a multi-frame
// pause followed by a jump in frame IDs). It changes NO SDK behavior; pass
// --debug to turn on the SDK's [ef.diag] stream diagnostics (InitParameters::
// verbose = 2) so the harness's host-side view can be lined up against the
// SDK-internal packet/frame/decoder timeline.
//
//   ./build/grab_stress_v3                         # 10 cycles, 1200p H.265, 3 s each
//   ./build/grab_stress_v3 --cycles 20 --debug     # + SDK [ef.diag] logging
//   ./build/grab_stress_v3 --secs 5 --gap-ms 200   # flag gaps over 200 ms
//
////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <ef/Device.hpp>

using namespace ef;

namespace {

uint64_t mono_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

struct CycleResult {
    bool     opened          = false;
    uint64_t frames          = 0;
    double   open_to_first_ms = 0.0;   // open() return -> first frame returned
    double   max_gap_ms      = 0.0;    // largest inter-frame delta
    uint32_t gap_prev_id     = 0;      // frame ids straddling the largest gap
    uint32_t gap_next_id     = 0;
    ERROR_CODE open_ec       = ERROR_CODE::SUCCESS;
};

}  // namespace

int main(int argc, char** argv) {
    int    cycles  = 10;
    double secs    = 3.0;     // per-cycle capture window; startup is what matters
    double gap_ms  = 100.0;   // flag a startup gap wider than this (30 fps ~ 33 ms)
    bool   debug   = false;

    InitParameters init;
    init.resolution  = RESOLUTION::HD1200;      // 1920x1200, the reported config
    init.compression = COMPRESSION_MODE::H265;  // the reported codec
    init.fps         = 30;

    for (int i = 1; i < argc; i++) {
        if      (!std::strcmp(argv[i], "--cycles") && i + 1 < argc) cycles = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--secs")   && i + 1 < argc) secs   = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--gap-ms") && i + 1 < argc) gap_ms = std::atof(argv[++i]);
        else if (!std::strcmp(argv[i], "--debug")) debug = true;
        else if (!std::strcmp(argv[i], "--codec")  && i + 1 < argc) {
            const std::string c = argv[++i];
            if      (c == "raw")    init.compression = COMPRESSION_MODE::RAW;
            else if (c == "h264")   init.compression = COMPRESSION_MODE::H264;
            else if (c == "h264hq") init.compression = COMPRESSION_MODE::H264_HQ;
            else if (c == "h265")   init.compression = COMPRESSION_MODE::H265;
            else if (c == "h265hq") init.compression = COMPRESSION_MODE::H265_HQ;
            else { std::fprintf(stderr, "unknown codec '%s'\n", c.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--res") && i + 1 < argc) {
            const std::string r = argv[++i];
            if      (r == "1200") init.resolution = RESOLUTION::HD1200;
            else if (r == "1080") init.resolution = RESOLUTION::HD1080;
            else if (r == "svga") init.resolution = RESOLUTION::SVGA;
            else { std::fprintf(stderr, "unknown res '%s' (1200|1080|svga)\n", r.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--ble") && i + 1 < argc) {
            init.input_type  = INPUT_TYPE::STREAM;
            init.ble_address = argv[++i];
        } else if (!std::strcmp(argv[i], "--udp") && i + 1 < argc) {
            std::string hp = argv[++i];
            std::string::size_type colon = hp.rfind(':');
            if (colon != std::string::npos) {
                init.udp_port = (uint16_t)std::atoi(hp.c_str() + colon + 1);
                hp.resize(colon);
            }
            init.udp_host = hp;
        } else if (!std::strcmp(argv[i], "--password") && i + 1 < argc) {
            init.ble_password = argv[++i];
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }

    // The single opt-in switch: level 2 turns on the SDK stream diagnostics.
    if (debug) init.verbose = 2;

    std::printf("grab_stress_v3: %d cycle(s), %.1fs each, gap threshold %.0f ms, "
                "diagnostics %s\n",
                cycles, secs, gap_ms, debug ? "ON (verbose=2)" : "off");

    std::vector<CycleResult> results;
    int suspicious = 0;

    for (int c = 0; c < cycles; c++) {
        CycleResult r;
        Device device;
        ERROR_CODE ec = device.open(init);
        r.open_ec = ec;
        if (ec != ERROR_CODE::SUCCESS) {
            std::printf("cycle %2d: open failed: %s\n", c, to_string(ec));
            results.push_back(r);
            continue;
        }
        r.opened = true;

        const uint64_t t_open = mono_ns();
        uint64_t t_prev = 0;
        uint32_t prev_id = 0;
        bool     have_prev = false;
        Mat      frame;

        while ((double)(mono_ns() - t_open) / 1e9 < secs) {
            ERROR_CODE g = device.grab();
            if (g == ERROR_CODE::GRAB_TIMEOUT || g == ERROR_CODE::CORRUPTED_FRAME) continue;
            if (g != ERROR_CODE::SUCCESS) {
                std::printf("cycle %2d: grab error: %s\n", c, to_string(g));
                break;
            }
            if (device.retrieve_image(frame, VIEW::NV12) != ERROR_CODE::SUCCESS) continue;

            const uint64_t now = mono_ns();
            r.frames++;
            if (!have_prev) {
                r.open_to_first_ms = (double)(now - t_open) / 1e6;
                have_prev = true;
            } else {
                double dt = (double)(now - t_prev) / 1e6;
                if (dt > r.max_gap_ms) {
                    r.max_gap_ms  = dt;
                    r.gap_prev_id = prev_id;
                    r.gap_next_id = frame.getFrameId();
                }
            }
            t_prev  = now;
            prev_id = frame.getFrameId();
        }

        device.close();

        const bool flagged = r.max_gap_ms > gap_ms;
        if (flagged) suspicious++;
        std::printf("cycle %2d: frames=%llu open->first=%.1f ms  max_gap=%.1f ms "
                    "ids[%u->%u skip=%d]%s\n",
                    c, (unsigned long long)r.frames, r.open_to_first_ms, r.max_gap_ms,
                    r.gap_prev_id, r.gap_next_id,
                    r.gap_next_id > r.gap_prev_id ? (int)(r.gap_next_id - r.gap_prev_id - 1) : 0,
                    flagged ? "  <== STARTUP GAP" : "");
        results.push_back(r);
    }

    // ---- summary ----
    double worst = 0.0; int worst_cycle = -1;
    uint64_t total_frames = 0; int opened = 0;
    for (size_t i = 0; i < results.size(); i++) {
        total_frames += results[i].frames;
        opened += results[i].opened ? 1 : 0;
        if (results[i].max_gap_ms > worst) { worst = results[i].max_gap_ms; worst_cycle = (int)i; }
    }
    std::printf("\n== summary ==\n");
    std::printf("cycles opened: %d/%d   total frames: %llu\n",
                opened, cycles, (unsigned long long)total_frames);
    std::printf("cycles with a startup gap > %.0f ms: %d\n", gap_ms, suspicious);
    if (worst_cycle >= 0)
        std::printf("worst gap: %.1f ms on cycle %d (ids %u -> %u, %d skipped)\n",
                    worst, worst_cycle, results[worst_cycle].gap_prev_id,
                    results[worst_cycle].gap_next_id,
                    results[worst_cycle].gap_next_id > results[worst_cycle].gap_prev_id
                        ? (int)(results[worst_cycle].gap_next_id - results[worst_cycle].gap_prev_id - 1)
                        : 0);
    if (debug)
        std::printf("(SDK [ef.diag] lines above carry the per-packet/frame/decoder timeline)\n");

    return suspicious > 0 ? 0 : 0;   // exit 0: this is a measurement tool, not a gate
}
