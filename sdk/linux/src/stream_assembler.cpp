////////////////////////////////////////////////////////////////////////////////
//
// File:      stream_assembler.cpp
// Purpose:   Transport-agnostic ef_stream reassembler (internal).
// Author:    Calvin Nguyen
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

#include "stream_assembler.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace ef {
namespace internal {

namespace {
// ef_stream wire layout: 8-B common + 36-B video header (payload @44)
// | 12-B IMU batch header (samples @20, 40 B each).
constexpr uint8_t  kMagic       = 0xEF;
constexpr uint8_t  kTypeVideo   = 1, kTypeImu = 2;
constexpr uint8_t  kFragStart   = 0x01, kFragEnd = 0x02;
constexpr int      kVidHdr      = 44;   // 8 + 36
constexpr int      kImuHdr      = 20;   // 8 + 12
constexpr int      kImuSample   = 40;
// Untrusted wire (no retransmit, crc unchecked): cap reassembly at 4K NV12.
constexpr uint32_t kMaxFrameBytes = 3840u * 2160u * 3u / 2u;

uint16_t rd16(const uint8_t* p) { return (uint16_t)(p[0] | p[1] << 8); }
uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
uint64_t rd64(const uint8_t* p) {
    uint64_t lo = rd32(p), hi = rd32(p + 4);
    return lo | (hi << 32);
}
float rdf(const uint8_t* p) { float f; std::memcpy(&f, p, 4); return f; }

// Does this encoded access unit begin an IDR/IRAP keyframe? Walks the leading
// Annex-B NAL units (encoder prepends VPS/SPS/PPS to every IDR), returns true on the
// first keyframe-class NAL. codec: 1=H264, 2=H265. Bounded scan of the first few NAL
// headers only (negligible at 60 fps). Handles 3- and 4-byte start codes (the 4-byte
// form contains the 3-byte pattern one byte in, which the scan lands on).
bool is_keyframe(const uint8_t* d, size_t n, int codec) {
    int scanned = 0;
    for (size_t i = 0; i + 3 < n && scanned < 16; ) {
        if (d[i] == 0 && d[i + 1] == 0 && d[i + 2] == 1) {
            uint8_t h = d[i + 3];                 // first byte of the NAL header
            if (codec == 1) {                     // H264: type = h & 0x1F
                int t = h & 0x1F;
                if (t == 5 || t == 7) return true;              // IDR slice, SPS
            } else {                              // H265: type = (h >> 1) & 0x3F
                int t = (h >> 1) & 0x3F;
                if ((t >= 16 && t <= 23) || t == 32 || t == 33) // IRAP, VPS, SPS
                    return true;
            }
            ++scanned;
            i += 3;
        } else {
            ++i;
        }
    }
    return false;
}
}  // namespace

int StreamAssembler::free_vbuf() const {
    for (int i = 0; i < VBUFS; i++)
        if (i != ready_ && i != consuming_ && i != reasm_) return i;
    return -1;  // unreachable with 4 buffers and <=2 held
}

void StreamAssembler::grow_vbufs(size_t need) {
    if (reasm_ >= 0 && vbuf_[reasm_].size() < need) vbuf_[reasm_].resize(need);
}

void StreamAssembler::mark_ended() {
    { std::lock_guard<std::mutex> lk(vmtx_); ended_ = true; }
    vcv_.notify_all();
}

// ---- reassembly (transport feeds one ef_stream packet at a time) -----------

void StreamAssembler::on_packet(const uint8_t* b, int len) {
    if (len < 8 || b[0] != kMagic) return;
    uint8_t type = b[2], flags = b[3];

    if (type == kTypeVideo) {
        if (len < kVidHdr) return;
        // Wire loss detection: header seq (b+4) is a per-stream monotonic packet
        // counter; a gap means a dropped video packet -> tell the transport (may
        // request a keyframe/PLI). Single transport thread, so last_vseq_ needs no
        // lock; call on_loss() BEFORE vmtx_ so the override never runs under it.
        uint32_t seq = rd32(b + 4);
        if (have_vseq_ && seq != (uint32_t)(last_vseq_ + 1)) {
            resync_pending_ = true;   // withhold P-frames until the next IDR
            on_loss();
        }
        last_vseq_ = seq;
        have_vseq_ = true;

        uint32_t frame_id = rd32(b + 8), offset = rd32(b + 12),
                 plen = rd32(b + 16), fsize = rd32(b + 20);
        uint16_t w = rd16(b + 24), h = rd16(b + 26);
        uint8_t  pixfmt = b[28];
        uint64_t ts = rd64(b + 32);
        int paylen = len - kVidHdr;   // >= 0 (guarded by len < kVidHdr above)
        // Unsigned compare: a plen > INT_MAX would make (int)plen negative and
        // silently skip this clamp, letting the memcpy below over-read the packet.
        if ((uint32_t)paylen < plen) plen = (uint32_t)paylen;  // defensive

        // ---- diagnostics: first encoded packet on the wire (transport thread) --
        const int dlvl = diag_level_.load(std::memory_order_relaxed);
        if (diag_on(dlvl) && !first_packet_seen_) {
            first_packet_seen_ = true;
            std::fprintf(stderr,
                         "[ef.diag] t=%llu s=%llu ev=first_encoded_packet "
                         "frame_id=%u dev_ts=%llu\n",
                         (unsigned long long)diag_mono_ns(),
                         (unsigned long long)diag_session_, frame_id,
                         (unsigned long long)ts);
        }

        bool superseded = false;
        bool need_pli   = false;
        // Diagnostics captured under vmtx_, emitted after it is released so no
        // stderr I/O ever happens while the video lock is held.
        bool     log_gap = false, log_rx = false;
        uint32_t gap_last = 0, gap_next = 0, gap_skipped = 0;
        bool     rx_key = false, rx_gated = false, rx_complete = false;
        uint32_t rx_id = 0;  uint64_t rx_dev_ts = 0, rx_host_ns = 0;
        int      rx_w = 0, rx_h = 0;  size_t rx_size = 0;  double rx_dt_ms = 0.0;
        {
            std::lock_guard<std::mutex> lk(vmtx_);
            if (flags & kFragStart) {
                if (fsize == 0 || fsize > kMaxFrameBytes) {
                    in_frame_ = false;   // reject corrupt/oversized frame
                } else {
                    // Frame-ID gap detection (observe-only; does NOT drive the
                    // resync gate, which stays keyed on the packet seq above).
                    // Surfaces the "skipped N frame IDs" symptom directly.
                    if (diag_on(dlvl) && have_frame_id_ &&
                        frame_id != (uint32_t)(last_frame_id_ + 1) &&
                        frame_id > last_frame_id_) {
                        log_gap     = true;
                        gap_last    = last_frame_id_;
                        gap_next    = frame_id;
                        gap_skipped = frame_id - last_frame_id_ - 1;
                    }
                    last_frame_id_ = frame_id;
                    have_frame_id_ = true;
                    if (reasm_ == -1) reasm_ = free_vbuf();
                    if (reasm_ >= 0) {
                        grow_vbufs(fsize);
                        in_frame_ = true; cur_frame_id_ = frame_id; cur_size_ = fsize; cur_accum_ = 0;
                        RawFrame m; m.frame_id = frame_id; m.ts_ns = ts;
                        m.width = w; m.height = h; m.pixfmt = pixfmt;
                        vmeta_[reasm_] = m;
                    }
                }
            }
            if (in_frame_ && reasm_ >= 0 && frame_id == cur_frame_id_ &&
                (size_t)offset + plen <= vbuf_[reasm_].size()) {
                std::memcpy(vbuf_[reasm_].data() + offset, b + kVidHdr, plen);
                cur_accum_ += plen;
            }
            if (flags & kFragEnd) {
                if (in_frame_ && reasm_ >= 0) {
                    vmeta_[reasm_].data     = vbuf_[reasm_].data();
                    vmeta_[reasm_].size     = cur_size_;
                    vmeta_[reasm_].complete = (cur_accum_ == cur_size_);

                    // A supersede (consumer never grabbed the last ready_, e.g. vsync
                    // throttled while stalled) breaks the reference chain for whatever
                    // it grabs next, so enter resync.
                    if (ready_ != -1) { superseded = true; resync_pending_ = true; }

                    // Keyframe classification, computed once. The gate needs it only
                    // while resync is pending; diagnostics want it on every complete
                    // frame. RAW/MJPEG (codec 0) is intra, so every complete frame is
                    // a keyframe and is never gated.
                    int  codec = vcodec_.load(std::memory_order_relaxed);
                    bool want_kf = diag_on(dlvl) || (codec != 0 && resync_pending_);
                    bool kf = false;
                    if (vmeta_[reasm_].complete) {
                        kf = (codec == 0) ? true
                           : (want_kf ? is_keyframe(vbuf_[reasm_].data(), cur_size_, codec)
                                      : false);
                    }
                    vmeta_[reasm_].keyframe = kf;

                    // Drop-until-IDR gate (encoded streams only). While resync pending,
                    // withhold every frame until an IDR/IRAP arrives, never feed the
                    // decoder an orphaned P-frame ("Could not find ref with POC" spam).
                    // Incomplete frames never clear it.
                    bool gated = false;
                    if (codec != 0 && resync_pending_) {
                        if (vmeta_[reasm_].complete && kf)
                            resync_pending_ = false;   // IDR: decoder resyncs here
                        else
                            gated = true;              // hold for the next IDR
                    }

                    // Track the last keyframe + rx timing for the diagnostics.
                    if (kf && vmeta_[reasm_].complete) {
                        last_keyframe_id_    = cur_frame_id_;
                        last_keyframe_ts_ns_ = vmeta_[reasm_].ts_ns;
                    }
                    if (vmeta_[reasm_].complete) total_frames_++;
                    if (diag_on(dlvl)) {
                        rx_host_ns = diag_mono_ns();
                        log_rx      = true;
                        rx_id       = cur_frame_id_;
                        rx_dev_ts   = vmeta_[reasm_].ts_ns;
                        rx_w        = vmeta_[reasm_].width;
                        rx_h        = vmeta_[reasm_].height;
                        rx_size     = cur_size_;
                        rx_key      = kf;
                        rx_complete = vmeta_[reasm_].complete;
                        rx_dt_ms    = last_frame_mono_ns_
                                          ? diag_ms_since(last_frame_mono_ns_, rx_host_ns)
                                          : 0.0;
                        last_frame_mono_ns_ = rx_host_ns;
                    }

                    if (gated) {
                        frames_dropped_++;   // new frame dropped; last ready_ kept
                        reasm_ = -1;   // drop; the last good ready_ (if any) stays
                        need_pli = true;     // keep asking for an IDR to resync
                    } else {
                        if (ready_ != -1) frames_dropped_++;  // replaced ungrabbed
                        ready_ = reasm_;
                        reasm_ = -1;
                        vcv_.notify_one();
                    }
                    rx_gated = gated;
                }
                in_frame_ = false;
            }
        }
        // Diagnostics emitted outside vmtx_ (one fprintf per line; libc locks the
        // FILE so lines from the transport and consumer threads never interleave).
        if (log_gap)
            std::fprintf(stderr,
                         "[ef.diag] t=%llu s=%llu ev=frame_gap last_id=%u next_id=%u "
                         "skipped=%u last_key_id=%u last_key_ts=%llu resync=%d\n",
                         (unsigned long long)diag_mono_ns(),
                         (unsigned long long)diag_session_, gap_last, gap_next,
                         gap_skipped, last_keyframe_id_,
                         (unsigned long long)last_keyframe_ts_ns_,
                         resync_pending_ ? 1 : 0);
        if (log_rx)
            std::fprintf(stderr,
                         "[ef.diag] t=%llu s=%llu rx frame_id=%u dev_ts=%llu "
                         "host_rx_ns=%llu res=%dx%d size=%zu key=%d complete=%d "
                         "gated=%d dt_ms=%.3f\n",
                         (unsigned long long)rx_host_ns,
                         (unsigned long long)diag_session_, rx_id,
                         (unsigned long long)rx_dev_ts,
                         (unsigned long long)rx_host_ns, rx_w, rx_h, rx_size,
                         rx_key ? 1 : 0, rx_complete ? 1 : 0, rx_gated ? 1 : 0,
                         rx_dt_ms);
        // Request a fresh keyframe (PLI) after a supersede or while gated. Called
        // OUTSIDE the video mutex; UDP overrides on_loss(), USB isoc (no back channel)
        // is a no-op. need_pli keeps requesting on every withheld frame until the IDR.
        if (superseded || need_pli) on_loss();
    } else if (type == kTypeImu) {
        if (len < kImuHdr) return;
        uint64_t base_seq = rd64(b + 8);
        uint16_t n = rd16(b + 16);
        uint16_t stride = rd16(b + 18);
        if (stride < kImuSample) return;
        std::lock_guard<std::mutex> lk(imtx_);
        for (uint16_t i = 0; i < n; i++) {
            int off = kImuHdr + i * stride;
            if (off + kImuSample > len) break;
            const uint8_t* p = b + off;
            ImuSample s;
            s.timestamp.data_ns    = rd64(p);
            s.acceleration[0]      = rdf(p + 8);
            s.acceleration[1]      = rdf(p + 12);
            s.acceleration[2]      = rdf(p + 16);
            s.angular_velocity[0]  = rdf(p + 20);
            s.angular_velocity[1]  = rdf(p + 24);
            s.angular_velocity[2]  = rdf(p + 28);
            s.temperature_c        = rdf(p + 32);
            s.sequence             = base_seq + i;
            if (imu_.size() >= IMU_CAP) { imu_.pop_front(); imu_dropped_++; }
            imu_.push_back(s);
        }
    }
}

// ---- consumer side ---------------------------------------------------------

Status StreamAssembler::grab(uint32_t timeout_ms, bool return_partial) {
    std::unique_lock<std::mutex> lk(vmtx_);
    if (consuming_ != -1) consuming_ = -1;   // release the previously grabbed slot
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(timeout_ms);
    for (;;) {
        if (ready_ != -1) {
            int s = ready_; ready_ = -1;
            if (vmeta_[s].complete || return_partial) {
                consuming_ = s;
                return vmeta_[s].complete ? Status::SUCCESS : Status::CORRUPTED_FRAME;
            }
            continue;  // drop the lossy frame, keep waiting for a complete one
        }
        if (ended_) return Status::END_OF_STREAM;
        if (vcv_.wait_until(lk, deadline) == std::cv_status::timeout && ready_ == -1)
            return ended_ ? Status::END_OF_STREAM : Status::TIMEOUT;
    }
}

bool StreamAssembler::current_video(RawFrame* out) const {
    std::lock_guard<std::mutex> lk(vmtx_);
    if (consuming_ == -1) return false;
    *out = vmeta_[consuming_];
    return true;
}

void StreamAssembler::get_stats(StreamStats* out) const {
    if (!out) return;
    {
        std::lock_guard<std::mutex> lk(vmtx_);
        out->last_frame_id       = last_frame_id_;
        out->have_frame_id       = have_frame_id_;
        out->last_keyframe_id    = last_keyframe_id_;
        out->last_keyframe_ts_ns = last_keyframe_ts_ns_;
        out->last_frame_mono_ns  = last_frame_mono_ns_;
        out->frames_dropped      = frames_dropped_;
        out->total_frames        = total_frames_;
        out->ready_slot          = ready_;
        out->reasm_slot          = reasm_;
        out->consuming_slot      = consuming_;
        out->resync_pending      = resync_pending_;   // best-effort read (see header)
        out->running             = running_.load();
    }
    std::lock_guard<std::mutex> lk(imtx_);
    out->imu_queue_depth = imu_.size();
}

void StreamAssembler::drain_imu(std::vector<ImuSample>& out, bool latest_only,
                                uint64_t* dropped) {
    std::lock_guard<std::mutex> lk(imtx_);
    if (dropped) *dropped = imu_dropped_;
    imu_dropped_ = 0;
    if (imu_.empty()) { out.clear(); return; }
    if (latest_only) { out.assign(1, imu_.back()); imu_.clear(); return; }
    out.assign(imu_.begin(), imu_.end());
    imu_.clear();
}

bool StreamAssembler::peek_latest_accel(float out[3]) {
    std::lock_guard<std::mutex> lk(imtx_);
    if (imu_.empty()) return false;
    const ImuSample& s = imu_.back();
    out[0] = s.acceleration[0];
    out[1] = s.acceleration[1];
    out[2] = s.acceleration[2];
    return true;
}

}  // namespace internal
}  // namespace ef
