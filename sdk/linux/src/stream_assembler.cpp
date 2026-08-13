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
// Plausibility bounds. A sequence number that moves absurdly is a bad read, not a
// real gap, and these counters are monotonic, so anything booked on one is
// permanent. Ahead by more than these books nothing. kMaxGap is ~2 min of solid
// blackout at 30 fps and kMaxSeqGap ~30 s of raw 1200p, both far past any outage a
// session survives.
constexpr int32_t kMaxGap    = 1 << 12;
constexpr int32_t kMaxSeqGap = 1 << 20;
// Consecutive frame ids behind the current one before it is a device that
// restarted its numbering rather than reordered fragments. A reorder cannot
// produce a run: the late frames are late once, and the stream resumes ahead.
constexpr int      kRestartRun = 4;
// The wire is not trusted to size an allocation: cap reassembly at 4K NV12.
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

// Insert [lo,hi) and merge with anything it touches. Ranges stay sorted and
// disjoint, so the in-order case (lo == cov_[0].hi) just extends the first one.
void StreamAssembler::cov_add(uint32_t lo, uint32_t hi) {
    if (cov_full_ || hi <= lo) return;
    int i = 0;
    while (i < ncov_ && cov_[i].hi < lo) i++;          // ranges entirely below
    if (i == ncov_ || cov_[i].lo > hi) {               // no overlap: insert
        if (ncov_ == kCovRanges) { cov_full_ = true; return; }
        for (int j = ncov_; j > i; j--) cov_[j] = cov_[j - 1];
        cov_[i] = { lo, hi };
        ncov_++;
        return;
    }
    if (lo < cov_[i].lo) cov_[i].lo = lo;              // absorb into cov_[i]
    if (hi > cov_[i].hi) cov_[i].hi = hi;
    int j = i + 1;                                     // swallow now-touching ones
    while (j < ncov_ && cov_[j].lo <= cov_[i].hi) {
        if (cov_[j].hi > cov_[i].hi) cov_[i].hi = cov_[j].hi;
        j++;
    }
    if (j > i + 1) {
        int drop = j - (i + 1);
        for (int k = i + 1; k + drop < ncov_; k++) cov_[k] = cov_[k + drop];
        ncov_ -= drop;
    }
}

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
            // Signed delta so a reordered packet (UDP) reads negative and counts
            // nothing; only a forward jump is missing packets, and only a plausible
            // one, since a bad read here is booked permanently.
            int32_t adv = (int32_t)(seq - last_vseq_);
            if (adv > 1 && adv <= kMaxSeqGap)
                packets_lost_.fetch_add((uint32_t)(adv - 1), std::memory_order_relaxed);
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

        // Frame accounting keys on EF_FRAG_START, the one packet that marks a frame
        // boundary, so a stream joined mid-frame does not book the frame it walked
        // in on.
        if (flags & kFragStart) {
            const int32_t adv = (int32_t)(frame_id - last_fid_);
            if (!have_fid_) {
                last_fid_ = frame_id;
                have_fid_ = true;
            } else if (adv > 0) {
                // Leaving an id closes it out; it never reached EF_FRAG_END, so its
                // tail was lost.
                if (!finalized_) frames_broken_.fetch_add(1, std::memory_order_relaxed);
                // Ids strictly in between are frames of which nothing arrived at all.
                if (adv > 1 && adv <= kMaxGap)
                    frames_gone_.fetch_add((uint32_t)adv - 1, std::memory_order_relaxed);
                last_fid_    = frame_id;
                finalized_   = false;
                behind_run_  = 0;
            } else if (adv < 0 && ++behind_run_ >= kRestartRun) {
                // A run of ids behind the current one is the device numbering from
                // zero again, which a UDP reader outlives. The run is what separates
                // it from a single reordered straggler.
                if (!finalized_) frames_broken_.fetch_add(1, std::memory_order_relaxed);
                last_fid_   = frame_id;
                finalized_  = false;
                behind_run_ = 0;
            }
            // else: the same id again, or a straggler; neither is news.
        }
        int paylen = len - kVidHdr;   // >= 0 (guarded by len < kVidHdr above)
        // Unsigned compare: a plen > INT_MAX would make (int)plen negative and
        // silently skip this clamp, letting the memcpy below over-read the packet.
        if ((uint32_t)paylen < plen) plen = (uint32_t)paylen;  // defensive

        bool superseded  = false;
        bool need_pli    = false;
        bool frame_whole = false;
        {
            std::lock_guard<std::mutex> lk(vmtx_);
            // A START that does not move the id forward is a duplicate or a late
            // straggler. Honouring it would reset cur_size_ and the coverage of the
            // frame being built, destroying it.
            const bool stale_start = in_frame_ &&
                                     (int32_t)(frame_id - cur_frame_id_) <= 0;
            if ((flags & kFragStart) && !stale_start) {
                if (fsize == 0 || fsize > kMaxFrameBytes) {
                    in_frame_ = false;   // reject corrupt/oversized frame
                } else {
                    if (reasm_ == -1) reasm_ = free_vbuf();
                    if (reasm_ >= 0) {
                        grow_vbufs(fsize);
                        in_frame_ = true; cur_frame_id_ = frame_id; cur_size_ = fsize;
                        cov_reset();
                        RawFrame m; m.frame_id = frame_id; m.ts_ns = ts;
                        m.width = w; m.height = h; m.pixfmt = pixfmt;
                        vmeta_[reasm_] = m;
                    }
                }
            }
            if (in_frame_ && reasm_ >= 0 && frame_id == cur_frame_id_ &&
                (size_t)offset + plen <= vbuf_[reasm_].size()) {
                std::memcpy(vbuf_[reasm_].data() + offset, b + kVidHdr, plen);
                cov_add(offset, offset + plen);
            }
            // The id guard belongs on the WHOLE end-of-frame block, not just on the
            // finalize inside it. An END for another frame must neither publish the
            // wrong frame nor clear in_frame_ underneath the one being built, and
            // END(N) swapped with START(N+1) is the most ordinary reorder there is.
            if ((flags & kFragEnd) && in_frame_ && frame_id == cur_frame_id_) {
                if (reasm_ >= 0) {
                    vmeta_[reasm_].data     = vbuf_[reasm_].data();
                    vmeta_[reasm_].size     = cur_size_;
                    vmeta_[reasm_].complete = cov_covers(cur_size_);
                    frame_whole = vmeta_[reasm_].complete;

                    // A supersede (consumer never grabbed the last ready_, e.g. vsync
                    // throttled while stalled) breaks the reference chain for whatever
                    // it grabs next, so enter resync.
                    if (ready_ != -1) {
                        superseded = true; resync_pending_ = true;
                        // The frame ALREADY waiting died ungrabbed. Booked here and
                        // not below because a supersede and a gate are facts about
                        // two different frames, and both can be true at once.
                        frames_superseded_.fetch_add(1, std::memory_order_relaxed);
                    }

                    // Drop-until-IDR gate (encoded streams only). While resync pending,
                    // withhold every frame until an IDR/IRAP arrives, never feed the
                    // decoder an orphaned P-frame ("Could not find ref with POC" spam).
                    // Incomplete frames never clear it; RAW/MJPEG (codec 0) is intra,
                    // never gated.
                    int  codec = vcodec_.load(std::memory_order_relaxed);
                    bool gated = false;
                    if (codec != 0 && resync_pending_) {
                        if (vmeta_[reasm_].complete &&
                            is_keyframe(vbuf_[reasm_].data(), cur_size_, codec))
                            resync_pending_ = false;   // IDR: decoder resyncs here
                        else
                            gated = true;              // hold for the next IDR
                    }

                    if (gated) {
                        // THIS frame is withheld until the next keyframe, which is a
                        // consequence of loss rather than loss itself.
                        frames_gated_.fetch_add(1, std::memory_order_relaxed);
                        reasm_ = -1;   // drop; the last good ready_ (if any) stays
                        need_pli = true;     // keep asking for an IDR to resync
                    } else {
                        ready_ = reasm_;
                        reasm_ = -1;
                        vcv_.notify_one();
                    }
                }
                // Booked once per id, outside the buffer guard above so a frame that
                // found no free vbuf still counts as sent. A stale duplicate id
                // books nothing.
                if (frame_id == last_fid_ && !finalized_) {
                    finalized_ = true;
                    (frame_whole ? frames_whole_ : frames_broken_)
                        .fetch_add(1, std::memory_order_relaxed);
                }
                in_frame_ = false;
            }
        }
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

void StreamAssembler::get_stats(StreamStats* out) const {
    *out = StreamStats{};
    out->received_whole   = frames_whole_.load(std::memory_order_relaxed);
    out->received_partial = frames_broken_.load(std::memory_order_relaxed);
    out->lost_in_transit  = frames_gone_.load(std::memory_order_relaxed);
    out->packets_lost     = packets_lost_.load(std::memory_order_relaxed);
    out->dropped_by_host  = frames_superseded_.load(std::memory_order_relaxed);
    out->withheld_resync  = frames_gated_.load(std::memory_order_relaxed);

    // The buckets partition every frame the device numbered up to the last one
    // that ended, so their sum IS what it sent. The frame still in flight is in
    // none of them, which is why this never reports a loss that later un-happens.
    out->device_frames = out->received_whole + out->received_partial +
                         out->lost_in_transit;
    out->loss_percent  = out->device_frames
        ? 100.f * (float)out->lost() / (float)out->device_frames
        : 0.f;
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
