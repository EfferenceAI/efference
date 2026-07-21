////////////////////////////////////////////////////////////////////////////////
//
// File:      stream_assembler.hpp
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

#ifndef EF_STREAM_ASSEMBLER_HPP
#define EF_STREAM_ASSEMBLER_HPP

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "ef/Core.hpp"   // ef::ImuSample
#include "internal_status.hpp"

namespace ef {
namespace internal {

// ef_stream reassembly + consumer state, shared by every data-plane reader (USB isoc
// StreamReader, WiFi UdpStreamReader). Owns the wire-format parse (on_packet), the
// latest-wins video quad buffer, the IMU ring, and the consumer API (grab /
// current_video / drain_imu). A concrete reader feeds on_packet() from its transport
// and implements start()/stop(); one wire-format home means USB and UDP can't drift.
class StreamAssembler {
public:
    StreamAssembler() = default;
    virtual ~StreamAssembler() = default;
    StreamAssembler(const StreamAssembler&)            = delete;
    StreamAssembler& operator=(const StreamAssembler&) = delete;

    bool is_running() const { return running_.load(); }

    // Transport-specific teardown (cancel I/O, join thread, release resources).
    // Idempotent. Concrete readers also call this from their destructor.
    virtual void stop() = 0;

    // Block for the next frame. return_partial surfaces lossy frames as
    // CORRUPTED_FRAME. Returns SUCCESS | TIMEOUT | CORRUPTED_FRAME | END_OF_STREAM.
    Status grab(uint32_t timeout_ms, bool return_partial);

    // Frame held by grab() (valid until the next grab()): NV12 for RAW, else an
    // encoded access unit the Device decodes in retrieve_image().
    struct RawFrame {
        const uint8_t* data  = nullptr;
        size_t         size  = 0;
        uint32_t       frame_id = 0;
        uint64_t       ts_ns = 0;
        int            width = 0, height = 0;
        uint8_t        pixfmt = 0;
        bool           complete = false;   // false => transit loss (CORRUPTED_FRAME)
    };
    bool current_video(RawFrame* out) const;

    // Drain IMU samples since the last call. latest_only keeps only the newest;
    // *dropped (optional) gets the ring-overrun count.
    void drain_imu(std::vector<ImuSample>& out, bool latest_only, uint64_t* dropped);

    // Copy the newest IMU sample's acceleration WITHOUT consuming the queue, so an
    // image-only consumer can resolve FLIP_MODE::AUTO from gravity without stealing
    // samples a later drain_imu() owes the recorder. False if none yet.
    bool peek_latest_accel(float out[3]);

    // Codec of the video stream, so the drop-until-IDR gate can classify keyframes.
    // 0 = RAW/MJPEG (intra, never gated), 1 = H264, 2 = H265. Set once by the Device
    // before the stream flows.
    void set_video_codec(int codec) {
        vcodec_.store(codec, std::memory_order_relaxed);
    }

protected:
    // Feed one ef_stream packet (8-B common + type header + payload). Thread-safe
    // vs the consumer. Called from the concrete reader's transport thread.
    void on_packet(const uint8_t* b, int len);

    // Wake a blocked grab() because the transport ended (device gone / socket
    // closed / fatal error). After this, grab() returns END_OF_STREAM.
    void mark_ended();

    // Called (transport thread, no lock) when on_packet() detects a video seq gap
    // (wire loss). A transport with a back channel (UDP) overrides this to request a
    // keyframe (PLI); default no-op (USB isoc recovers at the next GOP IDR).
    virtual void on_loss() {}

    int  free_vbuf() const;    // a slot not held by consumer/ready/reassembly
    void grow_vbufs(size_t need);

    // ---- worker thread + run state (started/owned by the concrete reader) ----
    std::thread       thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_{false};

    // ---- video quad buffer (producer=transport thread, consumer=grab) ----
    static constexpr int VBUFS = 4;
    mutable std::mutex           vmtx_;
    std::condition_variable      vcv_;
    std::vector<uint8_t>         vbuf_[VBUFS];
    size_t                       vlen_[VBUFS]   = {0,0,0,0};
    RawFrame                     vmeta_[VBUFS];        // metadata per slot
    int                          reasm_ = -1;          // slot being reassembled
    int                          ready_ = -1;          // complete frame awaiting grab
    int                          consuming_ = -1;      // slot held by grab()/retrieve
    bool                         ended_ = false;       // transport gone / fatal
    uint64_t                     frames_dropped_ = 0;  // superseded before grab

    // reassembly progress for the current in-flight frame
    bool     in_frame_ = false;
    uint32_t cur_frame_id_ = 0, cur_size_ = 0, cur_accum_ = 0;

    // video packet-seq tracking for loss detection (transport-thread only, no lock)
    uint32_t last_vseq_ = 0;
    bool     have_vseq_ = false;

    // Drop-until-IDR resync gate (encoded streams only). Set on wire loss or consumer
    // supersede; while set, on_packet withholds every complete frame until an IDR/IRAP
    // resets the decoder reference chain. Starts true so the first delivered frame is a
    // keyframe. Transport-thread only, no lock.
    bool     resync_pending_ = true;
    // Video codec for keyframe classification (0=RAW/MJPEG, 1=H264, 2=H265).
    // Written by the Device before streaming, read on the transport thread.
    std::atomic<int> vcodec_{0};

    // ---- IMU ring ----
    std::mutex             imtx_;
    std::deque<ImuSample>  imu_;
    uint64_t               imu_dropped_ = 0;
    static constexpr size_t IMU_CAP = 8192;
};

}  // namespace internal
}  // namespace ef

#endif  // EF_STREAM_ASSEMBLER_HPP
