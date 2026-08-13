////////////////////////////////////////////////////////////////////////////////
//
// File:      device_stream.cpp
// Purpose:   ef::Device data plane: grab / retrieve_image / retrieve_imu.
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

#include "internal/device_impl.hpp"

namespace ef {

// ---- data plane -----------------------------------------------------------------

ERROR_CODE Device::grab(RuntimeParameters params) {
    if (!impl_) return ERROR_CODE::INVALID_FUNCTION_CALL;   // moved-from handle
    ERROR_CODE se = impl_->ensure_streaming();   // start the data plane on first grab
    if (se != ERROR_CODE::SUCCESS) return se;
    auto reader = impl_->reader_snapshot();   // pins the assembler across the block
    if (impl_->state != DEVICE_STATE::STREAMING || !reader)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    impl_->runtime = params;
    uint32_t to      = params.timeout_ms ? params.timeout_ms : impl_->init.grab_timeout_ms;
    bool     partial = params.return_partial || !impl_->init.drop_partial_frames;
    ERROR_CODE ec = grab_err(reader->grab(to, partial));
    // MCAP replay: the file ran out, END_OF_BUFFER, not a transport fault.
    if (ec == ERROR_CODE::COMMUNICATION_ERROR &&
        impl_->init.input_type == INPUT_TYPE::MCAP)
        ec = ERROR_CODE::END_OF_BUFFER;
    auto hr = impl_->host_rec_snapshot();   // pins the recorder across the tee
    if (ec == ERROR_CODE::SUCCESS || ec == ERROR_CODE::CORRUPTED_FRAME) {
        internal::StreamAssembler::RawFrame rf;
        if (reader->current_video(&rf)) {
            impl_->last_frame_ts_ns = rf.ts_ns;
            // Stamp the .mcap with this frame's geometry, not the device's last
            // persisted config, they differ when the opened res != saved config.
            if (hr) hr->push_video(rf.data, rf.size, rf.ts_ns, rf.width, rf.height);
        }
    }
    // While recording, tee IMU every grab so the .mcap carries all samples even if
    // the app never calls retrieve_imu; they queue in the backlog for next time.
    if (hr && impl_->init.enable_imu) {
        std::vector<ImuSample> tmp;
        uint64_t d = 0;
        impl_->drain_imu_tee(tmp, &d, hr);
        std::lock_guard<std::mutex> lk(impl_->data_mtx);
        impl_->imu_backlog_dropped += d;
        if (impl_->imu_backlog.size() + tmp.size() <= Impl::IMU_BACKLOG_CAP)
            impl_->imu_backlog.insert(impl_->imu_backlog.end(), tmp.begin(), tmp.end());
        else
            impl_->imu_backlog_dropped += tmp.size();
    }
    return ec;
}

ERROR_CODE Device::retrieve_image(Mat& mat, VIEW view) {
    if (!impl_) return ERROR_CODE::INVALID_FUNCTION_CALL;   // moved-from handle
    auto reader = impl_->reader_snapshot();
    if (impl_->state != DEVICE_STATE::STREAMING || !reader)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    internal::StreamAssembler::RawFrame rf;
    if (!reader->current_video(&rf)) return ERROR_CODE::INVALID_FUNCTION_CALL;

    mat = Mat{};
    mat.frame_id_          = rf.frame_id;
    mat.timestamp_.data_ns = rf.ts_ns;
    mat.resolution_        = {rf.width, rf.height};
    mat.view_              = view;
    mat.type_              = mat_type_of(view);
    mat.memory_            = MEM::CPU;

    const bool raw = (pb_codec(impl_->init.compression) == ef_v1_Codec_CODEC_RAW);

    // RAW plane layout comes from wire width/height, independent of payload size;
    // reject a frame whose NV12 planes would over-read the buffer before any copy.
    if (raw) {
        const size_t need = (size_t)rf.width * rf.height +
                            (size_t)rf.width * ((rf.height + 1) / 2);
        if (rf.width <= 0 || rf.height <= 0 || need > rf.size)
            return ERROR_CODE::CORRUPTED_FRAME;
    }

    // FLIP_MODE::AUTO for an image-only consumer (no retrieve_imu / recording to
    // run the tee): resolve the gravity latch here with a non-consuming peek.
    if (impl_->init.flip_mode == FLIP_MODE::AUTO && impl_->flip_latched < 0 &&
        impl_->init.enable_imu) {
        float a[3];
        if (reader->peek_latest_accel(a)) impl_->latch_flip_from_accel(a);
    }

    // RAW + NV12: one copy into the owning buffer, no decode.
    if (raw && view == VIEW::NV12) {
        mat.step_ = rf.width;
        mat.data_.assign(rf.data, rf.data + rf.size);
        if (impl_->effective_flip()) mat.flip180();
        return ERROR_CODE::SUCCESS;
    }

#ifdef EF_WITH_FFMPEG
    if (raw) {
        // NV12 -> requested packed/gray format via swscale.
        const uint8_t* src[4] = { rf.data, rf.data + (size_t)rf.width * rf.height,
                                  nullptr, nullptr };
        int srcstride[4] = { rf.width, rf.width, 0, 0 };
        int step = 0;
        if (!impl_->decoder.convert(src, srcstride, rf.width, rf.height,
                                    AV_PIX_FMT_NV12, view, mat.data_, &step))
            return ERROR_CODE::CORRUPTED_FRAME;
        mat.step_ = step;
        if (impl_->effective_flip()) mat.flip180();
        return ERROR_CODE::SUCCESS;
    }
    // Encoded (H264/H265): decode one access unit, then colour-convert.
    if (!impl_->decoder.ensure(impl_->init.compression)) return ERROR_CODE::CORRUPTED_FRAME;
    AVPacket* pkt = impl_->decoder.pkt;
    av_packet_unref(pkt);
    pkt->data = const_cast<uint8_t*>(rf.data);
    pkt->size = (int)rf.size;
    // On decode failure, flush reference state so a later frame that lost its refs
    // (e.g. superseded while a consumer stalled) doesn't keep erroring; it resyncs
    // at the next keyframe (assembler's PLI-on-supersede already requests it over UDP).
    int sent = avcodec_send_packet(impl_->decoder.dec, pkt);
    if (sent < 0 && sent != AVERROR(EAGAIN)) {
        avcodec_flush_buffers(impl_->decoder.dec);
        return ERROR_CODE::CORRUPTED_FRAME;
    }
    AVFrame* frm = impl_->decoder.frm;
    int rc = avcodec_receive_frame(impl_->decoder.dec, frm);
    // Frame-threaded decoding pipelines the first few packets and returns
    // EAGAIN until the pipeline fills. That is "no frame YET", not an error:
    // flushing here would restart the priming and no frame would ever surface.
    // Surface it as the non-fatal GRAB_TIMEOUT ("retry next grab") instead.
    if (rc == AVERROR(EAGAIN)) return ERROR_CODE::GRAB_TIMEOUT;
    if (rc < 0) {
        avcodec_flush_buffers(impl_->decoder.dec);
        return ERROR_CODE::CORRUPTED_FRAME;
    }
    const uint8_t* src[4] = { frm->data[0], frm->data[1], frm->data[2], frm->data[3] };
    int srcstride[4] = { frm->linesize[0], frm->linesize[1], frm->linesize[2], frm->linesize[3] };
    int step = 0;
    if (!impl_->decoder.convert(src, srcstride, frm->width, frm->height,
                                (AVPixelFormat)frm->format, view, mat.data_, &step))
        return ERROR_CODE::CORRUPTED_FRAME;
    mat.resolution_ = {frm->width, frm->height};
    mat.step_       = step;
    if (impl_->effective_flip()) mat.flip180();
    return ERROR_CODE::SUCCESS;
#else
    (void)rf;
    // Built without FFmpeg: only the RAW -> NV12 path is available.
    return ERROR_CODE::UNSUPPORTED_COMPRESSION;
#endif
}

namespace {

// Apply InitParameters conventions to one IMU sample (both vectors): flip first
// (180 deg about optical z: negate x,y), then remap axes out of the IMAGE frame
// (x right, y down, z forward) into the requested system. Angular velocity gets
// the same remap as image/pose data, matching consumer camera SDKs.
void apply_imu_convention(ImuSample& s, bool flip, COORDINATE_SYSTEM cs) {
    auto remap = [flip, cs](float v[3]) {
        float x = v[0], y = v[1], z = v[2];
        if (flip) { x = -x; y = -y; }
        switch (cs) {
            case COORDINATE_SYSTEM::IMAGE:                                             break;
            case COORDINATE_SYSTEM::LEFT_HANDED_Y_UP:        v[0] = x; v[1] = -y; v[2] = z;  return;
            case COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP:       v[0] = x; v[1] = -y; v[2] = -z; return;
            case COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP:       v[0] = x; v[1] = z;  v[2] = -y; return;
            case COORDINATE_SYSTEM::LEFT_HANDED_Z_UP:        v[0] = z; v[1] = x;  v[2] = -y; return;
            case COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD: v[0] = z; v[1] = -x; v[2] = -y; return;
        }
        v[0] = x; v[1] = y; v[2] = z;
    };
    remap(s.acceleration);
    remap(s.angular_velocity);
}

}  // namespace

ERROR_CODE Device::get_stream_stats(StreamStats& out) const {
    if (!impl_) return ERROR_CODE::INVALID_FUNCTION_CALL;   // moved-from handle
    auto reader = impl_->reader_snapshot();
    if (!reader) return ERROR_CODE::INVALID_FUNCTION_CALL;
    reader->get_stats(&out);
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::retrieve_imu(SensorsData& data, TIME_REFERENCE ref) {
    if (!impl_) return ERROR_CODE::INVALID_FUNCTION_CALL;   // moved-from handle
    auto reader = impl_->reader_snapshot();
    if (impl_->state != DEVICE_STATE::STREAMING || !reader)
        return ERROR_CODE::INVALID_FUNCTION_CALL;

    // Backlog first (samples grab() drained for the recorder), then fresh ones,
    // preserving chronological order.
    auto hr = impl_->host_rec_snapshot();
    std::vector<ImuSample> fresh;
    uint64_t d = 0;
    impl_->drain_imu_tee(fresh, &d, hr);
    {
        std::lock_guard<std::mutex> lk(impl_->data_mtx);
        data.samples = std::move(impl_->imu_backlog);
        impl_->imu_backlog.clear();
        data.dropped = impl_->imu_backlog_dropped + d;
        impl_->imu_backlog_dropped = 0;
    }
    data.samples.insert(data.samples.end(), fresh.begin(), fresh.end());
    if (ref == TIME_REFERENCE::CURRENT && data.samples.size() > 1)
        data.samples.erase(data.samples.begin(), data.samples.end() - 1);

    // Coarse motion from the newest sample: free fall reads |a|~0, static reads
    // |a|~g with a quiet gyro. (Both magnitudes are invariant under the remap below.)
    data.motion_state = MOTION_STATE::MOVING;
    if (!data.samples.empty()) {
        const ImuSample& s = data.samples.back();
        double a = std::sqrt((double)s.acceleration[0] * s.acceleration[0] +
                             (double)s.acceleration[1] * s.acceleration[1] +
                             (double)s.acceleration[2] * s.acceleration[2]);
        double w = std::sqrt((double)s.angular_velocity[0] * s.angular_velocity[0] +
                             (double)s.angular_velocity[1] * s.angular_velocity[1] +
                             (double)s.angular_velocity[2] * s.angular_velocity[2]);
        if (a < 3.0)                                data.motion_state = MOTION_STATE::FALLING;
        else if (std::fabs(a - 9.81) < 0.5 && w < 0.05) data.motion_state = MOTION_STATE::STATIC;
    }

    // InitParameters::flip_mode / ::coordinate_system, applied at delivery
    // (recordings keep the raw IMAGE-frame data, like the device's own files).
    const bool flip = impl_->effective_flip();
    const COORDINATE_SYSTEM cs = impl_->init.coordinate_system;
    if (flip || cs != COORDINATE_SYSTEM::IMAGE)
        for (ImuSample& s : data.samples) apply_imu_convention(s, flip, cs);
    return ERROR_CODE::SUCCESS;
}

Timestamp Device::get_timestamp(TIME_REFERENCE ref) const {
    Timestamp t;
    t.data_ns = (ref == TIME_REFERENCE::IMAGE)
                    ? (impl_ ? impl_->last_frame_ts_ns : 0)
                    : host_now_ns();
    return t;
}

}  // namespace ef
