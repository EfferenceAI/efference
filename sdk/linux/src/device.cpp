////////////////////////////////////////////////////////////////////////////////
//
// File:      device.cpp
// Purpose:   ef::Device implementation.
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

#include "ef/Device.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <libusb-1.0/libusb.h>

// The transport / reader internals return ef::Status, the fine-grained result
// set of the internal ISOC implementation (Status left the public headers when
// the API went ERROR_CODE-only; it now lives in internal_status.hpp).
// Translated to ERROR_CODE at this layer; nothing above speaks Status.
#include "internal_status.hpp"

#include "connection.hpp"
#include "mcap.hpp"
#include "mcap_replay_reader.hpp"
#include "stream_reader.hpp"
#include "udp_stream_reader.hpp"
#include "usb_connection.hpp"
#ifdef EF_HAVE_BLE
#include <openssl/evp.h>
#include <openssl/hmac.h>

#include "ble_connection.hpp"
#endif

#ifdef EF_WITH_FFMPEG
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}
#endif

// Protobuf control plane (nanopb), the shared ef.proto wire with the device.
#include "ef.pb.h"
#include "pb_decode.h"
#include "pb_encode.h"

namespace ef {

// Readable aliases for the C types nanopb generates from proto/ef.proto
// (package ef.v1 -> ef_v1_*). Enum values, body tags, and the _init_zero /
// _fields macros keep their generated names. The generated names map directly
// to proto/ef.proto.
using WireRequest       = ef_v1_Request;
using WireResponse      = ef_v1_Response;
using WireErrorCode     = ef_v1_ErrorCode;
using WireCodec         = ef_v1_Codec;
using WireUploadState   = ef_v1_UploadState;
using WireDeviceInfo    = ef_v1_DeviceInformation;
using WireHealth        = ef_v1_HealthStatus;
using WireRecording     = ef_v1_RecordingStatus;
using WireRecordingList = ef_v1_RecordingList;
using WireUpload        = ef_v1_UploadStatus;
using WireOta           = ef_v1_OtaStatus;
using WireWifi          = ef_v1_WifiStatus;

// ---- free functions ---------------------------------------------------------

Resolution get_resolution(RESOLUTION r) {
    switch (r) {
        case RESOLUTION::HD1200: return {1920, 1200};
        case RESOLUTION::HD1080: return {1920, 1080};
        case RESOLUTION::SVGA:   return {960,  600};
        case RESOLUTION::AUTO:   return {0, 0};
    }
    return {0, 0};
}

namespace {

// The M1's USB identity (vendor interface; same IDs the connection classes latch).
constexpr uint16_t kVid = 0x39c5;
constexpr uint16_t kPid = 0x0001;

uint64_t host_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ---- wire <-> enum maps ------------------------------------------------------

WireCodec pb_codec(COMPRESSION_MODE c) {
    switch (c) {
        case COMPRESSION_MODE::H264:
        case COMPRESSION_MODE::H264_HQ: return ef_v1_Codec_H264;
        case COMPRESSION_MODE::H265:
        case COMPRESSION_MODE::H265_HQ: return ef_v1_Codec_H265;
        case COMPRESSION_MODE::RAW:     return ef_v1_Codec_CODEC_RAW;
    }
    return ef_v1_Codec_CODEC_RAW;
}

// The _HQ presets ride the same wire codec plus an explicit encoder quality.
// 90 ≈ perceptually lossless on the device encoder, a named constant so the
// two presets can't drift apart; tune here if the target moves.
constexpr uint32_t kHqQuality = 90;

uint32_t quality_for(COMPRESSION_MODE c) {
    return (c == COMPRESSION_MODE::H264_HQ || c == COMPRESSION_MODE::H265_HQ)
               ? kHqQuality
               : 0;   // 0 = device default
}

// HQ never round-trips: the wire carries codec and quality separately and the
// device reports the codec only, so a read-back config collapses to the base
// mode. The requested mode survives in the InitParameters cache.
COMPRESSION_MODE compression_from(WireCodec c) {
    switch (c) {
        case ef_v1_Codec_H264: return COMPRESSION_MODE::H264;
        case ef_v1_Codec_H265: return COMPRESSION_MODE::H265;
        default:               return COMPRESSION_MODE::RAW;
    }
}

// What kind of request produced a device error. The wire enum is smaller than
// ERROR_CODE and context-dependent (call-site error translation: errors map at
// the call site, so every public method returns a small documented subset).
enum class Ctx {
    CONTROL,     // queries, wifi, health, time, reboot
    SESSION,     // configure / start_stream during open()
    RECORDING,   // device-local recording verbs (incl. download)
    UPLOAD,      // upload verbs
    UPDATE,      // OTA verbs
};

// Device ErrorCode + request context -> ERROR_CODE. The device's message
// string (Response.message) still carries the specific reason.
ERROR_CODE err_from(WireErrorCode c, Ctx ctx) {
    // Codes with one meaning regardless of what was asked.
    switch (c) {
        case ef_v1_ErrorCode_SUCCESS:                  return ERROR_CODE::SUCCESS;
        case ef_v1_ErrorCode_NOT_OPENED:               return ERROR_CODE::DEVICE_NOT_INITIALIZED;
        case ef_v1_ErrorCode_CORRUPTED_FRAME:          return ERROR_CODE::CORRUPTED_FRAME;
        case ef_v1_ErrorCode_BANDWIDTH_EXCEEDED:
        case ef_v1_ErrorCode_NOT_SUPERSPEED:           return ERROR_CODE::LOW_USB_BANDWIDTH;
        case ef_v1_ErrorCode_CAMERA_UNAVAILABLE:       return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        case ef_v1_ErrorCode_ORCHESTRATOR_UNREACHABLE: return ERROR_CODE::DEVICE_NOT_AVAILABLE;
        case ef_v1_ErrorCode_STORAGE_FULL:             return ERROR_CODE::STORAGE_FULL;
        // Name conflict on record start (recording-specific, same meaning in any ctx).
        case ef_v1_ErrorCode_ALREADY_EXISTS:           return ERROR_CODE::RECORDING_ALREADY_EXISTS;
        // Whatever was asked, the gate that failed is the BLE password.
        case ef_v1_ErrorCode_AUTH_REQUIRED:
        case ef_v1_ErrorCode_AUTH_FAILED:              return ERROR_CODE::INVALID_PASSWORD;
        // A control request that timed out means the device stopped answering
        // (grab timeouts come from the host-side reader, never from here).
        case ef_v1_ErrorCode_TIMEOUT:
            return ctx == Ctx::UPDATE ? ERROR_CODE::FAILED_TO_UPDATE
                                      : ERROR_CODE::COMMUNICATION_ERROR;
        default: break;
    }

    // The rest depend on what was asked for.
    switch (ctx) {
        case Ctx::SESSION:
            // The host pre-validates against the capability menu, so a device-
            // side rejection here is a mode/codec the menu didn't rule out.
            switch (c) {
                case ef_v1_ErrorCode_INVALID_PARAMETER: return ERROR_CODE::INVALID_RESOLUTION;
                case ef_v1_ErrorCode_UNSUPPORTED:       return ERROR_CODE::UNSUPPORTED_COMPRESSION;
                case ef_v1_ErrorCode_BUSY:
                case ef_v1_ErrorCode_INVALID_STATE:     return ERROR_CODE::INVALID_FUNCTION_CALL;
                default:                                return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
            }
        case Ctx::RECORDING:
            switch (c) {
                case ef_v1_ErrorCode_NOT_FOUND:         return ERROR_CODE::RECORDING_NOT_FOUND;
                case ef_v1_ErrorCode_INVALID_PARAMETER:
                case ef_v1_ErrorCode_BUSY:
                case ef_v1_ErrorCode_INVALID_STATE:     return ERROR_CODE::INVALID_FUNCTION_CALL;
                default:                                return ERROR_CODE::SESSION_RECORDING_ERROR;
            }
        case Ctx::UPLOAD:
            switch (c) {
                case ef_v1_ErrorCode_NOT_FOUND:         return ERROR_CODE::RECORDING_NOT_FOUND;
                // Unmet precondition, not a broken link: provision WiFi first.
                case ef_v1_ErrorCode_WIFI_NOT_CONNECTED: return ERROR_CODE::WIFI_NOT_CONNECTED;
                case ef_v1_ErrorCode_INVALID_PARAMETER:
                case ef_v1_ErrorCode_BUSY:
                case ef_v1_ErrorCode_INVALID_STATE:     return ERROR_CODE::INVALID_FUNCTION_CALL;
                default:                                return ERROR_CODE::UNKNOWN_FAILURE;
            }
        case Ctx::UPDATE:
            switch (c) {
                case ef_v1_ErrorCode_BUSY:
                case ef_v1_ErrorCode_INVALID_STATE:     return ERROR_CODE::INVALID_FUNCTION_CALL;
                default:                                return ERROR_CODE::FAILED_TO_UPDATE;
            }
        case Ctx::CONTROL:
        default:
            switch (c) {
                case ef_v1_ErrorCode_INVALID_PARAMETER:
                case ef_v1_ErrorCode_UNSUPPORTED:
                case ef_v1_ErrorCode_BUSY:
                case ef_v1_ErrorCode_INVALID_STATE:
                case ef_v1_ErrorCode_NOT_FOUND:
                case ef_v1_ErrorCode_WIFI_NOT_CONNECTED: return ERROR_CODE::INVALID_FUNCTION_CALL;
                default:                                 return ERROR_CODE::UNKNOWN_FAILURE;
            }
    }
}

// ---- internal Status -> ERROR_CODE (transport bring-up / data plane) ----------

// Connection / reader bring-up failures, mapped for open().
ERROR_CODE open_err(Status s) {
    switch (s) {
        case Status::SUCCESS:          return ERROR_CODE::SUCCESS;
        case Status::DEVICE_NOT_FOUND: return ERROR_CODE::DEVICE_NOT_DETECTED;
        case Status::NO_STREAM:        return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        // libusb EACCES on open/claim, the actionable udev/permissions case.
        case Status::INSUFFICIENT_PERMISSIONS: return ERROR_CODE::INSUFFICIENT_PERMISSIONS;
        default:                       return ERROR_CODE::DEVICE_NOT_AVAILABLE;
    }
}

// The wire collapses every bad Configure knob into INVALID_PARAMETER, which the
// SESSION context turns into INVALID_RESOLUTION. The device's reason string
// (resp.message) names the real culprit, "codec", "fps", or a geometry/tuple
// issue ("resolution"/"width"/"height"/"camera:..."). Re-split it so INVALID_FPS
// and UNSUPPORTED_COMPRESSION are reachable and INVALID_RESOLUTION is the
// geometry case, not the catch-all that used to mask fps. (Currently inert:
// reachable only once the device rejects out-of-caps tuples with INVALID_PARAMETER.)
ERROR_CODE disambiguate_config_error(ERROR_CODE ec, const WireResponse& resp) {
    if (ec != ERROR_CODE::INVALID_RESOLUTION ||
        resp.code != ef_v1_ErrorCode_INVALID_PARAMETER)
        return ec;
    if (std::strstr(resp.message, "codec")) return ERROR_CODE::UNSUPPORTED_COMPRESSION;
    if (std::strstr(resp.message, "fps"))   return ERROR_CODE::INVALID_FPS;
    return ERROR_CODE::INVALID_RESOLUTION;
}

// StreamAssembler::grab results. TIMEOUT is the non-fatal keep-looping case;
// END_OF_STREAM means the transport died underneath the session.
ERROR_CODE grab_err(Status s) {
    switch (s) {
        case Status::SUCCESS:         return ERROR_CODE::SUCCESS;
        case Status::TIMEOUT:         return ERROR_CODE::GRAB_TIMEOUT;
        case Status::CORRUPTED_FRAME: return ERROR_CODE::CORRUPTED_FRAME;
        default:                      return ERROR_CODE::COMMUNICATION_ERROR;
    }
}

// ---- decode + colour convert (retrieve_image) --------------------------------

#ifdef EF_WITH_FFMPEG
AVPixelFormat av_of(VIEW v) {
    switch (v) {
        case VIEW::NV12: return AV_PIX_FMT_NV12;
        case VIEW::BGR:  return AV_PIX_FMT_BGR24;
        case VIEW::RGB:  return AV_PIX_FMT_RGB24;
        case VIEW::BGRA: return AV_PIX_FMT_BGRA;
        case VIEW::RGBA: return AV_PIX_FMT_RGBA;
        case VIEW::GRAY: return AV_PIX_FMT_GRAY8;
    }
    return AV_PIX_FMT_NV12;
}
#endif

MAT_TYPE mat_type_of(VIEW v) {
    switch (v) {
        case VIEW::GRAY: return MAT_TYPE::U8_C1;
        case VIEW::BGR:
        case VIEW::RGB:  return MAT_TYPE::U8_C3;
        case VIEW::BGRA:
        case VIEW::RGBA: return MAT_TYPE::U8_C4;
        case VIEW::NV12: return MAT_TYPE::NV12;
    }
    return MAT_TYPE::NV12;
}

// Decode (encoded codecs) + colour-convert (swscale) engine. Without FFmpeg
// only the RAW -> VIEW::NV12 path is served.
struct VideoDecoder {
#ifdef EF_WITH_FFMPEG
    AVCodecContext* dec = nullptr;
    AVPacket*       pkt = nullptr;
    AVFrame*        frm = nullptr;
    SwsContext*     sws = nullptr;
    int             sw = 0, sh = 0, sfmt = -1, dfmt = -1;
    COMPRESSION_MODE cur = COMPRESSION_MODE::RAW;
    bool            have = false;

    ~VideoDecoder() { reset(); }
    void reset() {
        if (sws) { sws_freeContext(sws); sws = nullptr; }
        if (frm) av_frame_free(&frm);
        if (pkt) av_packet_free(&pkt);
        if (dec) avcodec_free_context(&dec);
        have = false;
    }
    bool ensure(COMPRESSION_MODE c) {
        if (have && cur == c) return true;
        reset();
        AVCodecID id = (pb_codec(c) == ef_v1_Codec_H264) ? AV_CODEC_ID_H264
                     : (pb_codec(c) == ef_v1_Codec_H265) ? AV_CODEC_ID_HEVC
                                                         : AV_CODEC_ID_NONE;
        if (id == AV_CODEC_ID_NONE) return false;
        const AVCodec* codec = avcodec_find_decoder(id);
        if (!codec) return false;
        dec = avcodec_alloc_context3(codec);
        if (!dec) return false;
        if (avcodec_open2(dec, codec, nullptr) < 0) { avcodec_free_context(&dec); return false; }
        pkt = av_packet_alloc(); frm = av_frame_alloc();
        cur = c; have = pkt && frm;
        return have;
    }
    bool convert(const uint8_t* const src[4], const int srcstride[4], int w, int h,
                 AVPixelFormat srcfmt, VIEW view, std::vector<uint8_t>& buf, int* step) {
        AVPixelFormat df = av_of(view);
        if (!sws || sw != w || sh != h || sfmt != (int)srcfmt || dfmt != (int)df) {
            if (sws) sws_freeContext(sws);
            sws = sws_getContext(w, h, srcfmt, w, h, df, SWS_BILINEAR, nullptr, nullptr, nullptr);
            sw = w; sh = h; sfmt = (int)srcfmt; dfmt = (int)df;
        }
        if (!sws) return false;
        int need = av_image_get_buffer_size(df, w, h, 1);
        if (need <= 0) return false;
        buf.resize((size_t)need);
        uint8_t* dst[4]; int dstr[4];
        av_image_fill_arrays(dst, dstr, buf.data(), df, w, h, 1);
        sws_scale(sws, src, srcstride, 0, h, dst, dstr);
        *step = dstr[0];
        return true;
    }
#endif
};

// Host-side recorder: bounded queue + writer thread so a slow disk never
// back-pressures the grab loop. Writes a real MCAP container (video + IMU) in
// the exact schema/topic layout of a device-local recording (see mcap.hpp), so
// the HOST_FILE .mcap promised by RecordingParameters is honoured and both
// recording targets replay through the same MCAP reader.
struct HostRecorder {
    struct Item {
        bool                 imu = false;
        std::vector<uint8_t> data;   // frame bytes, or one packed ImuSample
        ImuSample            sample;
        uint64_t             ts_ns = 0;
        int                  w = 0, h = 0;   // per-frame geometry (RAW frames)
    };
    internal::mcap::Writer writer;
    std::thread            thr;
    std::mutex             mtx;
    std::condition_variable cv;
    std::deque<Item>       q;
    bool                   run = false;
    bool                   raw = false;      // RAW/NV12 vs encoded access units
    std::string            fmt;              // "h265"/"h264"/"nv12"
    int                    width = 0, height = 0;
    uint64_t               bytes = 0, frames = 0, dropped = 0;
    uint64_t               t0_ns = 0;
    static constexpr size_t QCAP = 16;

    bool start(const std::string& path, bool raw_, const std::string& fmt_,
               int w, int h) {
        raw    = raw_;
        fmt    = fmt_;
        width  = w;
        height = h;
        if (!writer.open(path)) return false;
        size_t n = 0;
        const unsigned char* fds;
        namespace mc = internal::mcap;
        fds = mc::fds_compressed_video(&n);
        writer.add_schema(mc::SCH_VIDEO, "foxglove.CompressedVideo", fds, n);
        fds = mc::fds_raw_image(&n);
        writer.add_schema(mc::SCH_RAW, "foxglove.RawImage", fds, n);
        fds = mc::fds_pose_in_frame(&n);
        writer.add_schema(mc::SCH_POSE, "foxglove.PoseInFrame", fds, n);
        writer.add_channel(mc::CH_IMAGE, mc::SCH_VIDEO, mc::TOPIC_IMAGE);
        writer.add_channel(mc::CH_IMAGE_RAW, mc::SCH_RAW, mc::TOPIC_IMAGE_RAW);
        writer.add_channel(mc::CH_ACCEL, mc::SCH_POSE, mc::TOPIC_ACCEL);
        writer.add_channel(mc::CH_GYRO, mc::SCH_POSE, mc::TOPIC_GYRO);
        if (!writer.ok()) return false;

        run   = true;
        t0_ns = host_now_ns();
        thr = std::thread([this] {
            namespace mc = internal::mcap;
            std::vector<uint8_t> msg;
            for (;;) {
                Item item;
                {
                    std::unique_lock<std::mutex> lk(mtx);
                    cv.wait(lk, [this] { return !q.empty() || !run; });
                    if (q.empty() && !run) break;
                    item = std::move(q.front());
                    q.pop_front();
                }
                if (item.imu) {
                    const ImuSample& s = item.sample;
                    mc::enc_pose_in_frame(msg, s.timestamp.data_ns, mc::FRAME_IMU,
                                          s.acceleration[0], s.acceleration[1],
                                          s.acceleration[2]);
                    writer.write_message(mc::CH_ACCEL, s.timestamp.data_ns,
                                         msg.data(), msg.size());
                    mc::enc_pose_in_frame(msg, s.timestamp.data_ns, mc::FRAME_IMU,
                                          s.angular_velocity[0], s.angular_velocity[1],
                                          s.angular_velocity[2]);
                    writer.write_message(mc::CH_GYRO, s.timestamp.data_ns,
                                         msg.data(), msg.size());
                } else if (raw) {
                    // Per-frame geometry when the pusher supplied it, else the
                    // resolution start() was opened with.
                    uint32_t fw = item.w > 0 ? (uint32_t)item.w : (uint32_t)width;
                    uint32_t fh = item.h > 0 ? (uint32_t)item.h : (uint32_t)height;
                    mc::enc_raw_image(msg, item.ts_ns, mc::FRAME_COLOR,
                                      fw, fh, "nv12",
                                      fw, item.data.data(),
                                      item.data.size());
                    writer.write_message(mc::CH_IMAGE_RAW, item.ts_ns,
                                         msg.data(), msg.size());
                } else {
                    mc::enc_compressed_video(msg, item.ts_ns, mc::FRAME_COLOR,
                                             fmt.c_str(), item.data.data(),
                                             item.data.size());
                    writer.write_message(mc::CH_IMAGE, item.ts_ns,
                                         msg.data(), msg.size());
                }
            }
        });
        return true;
    }
    void push_video(const uint8_t* d, size_t n, uint64_t ts_ns, int w = 0, int h = 0) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!run) return;
        if (q.size() >= QCAP) { dropped++; return; }
        Item it;
        it.data.assign(d, d + n);
        it.ts_ns = ts_ns ? ts_ns : host_now_ns();
        it.w = w;
        it.h = h;
        q.push_back(std::move(it));
        bytes += n;
        frames++;
        cv.notify_one();
    }
    void push_imu(const ImuSample& s) {
        std::lock_guard<std::mutex> lk(mtx);
        if (!run) return;
        if (q.size() >= QCAP * 16) { dropped++; return; }   // IMU items are tiny
        Item it;
        it.imu    = true;
        it.sample = s;
        q.push_back(std::move(it));
        cv.notify_one();
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(mtx);
            if (!run && !thr.joinable()) return;
            run = false;
        }
        cv.notify_all();
        if (thr.joinable()) thr.join();
        writer.finish();   // DataEnd + Footer + closing magic
    }
    ~HostRecorder() { stop(); }
};

// Internal copy of the device's advertised capability menu (validation only).
struct Caps {
    struct Mode { int w = 0, h = 0, fps = 0; bool usable = false; };
    std::vector<Mode>        modes;
    std::vector<std::string> codecs;
};

}  // namespace

// ---- Impl ---------------------------------------------------------------------

struct Device::Impl {
    std::unique_ptr<internal::Connection> connection;
    // Read on the data-plane thread (grab/retrieve_*) and written on the control
    // thread (open/close/update/refresh), atomic so those reads never tear.
    std::atomic<DEVICE_STATE> state{DEVICE_STATE::CLOSED};

    InitParameters      init;
    RuntimeParameters   runtime;       // last grab() knobs
    RecordingParameters recording;     // last enable_recording() params

    DeviceInformation info;            // cached at open()
    HealthStatus      health;          // last completed sweep
    Caps              caps;            // internal validation menu

    // USB isoc StreamReader, WiFi UdpStreamReader, or MCAP replay, one
    // reassembly core; the consumer API (grab/current_video/drain_imu) is
    // shared. shared_ptr + reader_mtx so a grab() blocked inside the assembler
    // on one thread can never have the object deleted underneath it by
    // update()/close()/reboot() on another: teardown swaps the pointer out
    // under the lock and stop() wakes the waiter; the last snapshot holder
    // frees it (see stop_streaming / reader_snapshot).
    std::shared_ptr<internal::StreamAssembler> reader;
    mutable std::mutex            reader_mtx;
    VideoDecoder                  decoder;
    // shared_ptr + data_mtx (below), same discipline as `reader`: a grab() teeing
    // into the recorder pins it with a snapshot so close()/update() resetting it
    // on the control thread frees the object only once no data-plane call holds it.
    std::shared_ptr<HostRecorder> host_rec;
    std::string                   device_rec_name;   // active DEVICE_LOCAL session
    bool                          device_rec = false;

    // Raw device activity facets, refreshed by refresh_device_state() alongside
    // the public `state`. These preserve the recording-vs-uploading distinction
    // the public 5-value DEVICE_STATE collapses into STREAMING, so the illegal-
    // state guards (health/record/update) stay precise without a public flag.
    bool                          dev_recording = false;  // device-side capture in flight (FSM COLLECT / device_state RECORDING)
    bool                          dev_uploading = false;  // device-side upload in flight (device_state UPLOADING)

    Resolution stream_res{};   // validated at open(), applied when streaming starts

    // IMU drained by grab() while a host recording runs (the recorder must see
    // every sample even if the app never calls retrieve_imu); served to the
    // next retrieve_imu ahead of fresh samples. Consumer-thread only.
    std::vector<ImuSample> imu_backlog;
    uint64_t               imu_backlog_dropped = 0;
    static constexpr size_t IMU_BACKLOG_CAP = 65536;

    // FLIP_MODE::AUTO latch: -1 unresolved, 0 upright, 1 mounted upside-down.
    // Resolved from gravity in the raw (IMAGE-frame) accelerometer stream.
    int flip_latched = -1;

    uint64_t last_frame_ts_ns = 0;
    uint32_t corr = 0;

    // Serializes control round trips against transport teardown: libusb_close
    // (or the BlueZ teardown) under an in-flight request is undefined
    // behaviour, so close_transport() takes this too. Multi-round-trip loops
    // (download_recording, ota_push, polls) re-take it per call(), so a
    // concurrent close() aborts them at the next chunk boundary with
    // DEVICE_NOT_INITIALIZED instead of crashing.
    std::mutex ctl_mtx;

    std::shared_ptr<internal::StreamAssembler> reader_snapshot() const {
        std::lock_guard<std::mutex> lk(reader_mtx);
        return reader;
    }

    // Guards the host_rec pointer (swapped out on teardown) and the IMU backlog
    // it feeds, so the data-plane thread and close()/update() can't race on them.
    std::mutex data_mtx;
    std::shared_ptr<HostRecorder> host_rec_snapshot() {
        std::lock_guard<std::mutex> lk(data_mtx);
        return host_rec;
    }
    // Detach and stop the recorder: swap the pointer out under the lock (so a
    // concurrent grab() snapshot can't be freed underneath it) then stop the
    // local copy; the last holder frees it. Safe to call with no recorder.
    void stop_host_rec() {
        std::shared_ptr<HostRecorder> hr;
        { std::lock_guard<std::mutex> lk(data_mtx); hr = std::move(host_rec); }
        if (hr) hr->stop();
    }

    // FLIP_MODE::AUTO: latch orientation from one near-static gravity reading.
    // IMAGE frame has +y down, so at rest the accelerometer reads ~ -g on y when
    // upright and ~ +g upside-down. Consumer-thread only (writes flip_latched).
    void latch_flip_from_accel(const float a[3]) {
        if (init.flip_mode != FLIP_MODE::AUTO || flip_latched >= 0) return;
        double mag = std::sqrt((double)a[0] * a[0] + (double)a[1] * a[1] +
                               (double)a[2] * a[2]);
        if (std::fabs(mag - 9.81) > 0.8) return;
        if      (a[1] < -5.f) flip_latched = 0;
        else if (a[1] >  5.f) flip_latched = 1;
    }

    // One protobuf round trip: encode -> transport -> decode -> code check.
    // `ctx` selects the call-site error translation; `expect`
    // (optional) additionally requires that response body tag.
    ERROR_CODE call(WireRequest& req, WireResponse& resp,
                    pb_size_t expect = 0, Ctx ctx = Ctx::CONTROL) {
        std::lock_guard<std::mutex> lk(ctl_mtx);
        if (!connection) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
        req.corr_id = ++corr;
        uint8_t buf[8192];   // == proto::MAX_PAYLOAD (EFR_CTL_MSG_MAX): an
                             // OtaPushChunk request encodes to ~7.2 KB.
        pb_ostream_t os = pb_ostream_from_buffer(buf, sizeof buf);
        if (!pb_encode(&os, ef_v1_Request_fields, &req))
            return ERROR_CODE::UNKNOWN_FAILURE;
        std::string reply;
        uint8_t type = 0;
        if (connection->request_raw(std::string((const char*)buf, os.bytes_written),
                                    reply, &type) != Status::SUCCESS)
            return ERROR_CODE::COMMUNICATION_ERROR;
        std::memset(&resp, 0, sizeof resp);
        pb_istream_t is = pb_istream_from_buffer((const uint8_t*)reply.data(), reply.size());
        if (!pb_decode(&is, ef_v1_Response_fields, &resp))
            return ERROR_CODE::COMMUNICATION_ERROR;
        if (resp.code != ef_v1_ErrorCode_SUCCESS) {
            // The device's message carries the specific reason a code collapsed.
            if (init.verbose && resp.message[0])
                std::fprintf(stderr, "[ef] device error %d: %s\n",
                             (int)resp.code, resp.message);
            return err_from(resp.code, ctx);
        }
        if (expect && resp.which_body != expect)  return ERROR_CODE::COMMUNICATION_ERROR;
        return ERROR_CODE::SUCCESS;
    }

    // Best-effort verbs (failures ignored by design).
    void stop_stream_quiet() {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_stop_stream_tag;
        WireResponse resp;
        (void)call(req, resp);
    }

    // Align the device clock with the host wall clock. Best-effort at open()
    // (a device that rejects it still streams; timestamps just stay device-epoch).
    void set_time_quiet() {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body            = ef_v1_Request_set_time_tag;
        req.body.set_time.wall_ns = host_now_ns();
        WireResponse resp;
        (void)call(req, resp);
    }

    // Device-truth state: ask the device FSM (get_state verb) and fold it into
    // the DEVICE_STATE cache. Best-effort: on any failure the cache keeps its
    // last known value (the public contract has no UNKNOWN state).
    void refresh_device_state() {
        if (!connection) return;
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_state_tag;
        WireResponse resp;
        if (call(req, resp, ef_v1_Response_state_info_tag) != ERROR_CODE::SUCCESS)
            return;
        const char* fsm = resp.body.state_info.state;         // device activity state
        const char* dev = resp.body.state_info.device_state;  // device data-plane state

        // Raw facets (internal, for the illegal-state guards). Recording is
        // reported by the activity state COLLECT (the data-plane RECORDING value
        // is a redundant echo of it); uploading is reported only by the data-plane state.
        dev_recording = !std::strcmp(fsm, "COLLECT") || !std::strcmp(dev, "RECORDING");
        dev_uploading = !std::strcmp(dev, "UPLOADING");

        // Public 4-value projection. Per product decision, STREAMING = the device
        // is moving data in ANY form, a live host stream, a recording, an upload,
        // or a calibration capture. There is no separate CALIBRATION device-state:
        // the device's M_CAL / "CAL" FSM state folds into STREAMING. Only a
        // genuinely quiet device (or a health sweep / SAFE / SLEEP) is IDLE.
        auto r = reader_snapshot();
        const bool reader_live = (r && r->is_running());
        if      (!std::strcmp(fsm, "OTA"))  state = DEVICE_STATE::UPDATING;
        else if (reader_live || dev_recording || dev_uploading ||
                 !std::strcmp(fsm, "CAL"))  state = DEVICE_STATE::STREAMING;
        else                                state = DEVICE_STATE::IDLE;
    }

    // Configure the ISO_LIVE session: geometry, NV12, codec + HQ quality, IMU.
    ERROR_CODE configure_session(const Resolution& res) {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body                        = ef_v1_Request_configure_tag;
        req.body.configure.mode               = ef_v1_Mode_ISO_LIVE;
        req.body.configure.has_video          = true;
        req.body.configure.video.width        = (uint32_t)res.width;
        req.body.configure.video.height       = (uint32_t)res.height;
        req.body.configure.video.fps          = (uint32_t)init.fps;
        req.body.configure.video.pixel_format = ef_v1_PixelFormat_NV12;
        req.body.configure.video.codec        = pb_codec(init.compression);
        req.body.configure.video.quality      = quality_for(init.compression);
        req.body.configure.has_imu            = true;
        req.body.configure.imu.enabled        = init.enable_imu;
        WireResponse resp;
        ERROR_CODE ec = call(req, resp, 0, Ctx::SESSION);
        // The wire collapses every bad Configure knob into INVALID_PARAMETER;
        // the device's reason string names the culprit ("camera:codec",
        // "camera:not_usable", "camera:not_advertised", ...). Every RESOLUTION
        // enum value is a real sensor geometry, so a rejected camera tuple that
        // isn't a codec complaint is attributed to fps, so INVALID_FPS has to be
        // reachable from a device rejection (the host pre-check is skipped when
        // the device advertises no caps menu).
        return disambiguate_config_error(ec, resp);
    }

    // Map the configured compression to the assembler's keyframe-gate codec id
    // (0 = RAW/MJPEG → no resync gate, 1 = H264, 2 = H265).
    int codec_gate_id() const {
        switch (pb_codec(init.compression)) {
            case ef_v1_Codec_H264: return 1;
            case ef_v1_Codec_H265: return 2;
            default:               return 0;
        }
    }

    // USB: configure + StartStream(LOCAL) + the host isoc reader.
    ERROR_CODE start_streaming(const Resolution& res) {
        ERROR_CODE ec = configure_session(res);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        // The reader must be listening BEFORE the device starts emitting:
        // isoc is fire-and-forget, and the first access unit carries the
        // encoder's SPS/PPS. Miss it and an encoded stream never decodes.
        auto r = std::make_shared<internal::StreamReader>();
        r->set_video_codec(codec_gate_id());   // drop-until-IDR resync classifier
        Status st = r->start(init.device_id, /*video=*/true,
                             init.enable_imu, init.verbose);
        if (st != Status::SUCCESS) return open_err(st);
        {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body             = ef_v1_Request_start_stream_tag;
            req.body.start_stream.mode = ef_v1_Mode_ISO_LIVE;
            WireResponse resp;
            ec = call(req, resp, 0, Ctx::SESSION);
            if (ec != ERROR_CODE::SUCCESS) { r->stop(); return ec; }
        }
        set_reader(std::move(r));
        return ERROR_CODE::SUCCESS;
    }

    // BLE + WiFi: configure + StartStream(ONLINE -> udp_host:udp_port over UDP)
    // + the host UDP reader bound on udp_port.
    ERROR_CODE start_streaming_online(const Resolution& res) {
        ERROR_CODE ec = configure_session(res);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        // Bind the UDP socket before StartStream for the same reason the
        // isoc reader starts first: the opening access unit (SPS/PPS) is
        // sent immediately and is not retransmitted.
        auto r = std::make_shared<internal::UdpStreamReader>();
        r->set_video_codec(codec_gate_id());   // drop-until-IDR resync classifier
        Status st = r->start(init.udp_port, /*video=*/true,
                             init.enable_imu, init.verbose);
        if (st != Status::SUCCESS)
            return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body                    = ef_v1_Request_start_stream_tag;
            req.body.start_stream.mode        = ef_v1_Mode_ISO_LIVE;
            req.body.start_stream.has_target  = true;
            req.body.start_stream.target.kind = ef_v1_StreamTarget_Kind_ONLINE;
            std::snprintf(req.body.start_stream.target.dest_host,
                          sizeof req.body.start_stream.target.dest_host, "%s",
                          init.udp_host.c_str());
            req.body.start_stream.target.dest_port = init.udp_port;
            req.body.start_stream.target.protocol  = ef_v1_Protocol_UDP;
            WireResponse resp;
            ec = call(req, resp, 0, Ctx::SESSION);
            if (ec != ERROR_CODE::SUCCESS) { r->stop(); return ec; }
        }
        set_reader(std::move(r));
        return ERROR_CODE::SUCCESS;
    }

    // Start the data plane on demand (first grab / HOST_FILE recording). No-op
    // once a reader exists. BLE without a udp_host has no data plane.
    ERROR_CODE ensure_streaming() {
        if (state == DEVICE_STATE::CLOSED) return ERROR_CODE::INVALID_FUNCTION_CALL;
        if (reader_snapshot())             return ERROR_CODE::SUCCESS;
        if (init.input_type == INPUT_TYPE::MCAP)
            return ERROR_CODE::INVALID_FUNCTION_CALL;      // replay reader is gone
        // Clear a stale stream left armed by a prior client that died mid-stream
        // (StopStream is harmless when idle). Done here, not at open(), so it
        // never touches a DEVICE_LOCAL recording started by another process.
        stop_stream_quiet();
        ERROR_CODE ec;
        if (init.input_type == INPUT_TYPE::STREAM) {
            if (init.udp_host.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
            ec = start_streaming_online(stream_res);
        } else {
            ec = start_streaming(stream_res);
        }
        if (ec == ERROR_CODE::SUCCESS) state = DEVICE_STATE::STREAMING;
        return ec;
    }

    void set_reader(std::shared_ptr<internal::StreamAssembler> r) {
        std::lock_guard<std::mutex> lk(reader_mtx);
        reader = std::move(r);
    }

    void stop_streaming() {
        std::shared_ptr<internal::StreamAssembler> r;
        {
            std::lock_guard<std::mutex> lk(reader_mtx);
            r = std::move(reader);
            reader.reset();
        }
        if (r) {
            r->stop();   // wakes any blocked grab(); the object stays alive
                         // until the last in-flight snapshot drops it
            stop_stream_quiet();   // stops only the stream this session started.
                                   // StopStream==StopCollect on the device, so
                                   // an unconditional call here would stop a
                                   // DEVICE_LOCAL recording too.
        }
    }

    void close_transport() {
        // ctl_mtx: never tear the transport down under an in-flight round trip
        // (libusb documents closing a handle with a blocked transfer as UB).
        std::lock_guard<std::mutex> lk(ctl_mtx);
        if (connection) connection->close();
        connection.reset();
        state = DEVICE_STATE::CLOSED;
    }

    // After a provisioning verb, refresh the cached wireless snapshot so
    // get_device_information().wireless reflects the change (best-effort):
    // live association from GetWifiStatus, saved networks from WifiList
    // (device truth, the cache would otherwise drift on re-adds and would
    // never see networks provisioned in earlier sessions).
    void refresh_wireless() {
        {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body = ef_v1_Request_get_wifi_status_tag;
            WireResponse resp;
            if (call(req, resp, ef_v1_Response_wifi_status_tag) == ERROR_CODE::SUCCESS) {
                const WireWifi& w = resp.body.wifi_status;
                info.wireless.wifi_connected     = w.connected;
                info.wireless.wifi_ssid          = w.ssid;
                info.wireless.wifi_ip_address    = w.ip;
                info.wireless.wifi_rssi          = w.rssi;
                info.wireless.internet_reachable = w.internet;
                info.wireless.wifi_link_speed    = w.link_speed;
                info.wireless.wifi_freq_mhz      = w.freq_mhz;
                info.wireless.wifi_security      = w.security;
                // Back-compat: firmware predating WifiStatus.state (field 8) leaves
                // it empty, derive the 3-state from the legacy `connected` bool so
                // the host never surfaces a blank state. (link_speed/freq/security
                // simply stay 0/"" on old firmware; the printers omit them.)
                info.wireless.wifi_state =
                    w.state[0] ? w.state : (w.connected ? "connected" : "disconnected");
            }
        }
        {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body = ef_v1_Request_wifi_list_tag;
            WireResponse resp;
            if (call(req, resp, ef_v1_Response_wifi_list_result_tag) == ERROR_CODE::SUCCESS) {
                const ef_v1_WifiListResult& l = resp.body.wifi_list_result;
                info.wireless.saved_networks.clear();
                for (pb_size_t i = 0; i < l.ssids_count; i++)
                    info.wireless.saved_networks.push_back(l.ssids[i]);
            }
        }
    }

    // Every IMU drain goes through here so an active host recording sees every
    // sample regardless of whether the app calls retrieve_imu. Raw IMAGE-frame
    // samples (recorder gets device truth; the user-facing convention transform
    // happens in retrieve_imu). Consumer-thread only.
    void drain_imu_tee(std::vector<ImuSample>& out, uint64_t* dropped,
                       const std::shared_ptr<HostRecorder>& hr) {
        out.clear();
        uint64_t d = 0;
        if (auto r = reader_snapshot()) r->drain_imu(out, false, &d);
        if (hr)
            for (const ImuSample& s : out) hr->push_imu(s);
        if (init.flip_mode == FLIP_MODE::AUTO && flip_latched < 0)
            for (const ImuSample& s : out) {
                latch_flip_from_accel(s.acceleration);
                if (flip_latched >= 0) break;
            }
        if (dropped) *dropped = d;
    }

    bool effective_flip() const {
        if (init.flip_mode == FLIP_MODE::ON)  return true;
        if (init.flip_mode == FLIP_MODE::OFF) return false;
        return flip_latched == 1;   // AUTO: unresolved (no gravity fix yet) = OFF
    }

#ifdef EF_HAVE_BLE
    // BLE control-plane authentication: prove the password with a PBKDF2 +
    // HMAC-SHA256 challenge-response so it never crosses the air. Runs once
    // right after the link is up (before any gated verb, or the device answers
    // AUTH_REQUIRED to everything on this connection).
    ERROR_CODE ble_auth(const std::string& pw) {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_auth_challenge_tag;
        WireResponse resp;
        ERROR_CODE ec = call(req, resp, ef_v1_Response_auth_challenge_tag);
        // Only AUTH_REQUIRED/AUTH_FAILED mean "wrong password", and err_from
        // already maps those unconditionally. Everything else (UNSUPPORTED,
        // NOT_OPENED during an OTA apply, transport loss, ...) surfaces
        // untranslated, no password was tested, so INVALID_PASSWORD would
        // send the user chasing the wrong problem.
        if (ec != ERROR_CODE::SUCCESS) return ec;

        const ef_v1_AuthChallenge& ch = resp.body.auth_challenge;
        uint8_t key[32], mac[32];
        PKCS5_PBKDF2_HMAC(pw.c_str(), (int)pw.size(), ch.salt.bytes, (int)ch.salt.size,
                          (int)ch.iters, EVP_sha256(), 32, key);
        unsigned int mlen = 0;
        HMAC(EVP_sha256(), key, 32, ch.nonce.bytes, ch.nonce.size, mac, &mlen);

        WireRequest areq = ef_v1_Request_init_zero;
        areq.which_body                     = ef_v1_Request_authenticate_tag;
        areq.body.authenticate.response.size = 32;
        std::memcpy(areq.body.authenticate.response.bytes, mac, 32);
        WireResponse aresp;
        // A wrong password answers AUTH_FAILED -> INVALID_PASSWORD via err_from.
        return call(areq, aresp);
    }
#endif

    // Sideload a local .eff over the control link: OtaPushBegin announces
    // {name,total}; each OtaPushChunk carries <= 7168 B (the request then
    // encodes under the 8192 control-frame cap); the eof chunk triggers
    // device-side reassembly + verification.
    ERROR_CODE ota_push(const std::string& path,
                        const std::function<void(uint64_t, uint64_t)>& progress) {
        std::FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) return ERROR_CODE::FAILED_TO_UPDATE;
        std::fseek(f, 0, SEEK_END);
        long sz = std::ftell(f);
        std::fseek(f, 0, SEEK_SET);
        if (sz < 0) { std::fclose(f); return ERROR_CODE::FAILED_TO_UPDATE; }
        const uint64_t total = (uint64_t)sz;

        // basename becomes the device-side staging file name.
        std::string base = path;
        if (auto slash = base.find_last_of('/'); slash != std::string::npos)
            base = base.substr(slash + 1);

        {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body = ef_v1_Request_ota_push_begin_tag;
            std::snprintf(req.body.ota_push_begin.name,
                          sizeof req.body.ota_push_begin.name, "%s", base.c_str());
            req.body.ota_push_begin.total_size = total;
            WireResponse resp;
            ERROR_CODE ec = call(req, resp, 0, Ctx::UPDATE);
            if (ec != ERROR_CODE::SUCCESS) { std::fclose(f); return ec; }
        }

        const size_t kChunk = sizeof(ef_v1_OtaPushChunk{}.data.bytes);   // 7168
        uint64_t offset = 0;
        ERROR_CODE result = ERROR_CODE::SUCCESS;
        for (;;) {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body = ef_v1_Request_ota_push_chunk_tag;
            size_t n = std::fread(req.body.ota_push_chunk.data.bytes, 1, kChunk, f);
            bool eof = (offset + n >= total);
            req.body.ota_push_chunk.offset    = offset;
            req.body.ota_push_chunk.data.size = (pb_size_t)n;
            req.body.ota_push_chunk.eof       = eof;
            WireResponse resp;
            ERROR_CODE ec = call(req, resp, 0, Ctx::UPDATE);
            if (ec != ERROR_CODE::SUCCESS) { result = ec; break; }
            offset += n;
            if (progress) progress(offset, total);
            if (eof) break;
            if (n == 0) { result = ERROR_CODE::FAILED_TO_UPDATE; break; }  // short read before eof
        }
        std::fclose(f);

        // On failure, ask the device to discard the partial staging (best-effort).
        if (result != ERROR_CODE::SUCCESS) {
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body          = ef_v1_Request_ota_push_end_tag;
            req.body.ota_push_end.ok = false;
            WireResponse resp;
            (void)call(req, resp);
        }
        return result;
    }
};

// ---- construction ---------------------------------------------------------------

Device::Device() : impl_(new Impl) {}
Device::~Device() { close(); }
Device::Device(Device&&) noexcept = default;
// Move-assignment releases the target's device cleanly first: the defaulted
// version would destroy the Impl directly, skipping StopStream/StopRecording,
// and leave the device with an armed session.
Device& Device::operator=(Device&& other) noexcept {
    if (this != &other) {
        close();
        impl_ = std::move(other.impl_);
    }
    return *this;
}

// ---- discovery --------------------------------------------------------------------

std::vector<DeviceProperties> Device::get_device_list(bool scan_ble, uint32_t scan_ms) {
    std::vector<DeviceProperties> out;
    libusb_context* ctx = nullptr;
    if (libusb_init(&ctx) == 0) {
        libusb_device** list = nullptr;
        ssize_t cnt = libusb_get_device_list(ctx, &list);
        int idx = 0;
        for (ssize_t k = 0; k < cnt; k++) {
            libusb_device_descriptor dd{};
            if (libusb_get_device_descriptor(list[k], &dd) != 0) continue;
            if (dd.idVendor != kVid || dd.idProduct != kPid)     continue;
            DeviceProperties p;
            p.input_type = INPUT_TYPE::USB;
            p.device_id  = idx++;
            // iSerialNumber needs a device open, which can fail on permissions;
            // discovery still lists the device (serial stays empty).
            libusb_device_handle* h = nullptr;
            if (dd.iSerialNumber && libusb_open(list[k], &h) == 0) {
                unsigned char tmp[256];
                int n = libusb_get_string_descriptor_ascii(h, dd.iSerialNumber,
                                                           tmp, sizeof tmp);
                if (n > 0) p.serial.assign((const char*)tmp, (size_t)n);
                libusb_close(h);
            }
            out.push_back(std::move(p));
        }
        if (cnt >= 0) libusb_free_device_list(list, 1);
        libusb_exit(ctx);
    }
#ifdef EF_HAVE_BLE
    if (scan_ble) {
        int bidx = 0;
        for (const auto& r : internal::BleConnection::scan(scan_ms, 0)) {
            DeviceProperties p;
            p.input_type  = INPUT_TYPE::STREAM;
            p.device_id   = bidx++;
            p.ble_address = r.address;
            p.ble_name    = r.name;
            out.push_back(std::move(p));
        }
    }
#else
    // Built without BLE support: only USB devices can be listed.
    (void)scan_ble; (void)scan_ms;
#endif
    return out;
}

// ---- open / close -------------------------------------------------------------------

namespace {

// Fill the public DeviceInformation tree + the internal caps menu from the wire.
void info_from_wire(const WireDeviceInfo& d, INPUT_TYPE transport,
                    const std::string& usb_serial, DeviceInformation* out, Caps* caps) {
    DeviceInformation di;
    di.model            = MODEL::M1;
    di.firmware_version = d.firmware_version_int;
    di.input_type       = transport;
    // The canonical serial (eMMC CID hex today) with the USB descriptor as a
    // fallback; serial_number is its numeric convenience form, parsed only when
    // the string is entirely decimal, a hex CID stays 0.
    di.serial = d.serial[0] ? d.serial : usb_serial;
    {
        bool numeric = !di.serial.empty();
        for (char c : di.serial) if (c < '0' || c > '9') { numeric = false; break; }
        if (numeric) di.serial_number = (unsigned int)std::strtoul(di.serial.c_str(), nullptr, 10);
    }

    CameraConfiguration& cc = di.camera_configuration;
    if (d.has_camera) {
        cc.calibration.fx    = d.camera.fx;
        cc.calibration.fy    = d.camera.fy;
        cc.calibration.cx    = d.camera.cx;
        cc.calibration.cy    = d.camera.cy;
        cc.calibration.xi    = d.camera.xi;
        cc.calibration.alpha = d.camera.alpha;
        cc.resolution.width  = (int)d.camera.width;
        cc.resolution.height = (int)d.camera.height;
    }
    if (d.has_current_config) {
        cc.resolution.width  = (int)d.current_config.width;
        cc.resolution.height = (int)d.current_config.height;
        cc.fps               = (int)d.current_config.fps;
        cc.compression       = compression_from(d.current_config.codec);
    }

    SensorsConfiguration& sc = di.sensors_configuration;
    sc.accelerometer.type = SENSOR_TYPE::ACCELEROMETER;
    sc.accelerometer.unit = SENSOR_UNIT::M_SEC_2;
    sc.gyroscope.type     = SENSOR_TYPE::GYROSCOPE;
    sc.gyroscope.unit     = SENSOR_UNIT::DEG_SEC;
    if (d.has_imu) {
        sc.accelerometer.state         = SENSOR_STATE::AVAILABLE;
        sc.gyroscope.state             = SENSOR_STATE::AVAILABLE;
        sc.accelerometer.noise_density = (float)d.imu.accel_noise_density;
        sc.gyroscope.noise_density     = (float)d.imu.gyro_noise_density;
        if (d.imu.imu_to_camera_count >= 16)
            for (int i = 0; i < 16; i++)
                sc.camera_imu_transform.m[(size_t)i] = d.imu.imu_to_camera[i];
    }

    di.wireless.wifi_mac_address = d.wifi_mac;   // vendor storage; "" if unprovisioned
    di.wireless.bt_mac_address   = d.bt_mac;
    if (d.has_wifi) {
        di.wireless.wifi_connected     = d.wifi.connected;
        di.wireless.wifi_ssid          = d.wifi.ssid;
        di.wireless.wifi_ip_address    = d.wifi.ip;
        di.wireless.wifi_rssi          = d.wifi.rssi;
        di.wireless.internet_reachable = d.wifi.internet;
    }

    if (d.has_caps) {
        for (pb_size_t i = 0; i < d.caps.modes_count; i++) {
            // internal validation menu keeps every advertised mode (usable flag
            // and all) so open() can give a precise INVALID_FPS/RESOLUTION reason.
            if (caps) {
                Caps::Mode m;
                m.w      = (int)d.caps.modes[i].width;
                m.h      = (int)d.caps.modes[i].height;
                m.fps    = (int)d.caps.modes[i].fps;
                m.usable = d.caps.modes[i].usable;
                caps->modes.push_back(m);
            }
            // public menu = ENABLED modes only (usable=1); advertised-but-disabled
            // rows are not offered to clients.
            if (d.caps.modes[i].usable) {
                SupportedMode sm;
                sm.resolution.width  = (int)d.caps.modes[i].width;
                sm.resolution.height = (int)d.caps.modes[i].height;
                sm.fps               = (int)d.caps.modes[i].fps;
                di.capabilities.modes.push_back(sm);
            }
        }
        for (pb_size_t i = 0; i < d.caps.codecs_count; i++) {
            if (caps) caps->codecs.push_back(d.caps.codecs[i]);
            di.capabilities.codecs.push_back(d.caps.codecs[i]);
        }
    }
    *out = di;
}

}  // namespace

ERROR_CODE Device::open(InitParameters params) {
    // A moved-from Device reads as CLOSED (is_open() == false), so honour the
    // invitation to reuse it instead of dereferencing a null impl_.
    if (!impl_) impl_.reset(new Impl);
    if (impl_->state != DEVICE_STATE::CLOSED) return ERROR_CODE::INVALID_FUNCTION_CALL;

    impl_->init                = params;
    impl_->flip_latched        = -1;
    impl_->imu_backlog.clear();
    impl_->imu_backlog_dropped = 0;

    // ---- MCAP replay (file-replay source): no transport, the file is the source --
    if (params.input_type == INPUT_TYPE::MCAP) {
        if (params.mcap_path.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
        auto r = std::make_shared<internal::McapReplayReader>();
        Status st = r->start(params.mcap_path, params.verbose);
        if (st == Status::FILE_NOT_FOUND) return ERROR_CODE::DEVICE_NOT_DETECTED;
        if (st != Status::SUCCESS)        return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        // The recording pins the codec; the InitParameters knob is overridden
        // (retrieve_image decodes what the file actually carries).
        const std::string& fmt = r->format();
        if      (fmt == "h265")               impl_->init.compression = COMPRESSION_MODE::H265;
        else if (fmt == "h264")               impl_->init.compression = COMPRESSION_MODE::H264;
        else if (fmt == "nv12" || fmt.empty()) impl_->init.compression = COMPRESSION_MODE::RAW;
        else { r->stop(); return ERROR_CODE::UNSUPPORTED_COMPRESSION; }  // e.g. jpeg
        impl_->info            = DeviceInformation{};
        impl_->info.input_type = INPUT_TYPE::MCAP;
        impl_->info.camera_configuration.resolution  = {r->width(), r->height()};
        impl_->info.camera_configuration.compression = impl_->init.compression;
        impl_->set_reader(std::move(r));
        impl_->state = DEVICE_STATE::STREAMING;
        return ERROR_CODE::SUCCESS;
    }

    // ---- transport -----------------------------------------------------------
    std::string usb_serial;
    if (params.input_type == INPUT_TYPE::STREAM) {
#ifdef EF_HAVE_BLE
        auto ble = std::unique_ptr<internal::BleConnection>(
            new internal::BleConnection(params.verbose));
        Status st = ble->open(params.ble_address);
        if (st != Status::SUCCESS) return open_err(st);
        {
            std::lock_guard<std::mutex> lk(impl_->ctl_mtx);
            impl_->connection = std::move(ble);
        }
        // Authenticate before any gated verb (including the stale StopStream
        // below), or the device answers AUTH_REQUIRED to everything.
        ERROR_CODE ac = impl_->ble_auth(params.ble_password);
        if (ac != ERROR_CODE::SUCCESS) { impl_->close_transport(); return ac; }
#else
        return ERROR_CODE::INVALID_FUNCTION_CALL;   // built without BLE support
#endif
    } else {
        auto usb = std::unique_ptr<internal::UsbConnection>(new internal::UsbConnection());
        Status st = usb->open(params.device_id, params.verbose);
        if (st != Status::SUCCESS) return open_err(st);
        usb_serial = usb->serial_descriptor();
        {
            std::lock_guard<std::mutex> lk(impl_->ctl_mtx);
            impl_->connection = std::move(usb);
        }
    }

    // NOTE: open() deliberately does NOT clear a stale stream here. On the device
    // StopStream maps to StopCollect, which also stops a running DEVICE_LOCAL
    // recording, so clearing at open() would kill a recording that another
    // process started (device-local recordings survive host disconnect). The
    // stale-stream clear happens in ensure_streaming(), just before it starts a
    // stream of its own.

    // ---- identity snapshot (cached; get_device_information() serves this) ----
    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_device_information_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_device_information_tag);
        if (ec != ERROR_CODE::SUCCESS) { impl_->close_transport(); return ec; }
        info_from_wire(resp.body.device_information, params.input_type, usb_serial,
                       &impl_->info, &impl_->caps);
    }

    // ---- validate the session configuration against the advertised menu ------
    Resolution res = get_resolution(params.resolution);
    if (params.resolution == RESOLUTION::AUTO) {
        res = {1920, 1080};
        for (const auto& m : impl_->caps.modes)
            if (m.usable && m.fps == params.fps) { res = {m.w, m.h}; break; }
    } else if (!impl_->caps.modes.empty()) {
        bool res_ok = false, fps_ok = false;
        for (const auto& m : impl_->caps.modes) {
            if (m.usable && m.w == res.width && m.h == res.height) {
                res_ok = true;
                if (m.fps == params.fps) fps_ok = true;
            }
        }
        if (!res_ok) { impl_->close_transport(); return ERROR_CODE::INVALID_RESOLUTION; }
        if (!fps_ok) { impl_->close_transport(); return ERROR_CODE::INVALID_FPS; }
    }
    if (!impl_->caps.codecs.empty()) {
        const char* want = (pb_codec(params.compression) == ef_v1_Codec_H264) ? "H264"
                         : (pb_codec(params.compression) == ef_v1_Codec_H265) ? "H265"
                                                                              : "RAW";
        bool ok = false;
        for (const auto& c : impl_->caps.codecs)
            if (c == want) { ok = true; break; }
        if (!ok) { impl_->close_transport(); return ERROR_CODE::UNSUPPORTED_COMPRESSION; }
    }

    // ---- align the device clock with the host (best-effort, non-fatal) --------
    impl_->set_time_quiet();

    // ---- seed the cached wireless view (saved networks come from WifiList; the
    // identity snapshot only carries the live association) ----------------------
    impl_->refresh_wireless();

    // The data plane starts on the first grab() (see ensure_streaming), so
    // open() leaves the device IDLE and control-only work is never blocked by a
    // COLLECT session. BLE without a udp_host has no data plane at all.
    impl_->stream_res = res;
    impl_->state      = DEVICE_STATE::IDLE;
    impl_->refresh_device_state();
    return ERROR_CODE::SUCCESS;
}

void Device::close() {
    if (!impl_) return;
    if (impl_->state == DEVICE_STATE::CLOSED) {
        // A failed update() reconnect can strand a host recorder with the
        // transport already gone, release it here so a later open() doesn't
        // silently append a new session to the old file.
        impl_->stop_host_rec();
        return;
    }
    // A DEVICE_LOCAL recording survives host disconnect by design, so closing
    // the handle must NOT stop it, only disable_recording() / `record stop`
    // does. Tear down host-side resources (host recorder + reader) only.
    impl_->stop_host_rec();
    impl_->stop_streaming();
    impl_->close_transport();
    impl_->info   = DeviceInformation{};
    impl_->health = HealthStatus{};
    impl_->caps   = Caps{};
    { std::lock_guard<std::mutex> lk(impl_->data_mtx);
      impl_->imu_backlog.clear();
      impl_->imu_backlog_dropped = 0; }
    impl_->flip_latched        = -1;
}

bool         Device::is_open() const { return impl_ && impl_->state != DEVICE_STATE::CLOSED; }
DEVICE_STATE Device::get_state() const {
    return impl_ ? impl_->state.load() : DEVICE_STATE::CLOSED;
}

// Cached get_* never block and stay safe on a moved-from Device (null impl_):
// they serve the type's defaults, matching the CLOSED state such a handle reports.
InitParameters      Device::get_init_parameters() const      { return impl_ ? impl_->init : InitParameters{}; }
RuntimeParameters   Device::get_runtime_parameters() const   { return impl_ ? impl_->runtime : RuntimeParameters{}; }
RecordingParameters Device::get_recording_parameters() const { return impl_ ? impl_->recording : RecordingParameters{}; }
DeviceInformation   Device::get_device_information() const   { return impl_ ? impl_->info : DeviceInformation{}; }
HealthStatus        Device::get_health_status() const        { return impl_ ? impl_->health : HealthStatus{}; }

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
    // While recording, tee the IMU stream every grab so the .mcap carries all
    // samples even if the app never calls retrieve_imu; they queue in the
    // backlog for the next retrieve_imu.
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

    // RAW planes are laid out from the wire-supplied width/height, which are
    // independent of the payload size, reject a frame whose NV12 planes would
    // over-read the buffer before any copy/convert touches it.
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
    // On a decode failure, flush the decoder's reference state so a later frame
    // that lost its references (e.g. frames superseded while a consumer
    // stalled) doesn't keep erroring, it cleanly resyncs at the next keyframe
    // (which the assembler's PLI-on-supersede already requests over UDP).
    if (avcodec_send_packet(impl_->decoder.dec, pkt) < 0) {
        avcodec_flush_buffers(impl_->decoder.dec);
        return ERROR_CODE::CORRUPTED_FRAME;
    }
    AVFrame* frm = impl_->decoder.frm;
    if (avcodec_receive_frame(impl_->decoder.dec, frm) < 0) {
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

// InitParameters conventions applied to one IMU sample (both vectors): first
// the flip (180 degrees about the optical z axis: x,y negate), then the axis
// remap out of the IMAGE frame (x right, y down, z forward) into the requested
// coordinate system. Angular velocity gets the same axis remap the image/pose
// data does, the convention consumer camera SDKs use.
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

    // Coarse motion classification from the newest sample: free fall reads
    // |a| ~ 0; static reads |a| ~ g with a quiet gyro. (Both magnitudes are
    // invariant under the convention remap below.)
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

// ---- health ------------------------------------------------------------------------

namespace {

// Map one wire sweep into the public HealthStatus (live states derived from
// the camera / imu probes). A skipped probe passes (the tier wasn't run, not
// failed) with "(skipped)" appended so the detail says why it's hollow.
HealthStatus health_from_wire(const WireHealth& s) {
    HealthStatus h;
    // The sweep verdict is the authority for the post-sweep result: PASS unless some
    // check FAILed (SKIP ignored). `health_passed` is the fast inline gate that only
    // StartHealthTest returns, it is NOT carried on the GetHealthResult poll path this
    // result comes from, so ANDing it in here would force the result to FAIL always.
    h.passed            = (s.overall != ef_v1_HealthVerdict_HV_FAIL);
    h.deep              = s.deep;
    h.timestamp.data_ns = (uint64_t)s.timestamp_ns;
    for (pb_size_t i = 0; i < s.checks_count; i++) {
        HealthCheck c;
        c.name   = s.checks[i].name;
        c.passed = (s.checks[i].status != ef_v1_HealthVerdict_HV_FAIL);
        c.detail = s.checks[i].detail;
        if (s.checks[i].status == ef_v1_HealthVerdict_HV_SKIP)
            c.detail += c.detail.empty() ? "(skipped)" : " (skipped)";

        // Live attributes: whether the camera / IMU can currently be read. Derive this
        // BEFORE moving c into the vector, std::move(c) empties c.name, so reading it
        // afterward would never match and leave both stuck at NOT_AVAILABLE.
        bool ok = (s.checks[i].status == ef_v1_HealthVerdict_HV_PASS);
        if (c.name.find("camera") != std::string::npos ||
            c.name.find("mipi")   != std::string::npos)
            h.camera = ok ? CAMERA_STATE::AVAILABLE : CAMERA_STATE::NOT_AVAILABLE;
        if (c.name.find("imu") != std::string::npos)
            h.imu = ok ? SENSOR_STATE::AVAILABLE : SENSOR_STATE::NOT_AVAILABLE;

        h.checks.push_back(std::move(c));
    }
    return h;
}

}  // namespace

ERROR_CODE Device::health_check(HealthStatus& out, bool deep) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    impl_->refresh_device_state();

    // Host-side illegal-state pre-check (contract: decided without touching the
    // device). A sweep can't run while recording / uploading / updating. A sweep
    // already in flight is fine, the poll below attaches to it (device returns BUSY).
    if (impl_->dev_recording || impl_->dev_uploading ||
        impl_->state == DEVICE_STATE::UPDATING || impl_->device_rec)
        return ERROR_CODE::INVALID_FUNCTION_CALL;

    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body                   = ef_v1_Request_start_health_check_tag;
        req.body.start_health_check.deep = deep;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp);
        // ONLY a wire BUSY means "a sweep is already running, poll it".
        // err_from collapses BUSY and INVALID_STATE (and more) into
        // INVALID_FUNCTION_CALL, so check the wire code: an INVALID_STATE
        // rejection (recording/OTA started elsewhere) must surface as the
        // error, not silently return the previous sweep as a fresh pass.
        const bool sweep_running = (ec == ERROR_CODE::INVALID_FUNCTION_CALL &&
                                    resp.code == ef_v1_ErrorCode_BUSY);
        if (ec != ERROR_CODE::SUCCESS && !sweep_running)
            return ec;
    }

    // Poll until the sweep lands (typical completion: shallow ~30 s, deep ~2-3 min).
    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::seconds(deep ? 300 : 90);
    for (;;) {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_health_status_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_health_status_tag);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        if (!resp.body.health_status.sweep_in_flight) {
            impl_->health = health_from_wire(resp.body.health_status);
            out = impl_->health;
            return ERROR_CODE::SUCCESS;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            return ERROR_CODE::UNKNOWN_FAILURE;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

// ---- recording -----------------------------------------------------------------------

ERROR_CODE Device::enable_recording(RecordingParameters params) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    if (params.target == RECORDING_TARGET::HOST_FILE) {
        // Validate before touching the data plane: a bad path or a recording
        // already in flight is decided without starting the device streaming.
        if (params.path.empty())        return ERROR_CODE::INVALID_FUNCTION_CALL;
        if (impl_->host_rec_snapshot()) return ERROR_CODE::INVALID_FUNCTION_CALL;
        ERROR_CODE se = impl_->ensure_streaming();   // HOST_FILE tees the grab stream
        if (se != ERROR_CODE::SUCCESS)               return se;
        const bool raw = (pb_codec(impl_->init.compression) == ef_v1_Codec_CODEC_RAW);
        const char* fmt = raw ? "nv12"
                        : (pb_codec(impl_->init.compression) == ef_v1_Codec_H264) ? "h264"
                                                                                  : "h265";
        auto rec = std::make_shared<HostRecorder>();
        if (!rec->start(params.path, raw, fmt,
                        impl_->info.camera_configuration.resolution.width,
                        impl_->info.camera_configuration.resolution.height))
            return ERROR_CODE::SESSION_RECORDING_ERROR;
        { std::lock_guard<std::mutex> lk(impl_->data_mtx); impl_->host_rec = std::move(rec); }
        impl_->recording = params;
        return ERROR_CODE::SUCCESS;
    }

    // DEVICE_LOCAL: the device records to eMMC; survives host disconnect.
    // Host-side illegal-state pre-check first (contract: decided without
    // touching the device): a second recording, or one during an update /
    // upload, is known-illegal locally. A concurrent health sweep is caught by
    // the device (BUSY/INVALID_STATE -> INVALID_FUNCTION_CALL via Ctx::RECORDING).
    if (impl_->device_rec || impl_->dev_recording || impl_->dev_uploading ||
        impl_->state == DEVICE_STATE::UPDATING)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                  = ef_v1_Request_start_recording_tag;
    req.body.start_recording.target = ef_v1_RecordingTarget_REC_DEVICE_LOCAL;
    std::snprintf(req.body.start_recording.name,
                  sizeof req.body.start_recording.name, "%s", params.name.c_str());
    if (params.has_location) {
        req.body.start_recording.has_location = true;
        req.body.start_recording.location.latitude        = params.location.latitude;
        req.body.start_recording.location.longitude       = params.location.longitude;
        req.body.start_recording.location.altitude        = params.location.altitude;
        req.body.start_recording.location.covariance_diag = params.location.covariance_diag;
    }
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::RECORDING);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    impl_->device_rec      = true;
    impl_->device_rec_name = params.name;
    impl_->recording       = params;
    impl_->refresh_device_state();
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::disable_recording() {
    if (!impl_) return ERROR_CODE::DEVICE_NOT_INITIALIZED;   // moved-from handle
    // A host-file recording is purely local: stopping it is the real action and
    // always succeeds, independent of any device session.
    const bool had_host_rec = (bool)impl_->host_rec_snapshot();
    impl_->stop_host_rec();
    if (!impl_->connection) return ERROR_CODE::SUCCESS;
    // Stop the device's recording session. DEVICE_LOCAL recordings are
    // cross-process (`record stop` runs in a different process than the
    // `record start` that began it), so this is NOT gated on this handle's
    // device_rec: an empty name tells the device to stop its current session.
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_stop_recording_tag;
    std::snprintf(req.body.stop_recording.name,
                  sizeof req.body.stop_recording.name, "%s",
                  impl_->device_rec_name.c_str());   // "" => current session
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::RECORDING);
    impl_->device_rec = false;
    impl_->refresh_device_state();
    // The device rejects a stop when nothing is recording (INVALID_STATE ->
    // INVALID_FUNCTION_CALL), surface that. But if this call just stopped a local
    // host-file recording, that was a real stop, so report success regardless.
    return had_host_rec ? ERROR_CODE::SUCCESS : ec;
}

namespace {

void recording_from_wire(const WireRecording& s, RecordingStatus* out) {
    out->name        = s.name;
    out->target      = RECORDING_TARGET::DEVICE_LOCAL;
    out->recording   = s.recording;
    out->bytes       = s.bytes;
    out->frames      = s.frames;
    out->duration_ms = s.duration_ms;
}

}  // namespace

ERROR_CODE Device::get_recording_status(RecordingStatus& out, const std::string& name) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    // Host-side recording: everything is local. Pin the recorder with a snapshot
    // so a concurrent close()/update()/disable_recording() can't free it between
    // the check and the dereferences below (same discipline as grab/retrieve).
    if (auto hr = impl_->host_rec_snapshot(); hr && name.empty()) {
        out = RecordingStatus{};
        out.name        = impl_->recording.path;
        out.target      = RECORDING_TARGET::HOST_FILE;
        out.recording   = true;
        out.bytes       = hr->bytes;
        out.frames      = hr->frames;
        out.duration_ms = (host_now_ns() - hr->t0_ns) / 1000000ULL;
        return ERROR_CODE::SUCCESS;
    }

    impl_->refresh_device_state();

    out = RecordingStatus{};
    {
        // The device firmware has no get_recording_status verb (it would answer
        // UNSUPPORTED); compose the same reply from list_recordings (the verb
        // the implementation uses) plus storage and upload
        // status below.
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_list_recordings_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_recording_list_tag,
                                    Ctx::RECORDING);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        const WireRecordingList& L = resp.body.recording_list;
        // "" = the active session: ours if one is running, else whichever the
        // device reports as recording.
        const std::string want = name.empty() ? impl_->device_rec_name : name;
        const WireRecording* hit = nullptr;
        for (pb_size_t i = 0; i < L.recordings_count; i++) {
            if (want.empty() ? L.recordings[i].recording
                             : (want == L.recordings[i].name)) {
                hit = &L.recordings[i];
                break;
            }
        }
        if (!hit) return ERROR_CODE::RECORDING_NOT_FOUND;
        recording_from_wire(*hit, &out);
    }
    {   // storage headroom
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_storage_tag;
        WireResponse resp;
        if (impl_->call(req, resp, ef_v1_Response_storage_info_tag) == ERROR_CODE::SUCCESS) {
            out.storage_free_bytes  = resp.body.storage_info.free_bytes;
            out.storage_total_bytes = resp.body.storage_info.total_bytes;
        }
    }
    {   // upload progress of this session (best-effort)
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_upload_status_tag;
        std::snprintf(req.body.get_upload_status.recording_name,
                      sizeof req.body.get_upload_status.recording_name, "%s",
                      name.empty() ? out.name.c_str() : name.c_str());
        WireResponse resp;
        if (impl_->call(req, resp, ef_v1_Response_upload_status_tag) == ERROR_CODE::SUCCESS) {
            const WireUpload& u = resp.body.upload_status;
            out.upload_bytes_sent  = u.bytes_sent;
            out.upload_bytes_total = u.bytes_total;
            switch (u.state) {
                case ef_v1_UploadState_UPLOAD_RUNNING:
                    out.upload = UPLOAD_STATE::RUNNING;
                    break;
                case ef_v1_UploadState_UPLOAD_DONE:
                    out.upload            = UPLOAD_STATE::OFF;
                    out.upload_bytes_sent = u.bytes_total;   // finished => all sent
                    break;
                case ef_v1_UploadState_UPLOAD_FAILED: {
                    out.upload = UPLOAD_STATE::OFF;
                    ERROR_CODE le = err_from(u.last_error, Ctx::UPLOAD);
                    out.last_error = (le == ERROR_CODE::SUCCESS)
                                         ? ERROR_CODE::UNKNOWN_FAILURE : le;
                    break;
                }
                default:   // UPLOAD_IDLE
                    out.upload = UPLOAD_STATE::OFF;
                    break;
            }
        }
    }
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::list_recordings(std::vector<RecordingStatus>& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    out.clear();
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_list_recordings_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_recording_list_tag,
                                Ctx::RECORDING);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    const WireRecordingList& L = resp.body.recording_list;
    for (pb_size_t i = 0; i < L.recordings_count; i++) {
        RecordingStatus rs;
        recording_from_wire(L.recordings[i], &rs);
        out.push_back(std::move(rs));
    }
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::delete_recording(const std::string& name) {
    if (!is_open())   return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (name.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_delete_recording_tag;
    std::snprintf(req.body.delete_recording.name,
                  sizeof req.body.delete_recording.name, "%s", name.c_str());
    WireResponse resp;
    return impl_->call(req, resp, 0, Ctx::RECORDING);
}

ERROR_CODE Device::download_recording(const std::string& name,
                                      const std::string& dest_path) {
    if (!is_open())                        return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (name.empty() || dest_path.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;

    // The device answers plain FAILURE (not NOT_FOUND) for a missing recording,
    // which err_from can only map to SESSION_RECORDING_ERROR, pre-validate the
    // name against list_recordings so a bad name reports the documented
    // RECORDING_NOT_FOUND (and no empty dest file is created).
    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_list_recordings_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_recording_list_tag,
                                    Ctx::RECORDING);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        const WireRecordingList& L = resp.body.recording_list;
        bool found = false;
        for (pb_size_t i = 0; i < L.recordings_count && !found; i++)
            found = (name == L.recordings[i].name);
        if (!found) return ERROR_CODE::RECORDING_NOT_FOUND;
    }

    std::FILE* f = std::fopen(dest_path.c_str(), "wb");
    if (!f) return ERROR_CODE::SESSION_RECORDING_ERROR;

    // Pull loop, the mirror of the OTA sideload push. Ask for a window at each
    // offset; the device returns up to 7168 B per RESPONSE; loop to eof. Rides
    // the control transport (USB or BLE) like every verb, so no WiFi is needed.
    const uint32_t kChunk = (uint32_t)sizeof(ef_v1_RecordingChunk{}.data.bytes);  // 7168
    uint64_t offset = 0;
    ERROR_CODE result = ERROR_CODE::SUCCESS;
    for (;;) {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_download_recording_tag;
        std::snprintf(req.body.download_recording.name,
                      sizeof req.body.download_recording.name, "%s", name.c_str());
        req.body.download_recording.offset = offset;
        req.body.download_recording.len    = kChunk;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_recording_chunk_tag,
                                    Ctx::RECORDING);
        if (ec != ERROR_CODE::SUCCESS) { result = ec; break; }

        const ef_v1_RecordingChunk& c = resp.body.recording_chunk;
        if (c.data.size) {
            if (std::fwrite(c.data.bytes, 1, c.data.size, f) != c.data.size) {
                result = ERROR_CODE::SESSION_RECORDING_ERROR;
                break;
            }
            offset += c.data.size;
        }
        if (c.eof) break;
        if (c.data.size == 0) { result = ERROR_CODE::COMMUNICATION_ERROR; break; }  // no progress, not eof
    }
    std::fclose(f);
    if (result != ERROR_CODE::SUCCESS) std::remove(dest_path.c_str());   // no partial files
    return result;
}

// The device uploads via an HTTP PUT to a pre-signed URL, so only http(s)
// schemes are meaningful. Reject anything else host-side (case-insensitive,
// per RFC 3986) rather than shipping a bad URL that fails opaquely on-device.
static bool is_http_url(const std::string& u) {
    auto starts_with_ci = [&](const char* p) {
        std::size_t n = std::strlen(p);
        if (u.size() < n) return false;
        for (std::size_t i = 0; i < n; ++i)
            if (std::tolower((unsigned char)u[i]) != p[i]) return false;
        return true;
    };
    return starts_with_ci("http://") || starts_with_ci("https://");
}

ERROR_CODE Device::upload_recording(const std::string& name, const std::string& url) {
    if (!is_open())                 return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (name.empty() || url.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    if (!is_http_url(url))          return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_start_upload_tag;
    std::snprintf(req.body.start_upload.recording_name,
                  sizeof req.body.start_upload.recording_name, "%s", name.c_str());
    req.body.start_upload.has_spec  = true;
    req.body.start_upload.spec.dest = ef_v1_UploadDest_UPLOAD_CLOUD_URL;
    req.body.start_upload.spec.mode = ef_v1_UploadMode_UPLOAD_AFTER_STOP;
    // Reject rather than silently truncate a URL longer than the wire buffer.
    if (url.size() >= sizeof req.body.start_upload.spec.presigned_url)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    std::snprintf(req.body.start_upload.spec.presigned_url,
                  sizeof req.body.start_upload.spec.presigned_url, "%s", url.c_str());
    WireResponse resp;
    return impl_->call(req, resp, 0, Ctx::UPLOAD);
}

ERROR_CODE Device::stop_upload(const std::string& name) {
    if (!is_open())   return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (name.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_stop_upload_tag;
    std::snprintf(req.body.stop_upload.recording_name,
                  sizeof req.body.stop_upload.recording_name, "%s", name.c_str());
    WireResponse resp;
    return impl_->call(req, resp, 0, Ctx::UPLOAD);
}

// ---- updates --------------------------------------------------------------------------

namespace {

// Wire OtaStatus -> public UpdateStatus. UPDATE_STATE covers the in-flight
// phases only; everything else is `active == false` + a terminal code.
UpdateStatus update_from_wire(const WireOta& s) {
    UpdateStatus u;
    u.progress            = s.progress;
    u.message             = s.message;
    u.running_version     = s.running_version_str;
    u.running_version_int = s.running_version_int;
    u.target_version_int  = s.target_version_int;
    switch (s.phase) {
        case ef_v1_OtaPhase_OTA_CHECKING:
        case ef_v1_OtaPhase_OTA_DOWNLOADING:
            u.active = true; u.state = UPDATE_STATE::DOWNLOADING; break;
        case ef_v1_OtaPhase_OTA_VERIFYING:
            u.active = true; u.state = UPDATE_STATE::VERIFYING; break;
        case ef_v1_OtaPhase_OTA_READY:
            u.active = true; u.state = UPDATE_STATE::READY_TO_APPLY; break;
        case ef_v1_OtaPhase_OTA_APPLYING:
        case ef_v1_OtaPhase_OTA_REBOOTING:
            u.active = true; u.state = UPDATE_STATE::APPLYING; break;
        case ef_v1_OtaPhase_OTA_ERROR:
        case ef_v1_OtaPhase_OTA_ROLLEDBACK:
            u.last_error = ERROR_CODE::FAILED_TO_UPDATE; break;
        case ef_v1_OtaPhase_OTA_UPTODATE:
            u.last_error = ERROR_CODE::DEVICE_UP_TO_DATE; break;
        default: break;   // UNKNOWN / AVAILABLE / SUCCESS: inactive
    }
    return u;
}

}  // namespace

ERROR_CODE Device::get_update_status(UpdateStatus& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_get_ota_status_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_ota_status_tag);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    out = update_from_wire(resp.body.ota_status);
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::check_update(bool& available) {
    available = false;
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_ota_check_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
        if (ec != ERROR_CODE::SUCCESS) return ec;
    }
    // Poll the check to completion (manifest fetch + signature verify).
    //
    // The device spawns the check DETACHED and acks the verb before the status
    // file is rewritten, so the first polls can still read the PRE-check phase:
    // OTA_UNKNOWN on a fresh device, or a stale terminal phase (error /
    // success / uptodate) from an earlier attempt. Nothing is trusted until
    // CHECKING has been observed, or a start grace elapses without it (a very
    // fast check can finish between two polls).
    auto deadline    = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    auto start_grace = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool seen_checking = false;
    for (;;) {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_ota_status_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_ota_status_tag);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        const WireOta& s = resp.body.ota_status;
        if (s.phase == ef_v1_OtaPhase_OTA_CHECKING) {
            seen_checking = true;
        } else if (seen_checking ||
                   std::chrono::steady_clock::now() >= start_grace) {
            if (s.phase == ef_v1_OtaPhase_OTA_AVAILABLE) { available = true;  break; }
            if (s.phase == ef_v1_OtaPhase_OTA_UPTODATE)  { available = false; break; }
            if (s.phase == ef_v1_OtaPhase_OTA_ERROR)     return ERROR_CODE::FAILED_TO_UPDATE;
            if (s.staged)                                { available = true;  break; }
            break;   // some other settled phase: nothing further in flight
        }
        if (std::chrono::steady_clock::now() >= deadline)
            return ERROR_CODE::UNKNOWN_FAILURE;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    impl_->refresh_device_state();
    return ERROR_CODE::SUCCESS;
}

// Drive the whole A/B update: acquire (URL download or local-file sideload) ->
// verify -> apply -> device reboot -> reconnect -> confirm the new version is
// running. Blocking.
ERROR_CODE Device::update(const std::string& url,
                          const std::function<void(const UpdateStatus&)>& on_progress) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    // Host-side illegal-state pre-check (contract: decided without touching
    // the device): an update is legal from IDLE, or while only a live host
    // stream is running (paused below), never while the device is
    // recording or uploading (both now fold into STREAMING, so check the raw
    // facets), and never on an MCAP replay session (there is no device).
    if ((impl_->state != DEVICE_STATE::IDLE &&
         impl_->state != DEVICE_STATE::STREAMING) ||
        impl_->device_rec || impl_->dev_recording || impl_->dev_uploading ||
        impl_->init.input_type == INPUT_TYPE::MCAP)
        return ERROR_CODE::INVALID_FUNCTION_CALL;

    const uint32_t old_version = impl_->info.firmware_version;

    // A url that names an existing local file means sideload: push the .eff
    // over the control link instead of asking the device to download it.
    struct stat st{};
    const bool sideload = !url.empty() && ::stat(url.c_str(), &st) == 0 &&
                          S_ISREG(st.st_mode);

    // A host-file recording cannot meaningfully span the update (transport
    // close + firmware reboot): finish the .mcap now instead of silently
    // concatenating two firmware sessions, or orphaning the writer if the
    // post-apply reconnect fails.
    impl_->stop_host_rec();

    // The device applies OTA from IDLE; pause the local streaming session first.
    const bool was_streaming = (impl_->state == DEVICE_STATE::STREAMING);
    if (was_streaming) impl_->stop_streaming();
    impl_->state = DEVICE_STATE::UPDATING;

    // Restore streaming on every early exit (the update never got to apply).
    auto bail = [&](ERROR_CODE ec) -> ERROR_CODE {
        impl_->state = DEVICE_STATE::IDLE;
        if (was_streaming) {
            // Resume at the validated streaming resolution the session ran at,
            // not the device's cached/persisted config (they can differ).
            Resolution res = impl_->stream_res;
            ERROR_CODE rc = (impl_->init.input_type == INPUT_TYPE::STREAM)
                                ? impl_->start_streaming_online(res)
                                : impl_->start_streaming(res);
            if (rc == ERROR_CODE::SUCCESS) impl_->state = DEVICE_STATE::STREAMING;
        }
        impl_->refresh_device_state();
        return ec;
    };

    // 1. Get the image onto the device (sideload push or device-side download).
    if (sideload) {
        // Sideload progress reads as the DOWNLOADING phase, the device is
        // downloading from this host, just over the control link.
        auto push_progress = [&](uint64_t sent, uint64_t total) {
            if (!on_progress) return;
            UpdateStatus u;
            u.active   = true;
            u.state    = UPDATE_STATE::DOWNLOADING;
            u.progress = total ? (int)((sent * 100) / total) : -1;
            on_progress(u);
        };
        ERROR_CODE ec = impl_->ota_push(url, push_progress);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
    } else {
        // Only the "" = configured-server path is gated by check_update():
        // OtaCheck consults the device's CONFIGURED server (the proto message
        // carries no URL), so gating an explicit caller URL on it would never
        // attempt the download (e.g. a LAN mirror while the configured server
        // is unreachable). An explicit URL goes straight to ota_download.
        if (url.empty()) {
            bool available = false;
            ERROR_CODE ec = check_update(available);
            if (ec != ERROR_CODE::SUCCESS) return bail(ec);
            if (!available)                return bail(ERROR_CODE::DEVICE_UP_TO_DATE);
        }
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_ota_download_tag;
        std::snprintf(req.body.ota_download.base_url,
                      sizeof req.body.ota_download.base_url, "%s", url.c_str());
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
    }

    // 2. Device-side verify, reporting progress until the image is READY.
    // The acquire verbs above are asynchronous on the device and its OTA
    // status file may still hold a stale terminal phase from an earlier
    // attempt (see check_update): a FAILED_TO_UPDATE last_error is only
    // believed once this operation has been seen active, or after a start
    // grace with no activity at all.
    auto deadline    = std::chrono::steady_clock::now() + std::chrono::minutes(15);
    auto start_grace = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool seen_active = false;
    for (;;) {
        UpdateStatus u;
        ERROR_CODE ec = get_update_status(u);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
        if (on_progress) on_progress(u);
        if (u.active) seen_active = true;
        if (u.active && u.state == UPDATE_STATE::READY_TO_APPLY) break;
        if ((u.last_error == ERROR_CODE::FAILED_TO_UPDATE ||
             u.last_error == ERROR_CODE::DEVICE_UP_TO_DATE) &&
            (seen_active || std::chrono::steady_clock::now() >= start_grace))
            return bail(u.last_error);
        if (std::chrono::steady_clock::now() >= deadline) return bail(ERROR_CODE::FAILED_TO_UPDATE);
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // 3. Apply, the device writes the boot slot and reboots underneath the host.
    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_ota_apply_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
    }

    // 4. Reconnect after the reboot, then confirm what's running.
    const InitParameters saved = impl_->init;
    impl_->close_transport();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    ERROR_CODE reopened = ERROR_CODE::DEVICE_NOT_DETECTED;
    auto reconnect_deadline = std::chrono::steady_clock::now() + std::chrono::minutes(3);
    while (std::chrono::steady_clock::now() < reconnect_deadline) {
        reopened = open(saved);
        if (reopened == ERROR_CODE::SUCCESS) break;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
    if (reopened != ERROR_CODE::SUCCESS) return ERROR_CODE::FAILED_TO_UPDATE;

    // Rolled back or same version => the update did not take.
    UpdateStatus u;
    if (get_update_status(u) == ERROR_CODE::SUCCESS && on_progress) on_progress(u);
    if (u.last_error == ERROR_CODE::FAILED_TO_UPDATE ||
        impl_->info.firmware_version <= old_version)
        return ERROR_CODE::FAILED_TO_UPDATE;
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::abort_update() {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_ota_abort_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
    impl_->refresh_device_state();
    return ec;
}

// ---- wireless -----------------------------------------------------------------------

ERROR_CODE Device::wifi_add(const std::string& ssid, const std::string& psk,
                            const std::string& country) {
    if (!is_open())   return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (ssid.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_wifi_add_tag;
    std::snprintf(req.body.wifi_add.ssid, sizeof req.body.wifi_add.ssid, "%s", ssid.c_str());
    std::snprintf(req.body.wifi_add.psk,  sizeof req.body.wifi_add.psk,  "%s", psk.c_str());
    // ISO 3166-1 alpha-2 regdomain ("US" unlocks 5 GHz); "" keeps the current
    // one. The wire field is 3 chars + NUL, snprintf truncates anything longer.
    std::snprintf(req.body.wifi_add.country, sizeof req.body.wifi_add.country,
                  "%s", country.c_str());
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    if (ec == ERROR_CODE::SUCCESS) {
        // Optimistic dedup'd update; refresh_wireless() then replaces the list
        // with device truth (WifiList), so a re-add / psk rotation never
        // accumulates duplicates.
        auto& v = impl_->info.wireless.saved_networks;
        if (std::find(v.begin(), v.end(), ssid) == v.end()) v.push_back(ssid);
        impl_->refresh_wireless();
    }
    return ec;
}

ERROR_CODE Device::wifi_remove(const std::string& ssid) {
    if (!is_open())   return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (ssid.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_wifi_remove_tag;
    std::snprintf(req.body.wifi_remove.ssid, sizeof req.body.wifi_remove.ssid, "%s", ssid.c_str());
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    if (ec == ERROR_CODE::SUCCESS) {
        auto& v = impl_->info.wireless.saved_networks;
        for (auto it = v.begin(); it != v.end();) it = (*it == ssid) ? v.erase(it) : it + 1;
        impl_->refresh_wireless();
    }
    return ec;
}

ERROR_CODE Device::wifi_select(const std::string& ssid) {
    if (!is_open())   return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (ssid.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_wifi_select_tag;
    std::snprintf(req.body.wifi_select.ssid, sizeof req.body.wifi_select.ssid, "%s", ssid.c_str());
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    if (ec == ERROR_CODE::SUCCESS) impl_->refresh_wireless();
    return ec;
}

// ---- system --------------------------------------------------------------------------

ERROR_CODE Device::set_ble_password(const std::string& old_password,
                                    const std::string& new_password) {
    if (!is_open())           return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (new_password.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    // Over BLE the device demands the old password; over USB an empty
    // old_password resets a forgotten one (physical access is the credential).
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_set_ble_password_tag;
    std::snprintf(req.body.set_ble_password.old_password,
                  sizeof req.body.set_ble_password.old_password, "%s", old_password.c_str());
    std::snprintf(req.body.set_ble_password.new_password,
                  sizeof req.body.set_ble_password.new_password, "%s", new_password.c_str());
    WireResponse resp;
    return impl_->call(req, resp);   // AUTH_FAILED -> INVALID_PASSWORD
}


ERROR_CODE Device::get_storage(uint64_t& free_bytes, uint64_t& total_bytes) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_get_storage_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_storage_info_tag);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    free_bytes  = resp.body.storage_info.free_bytes;
    total_bytes = resp.body.storage_info.total_bytes;
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::sync_time() {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body            = ef_v1_Request_set_time_tag;
    req.body.set_time.wall_ns = host_now_ns();
    WireResponse resp;
    return impl_->call(req, resp);
}

ERROR_CODE Device::get_device_time(Timestamp& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                = ef_v1_Request_time_sync_tag;
    req.body.time_sync.host_tx_ns = host_now_ns();
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    if (resp.which_body != ef_v1_Response_time_sync_reply_tag)
        return ERROR_CODE::COMMUNICATION_ERROR;
    out.data_ns = resp.body.time_sync_reply.device_wall_ns;
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::set_location(double latitude, double longitude,
                                double altitude, double covariance_diag) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                          = ef_v1_Request_set_location_tag;
    req.body.set_location.has_location      = true;
    req.body.set_location.location.latitude        = latitude;
    req.body.set_location.location.longitude       = longitude;
    req.body.set_location.location.altitude        = altitude;
    req.body.set_location.location.covariance_diag = covariance_diag;
    WireResponse resp;
    return impl_->call(req, resp);
}

ERROR_CODE Device::get_location(Location& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_get_location_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    if (resp.which_body != ef_v1_Response_location_tag)
        return ERROR_CODE::COMMUNICATION_ERROR;
    out.latitude        = resp.body.location.latitude;
    out.longitude       = resp.body.location.longitude;
    out.altitude        = resp.body.location.altitude;
    out.covariance_diag = resp.body.location.covariance_diag;
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::reboot() {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_reboot_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    // Only tear the session down when the device actually took the reboot: an
    // accepted verb (or a link that died mid-request, the device may reboot
    // before replying) means the transport is gone. The device firmware
    // refuses the reboot (UNSUPPORTED/error, session kept) while a
    // capture is active, the caller sees that and the session stays usable.
    if (ec == ERROR_CODE::SUCCESS || ec == ERROR_CODE::COMMUNICATION_ERROR)
        close();
    return ec;
}

ERROR_CODE Device::set_configuration(int width, int height, int fps,
                                     COMPRESSION_MODE codec) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                        = ef_v1_Request_configure_tag;
    // mode left UNKNOWN => a geometry/codec-only partial update that doesn't
    // touch the capture mode. The device firmware applies it
    // IDLE-only and refuses while streaming/recording, so a live session is
    // never disrupted.
    req.body.configure.has_video          = true;
    req.body.configure.video.width        = (uint32_t)width;
    req.body.configure.video.height       = (uint32_t)height;
    req.body.configure.video.fps          = (uint32_t)fps;
    req.body.configure.video.pixel_format = ef_v1_PixelFormat_NV12;
    req.body.configure.video.codec        = pb_codec(codec);
    req.body.configure.video.quality      = quality_for(codec);
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::SESSION);
    // Same INVALID_PARAMETER disambiguation configure_session() uses: the wire
    // collapses every bad knob into INVALID_PARAMETER; the reason string names it.
    return disambiguate_config_error(ec, resp);
}

// ---- enum -> string ---------------------------------------------------------------------

const char* to_string(ERROR_CODE e) {
    switch (e) {
        case ERROR_CODE::SUCCESS:                        return "SUCCESS";
        case ERROR_CODE::SENSOR_CONFIGURATION_CHANGED:   return "SENSOR_CONFIGURATION_CHANGED";
        case ERROR_CODE::CONFIGURATION_FALLBACK:         return "CONFIGURATION_FALLBACK";
        case ERROR_CODE::DEVICE_REBOOTING:               return "DEVICE_REBOOTING";
        case ERROR_CODE::FAILED_TO_UPDATE:               return "FAILED_TO_UPDATE";
        case ERROR_CODE::DEVICE_UP_TO_DATE:              return "DEVICE_UP_TO_DATE";
        case ERROR_CODE::DEVICE_NOT_DETECTED:            return "DEVICE_NOT_DETECTED";
        case ERROR_CODE::DEVICE_NOT_INITIALIZED:         return "DEVICE_NOT_INITIALIZED";
        case ERROR_CODE::DEVICE_NOT_AVAILABLE:           return "DEVICE_NOT_AVAILABLE";
        case ERROR_CODE::INVALID_FIRMWARE:               return "INVALID_FIRMWARE";
        case ERROR_CODE::INVALID_FUNCTION_CALL:          return "INVALID_FUNCTION_CALL";
        case ERROR_CODE::INVALID_PASSWORD:               return "INVALID_PASSWORD";
        case ERROR_CODE::INSUFFICIENT_PERMISSIONS:       return "INSUFFICIENT_PERMISSIONS";
        case ERROR_CODE::UNSUPPORTED:                    return "UNSUPPORTED";
        case ERROR_CODE::INVALID_RESOLUTION:             return "INVALID_RESOLUTION";
        case ERROR_CODE::INVALID_FPS:                    return "INVALID_FPS";
        case ERROR_CODE::UNSUPPORTED_COMPRESSION:        return "UNSUPPORTED_COMPRESSION";
        case ERROR_CODE::CALIBRATION_FILE_NOT_AVAILABLE: return "CALIBRATION_FILE_NOT_AVAILABLE";
        case ERROR_CODE::INVALID_CALIBRATION_FILE:       return "INVALID_CALIBRATION_FILE";
        case ERROR_CODE::POTENTIAL_CALIBRATION_ISSUE:    return "POTENTIAL_CALIBRATION_ISSUE";
        case ERROR_CODE::LOW_USB_BANDWIDTH:              return "LOW_USB_BANDWIDTH";
        case ERROR_CODE::CANNOT_START_CAMERA_STREAM:     return "CANNOT_START_CAMERA_STREAM";
        case ERROR_CODE::COMMUNICATION_ERROR:            return "COMMUNICATION_ERROR";
        case ERROR_CODE::WIFI_NOT_CONNECTED:             return "WIFI_NOT_CONNECTED";
        case ERROR_CODE::CORRUPTED_FRAME:                return "CORRUPTED_FRAME";
        case ERROR_CODE::SESSION_RECORDING_ERROR:        return "SESSION_RECORDING_ERROR";
        case ERROR_CODE::END_OF_BUFFER:                  return "END_OF_BUFFER";
        case ERROR_CODE::GRAB_TIMEOUT:                   return "GRAB_TIMEOUT";
        case ERROR_CODE::STORAGE_FULL:                   return "STORAGE_FULL";
        case ERROR_CODE::RECORDING_NOT_FOUND:            return "RECORDING_NOT_FOUND";
        case ERROR_CODE::RECORDING_ALREADY_EXISTS:       return "RECORDING_ALREADY_EXISTS";
        case ERROR_CODE::UNKNOWN_FAILURE:                return "UNKNOWN_FAILURE";
    }
    return "UNKNOWN_FAILURE";
}

const char* to_string(MODEL m) {
    switch (m) { case MODEL::M1: return "M1"; }
    return "M1";
}

const char* to_string(INPUT_TYPE t) {
    switch (t) {
        case INPUT_TYPE::USB:    return "USB";
        case INPUT_TYPE::STREAM: return "STREAM";
        case INPUT_TYPE::MCAP:   return "MCAP";
    }
    return "USB";
}

const char* to_string(RESOLUTION r) {
    switch (r) {
        case RESOLUTION::HD1200: return "HD1200";
        case RESOLUTION::HD1080: return "HD1080";
        case RESOLUTION::SVGA:   return "SVGA";
        case RESOLUTION::AUTO:   return "AUTO";
    }
    return "AUTO";
}

const char* to_string(COMPRESSION_MODE c) {
    switch (c) {
        case COMPRESSION_MODE::RAW:      return "RAW";
        case COMPRESSION_MODE::H264:     return "H264";
        case COMPRESSION_MODE::H264_HQ:  return "H264_HQ";
        case COMPRESSION_MODE::H265:     return "H265";
        case COMPRESSION_MODE::H265_HQ:  return "H265_HQ";
    }
    return "RAW";
}

const char* to_string(SENSOR_TYPE t) {
    return t == SENSOR_TYPE::GYROSCOPE ? "GYROSCOPE" : "ACCELEROMETER";
}

const char* to_string(SENSOR_UNIT u) {
    switch (u) {
        case SENSOR_UNIT::M_SEC_2: return "M_SEC_2";
        case SENSOR_UNIT::DEG_SEC: return "DEG_SEC";
        case SENSOR_UNIT::CELSIUS: return "CELSIUS";
        case SENSOR_UNIT::HERTZ:   return "HERTZ";
    }
    return "M_SEC_2";
}

const char* to_string(LENS_DISTORTION_MODEL m) {
    switch (m) { case LENS_DISTORTION_MODEL::DS: return "DS"; }
    return "DS";
}

const char* to_string(FLIP_MODE f) {
    switch (f) {
        case FLIP_MODE::ON:   return "ON";
        case FLIP_MODE::OFF:  return "OFF";
        case FLIP_MODE::AUTO: return "AUTO";
    }
    return "OFF";
}

const char* to_string(VIEW v) {
    switch (v) {
        case VIEW::GRAY: return "GRAY";
        case VIEW::BGR:  return "BGR";
        case VIEW::RGB:  return "RGB";
        case VIEW::BGRA: return "BGRA";
        case VIEW::RGBA: return "RGBA";
        case VIEW::NV12: return "NV12";
    }
    return "BGR";
}

const char* to_string(DEVICE_STATE s) {
    switch (s) {
        case DEVICE_STATE::CLOSED:       return "CLOSED";
        case DEVICE_STATE::IDLE:         return "IDLE";
        case DEVICE_STATE::STREAMING:    return "STREAMING";
        case DEVICE_STATE::UPDATING:     return "UPDATING";
    }
    return "CLOSED";
}

const char* to_string(SENSOR_STATE s) {
    return s == SENSOR_STATE::AVAILABLE ? "AVAILABLE" : "NOT_AVAILABLE";
}

const char* to_string(CAMERA_STATE s) {
    return s == CAMERA_STATE::AVAILABLE ? "AVAILABLE" : "NOT_AVAILABLE";
}

const char* to_string(TIME_REFERENCE r) {
    return r == TIME_REFERENCE::IMAGE ? "IMAGE" : "CURRENT";
}

const char* to_string(COORDINATE_SYSTEM c) {
    switch (c) {
        case COORDINATE_SYSTEM::IMAGE:                   return "IMAGE";
        case COORDINATE_SYSTEM::LEFT_HANDED_Y_UP:        return "LEFT_HANDED_Y_UP";
        case COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP:       return "RIGHT_HANDED_Y_UP";
        case COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP:       return "RIGHT_HANDED_Z_UP";
        case COORDINATE_SYSTEM::LEFT_HANDED_Z_UP:        return "LEFT_HANDED_Z_UP";
        case COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD: return "RIGHT_HANDED_Z_UP_X_FWD";
    }
    return "IMAGE";
}

const char* to_string(MEM m) {
    switch (m) {
        case MEM::CPU:  return "CPU";
        case MEM::GPU:  return "GPU";
        case MEM::BOTH: return "BOTH";
    }
    return "CPU";
}

const char* to_string(MOTION_STATE s) {
    switch (s) {
        case MOTION_STATE::STATIC:  return "STATIC";
        case MOTION_STATE::MOVING:  return "MOVING";
        case MOTION_STATE::FALLING: return "FALLING";
    }
    return "STATIC";
}

const char* to_string(RECORDING_TARGET t) {
    return t == RECORDING_TARGET::DEVICE_LOCAL ? "DEVICE_LOCAL" : "HOST_FILE";
}

const char* to_string(UPLOAD_STATE s) {
    return s == UPLOAD_STATE::RUNNING ? "RUNNING" : "OFF";
}

const char* to_string(UPDATE_STATE s) {
    switch (s) {
        case UPDATE_STATE::DOWNLOADING:    return "DOWNLOADING";
        case UPDATE_STATE::VERIFYING:      return "VERIFYING";
        case UPDATE_STATE::READY_TO_APPLY: return "READY_TO_APPLY";
        case UPDATE_STATE::APPLYING:       return "APPLYING";
    }
    return "DOWNLOADING";
}

const char* to_string(MAT_TYPE t) {
    switch (t) {
        case MAT_TYPE::U8_C1: return "U8_C1";
        case MAT_TYPE::U8_C3: return "U8_C3";
        case MAT_TYPE::U8_C4: return "U8_C4";
        case MAT_TYPE::NV12:  return "NV12";
    }
    return "NV12";
}

}  // namespace ef
