////////////////////////////////////////////////////////////////////////////////
//
// File:      internal/device_impl.hpp
// Purpose:   ef::Device::Impl + shared internals (internal header).
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

#ifndef EF_DEVICE_IMPL_HPP
#define EF_DEVICE_IMPL_HPP

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

// Transport/reader internals speak ef::Status (fine-grained ISOC result set);
// translated to ERROR_CODE at this layer, nothing above speaks Status.
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
// (ef.v1 -> ef_v1_*). Enum values, body tags, and macros keep generated names.
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

inline Resolution get_resolution(RESOLUTION r) {
    switch (r) {
        case RESOLUTION::HD1200: return {1920, 1200};
        case RESOLUTION::HD1080: return {1920, 1080};
        case RESOLUTION::SVGA:   return {960,  600};
        case RESOLUTION::AUTO:   return {0, 0};
    }
    return {0, 0};
}


// The M1's USB identity (vendor interface; same IDs the connection classes latch).
inline constexpr uint16_t kVid = 0x39c5;
inline constexpr uint16_t kPid = 0x0001;

inline uint64_t host_now_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ---- wire <-> enum maps ------------------------------------------------------

inline WireCodec pb_codec(COMPRESSION_MODE c) {
    switch (c) {
        case COMPRESSION_MODE::H264:
        case COMPRESSION_MODE::H264_HQ: return ef_v1_Codec_H264;
        case COMPRESSION_MODE::H265:
        case COMPRESSION_MODE::H265_HQ: return ef_v1_Codec_H265;
        case COMPRESSION_MODE::RAW:     return ef_v1_Codec_CODEC_RAW;
    }
    return ef_v1_Codec_CODEC_RAW;
}

// _HQ presets ride the same wire codec plus explicit encoder quality;
// 90 is perceptually near-lossless. Tune here.
inline constexpr uint32_t kHqQuality = 90;

inline uint32_t quality_for(COMPRESSION_MODE c) {
    return (c == COMPRESSION_MODE::H264_HQ || c == COMPRESSION_MODE::H265_HQ)
               ? kHqQuality
               : 0;   // 0 = device default
}

// HQ never round-trips: the device reports only the codec, so a read-back
// collapses to the base mode. The requested mode survives in InitParameters.
inline COMPRESSION_MODE compression_from(WireCodec c) {
    switch (c) {
        case ef_v1_Codec_H264: return COMPRESSION_MODE::H264;
        case ef_v1_Codec_H265: return COMPRESSION_MODE::H265;
        default:               return COMPRESSION_MODE::RAW;
    }
}

// What kind of request produced a device error; selects the ERROR_CODE subset
// to translate into, since the wire enum is smaller than ERROR_CODE.
enum class Ctx {
    CONTROL,     // queries, health, time, reboot
    SESSION,     // configure / start_stream during open()
    RECORDING,   // device-local recording verbs (incl. download)
    UPLOAD,      // upload verbs
    UPDATE,      // OTA verbs
    WIFI,        // wifi verbs; BUSY surfaces as a retryable ERROR_CODE::BUSY
};

// Device ErrorCode + request context -> ERROR_CODE. The device's message
// string (Response.message) still carries the specific reason.
inline ERROR_CODE err_from(WireErrorCode c, Ctx ctx) {
    // Codes with one meaning regardless of what was asked.
    switch (c) {
        case ef_v1_ErrorCode_SUCCESS:                  return ERROR_CODE::SUCCESS;
        case ef_v1_ErrorCode_NOT_OPENED:               return ERROR_CODE::DEVICE_NOT_INITIALIZED;
        case ef_v1_ErrorCode_CORRUPTED_FRAME:          return ERROR_CODE::CORRUPTED_FRAME;
        case ef_v1_ErrorCode_BANDWIDTH_EXCEEDED:
        case ef_v1_ErrorCode_NOT_SUPERSPEED:           return ERROR_CODE::LOW_USB_BANDWIDTH;
        case ef_v1_ErrorCode_CAMERA_UNAVAILABLE:       return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        case ef_v1_ErrorCode_DEVICE_SERVICE_UNREACHABLE: return ERROR_CODE::DEVICE_NOT_AVAILABLE;
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
                // delete of the session that is recording right now; retry
                // after stop (same shape as the WIFI-context BUSY).
                case ef_v1_ErrorCode_BUSY:              return ERROR_CODE::DEVICE_BUSY;
                case ef_v1_ErrorCode_INVALID_PARAMETER:
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
        case Ctx::WIFI:
            switch (c) {
                // The device is recording/livestreaming; the caller can retry
                // once it stops, so keep BUSY distinct from a generic failure.
                case ef_v1_ErrorCode_BUSY:              return ERROR_CODE::DEVICE_BUSY;
                case ef_v1_ErrorCode_WIFI_NOT_CONNECTED: return ERROR_CODE::WIFI_NOT_CONNECTED;
                // wifi remove of an ssid that was never saved.
                case ef_v1_ErrorCode_NOT_FOUND:
                case ef_v1_ErrorCode_INVALID_PARAMETER:
                case ef_v1_ErrorCode_INVALID_STATE:     return ERROR_CODE::INVALID_FUNCTION_CALL;
                default:                                return ERROR_CODE::UNKNOWN_FAILURE;
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
inline ERROR_CODE open_err(Status s) {
    switch (s) {
        case Status::SUCCESS:          return ERROR_CODE::SUCCESS;
        case Status::DEVICE_NOT_FOUND: return ERROR_CODE::DEVICE_NOT_DETECTED;
        case Status::NO_STREAM:        return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        // libusb EACCES on open/claim, the actionable udev/permissions case.
        case Status::INSUFFICIENT_PERMISSIONS: return ERROR_CODE::INSUFFICIENT_PERMISSIONS;
        // Interface claim refused: another client already has the device open.
        case Status::BUSY:             return ERROR_CODE::DEVICE_BUSY;
        default:                       return ERROR_CODE::DEVICE_NOT_AVAILABLE;
    }
}

// SESSION maps a bad-Configure INVALID_PARAMETER to INVALID_RESOLUTION; re-split
// it from the device's reason string so codec/fps get their own codes and
// INVALID_RESOLUTION stays the geometry case. Inert until an out-of-caps reject.
inline ERROR_CODE disambiguate_config_error(ERROR_CODE ec, const WireResponse& resp) {
    if (ec != ERROR_CODE::INVALID_RESOLUTION ||
        resp.code != ef_v1_ErrorCode_INVALID_PARAMETER)
        return ec;
    if (std::strstr(resp.message, "codec")) return ERROR_CODE::UNSUPPORTED_COMPRESSION;
    if (std::strstr(resp.message, "fps"))   return ERROR_CODE::INVALID_FPS;
    return ERROR_CODE::INVALID_RESOLUTION;
}

// StreamAssembler::grab results. TIMEOUT is the non-fatal keep-looping case;
// END_OF_STREAM means the transport died underneath the session.
inline ERROR_CODE grab_err(Status s) {
    switch (s) {
        case Status::SUCCESS:         return ERROR_CODE::SUCCESS;
        case Status::TIMEOUT:         return ERROR_CODE::GRAB_TIMEOUT;
        case Status::CORRUPTED_FRAME: return ERROR_CODE::CORRUPTED_FRAME;
        default:                      return ERROR_CODE::COMMUNICATION_ERROR;
    }
}

// ---- decode + colour convert (retrieve_image) --------------------------------

#ifdef EF_WITH_FFMPEG
inline AVPixelFormat av_of(VIEW v) {
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

inline MAT_TYPE mat_type_of(VIEW v) {
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
        // A single decode thread (FFmpeg's default) can't keep 1200p30 HEVC
        // real-time, and falling behind triggers a drop-until-IDR resync stall
        // (no PLI back-channel over USB). Frame threading adds one frame of
        // latency per thread, so 2 is the sweet spot: real-time at ~66 ms.
        dec->thread_count = 2;
        if (avcodec_open2(dec, codec, nullptr) < 0) { avcodec_free_context(&dec); return false; }
        pkt = av_packet_alloc(); frm = av_frame_alloc();
        cur = c; have = pkt && frm;
        return have;
    }
    bool convert(const uint8_t* const src[4], const int srcstride[4], int w, int h,
                 AVPixelFormat srcfmt, VIEW view, std::vector<uint8_t>& buf, int* step) {
        AVPixelFormat df = av_of(view);
        // The H.264/H.265 decoder reports full-range 4:2:0 as the deprecated
        // YUVJ420P. Remap it to YUV420P and signal full range explicitly below,
        // so swscale does not print the "deprecated pixel format" warning. The
        // converted pixels are identical.
        bool src_full = true;
        AVPixelFormat eff;
        switch (srcfmt) {
            case AV_PIX_FMT_YUVJ420P: eff = AV_PIX_FMT_YUV420P; break;
            case AV_PIX_FMT_YUVJ422P: eff = AV_PIX_FMT_YUV422P; break;
            case AV_PIX_FMT_YUVJ444P: eff = AV_PIX_FMT_YUV444P; break;
            case AV_PIX_FMT_YUVJ440P: eff = AV_PIX_FMT_YUV440P; break;
            default: eff = srcfmt; src_full = false; break;
        }
        if (!sws || sw != w || sh != h || sfmt != (int)srcfmt || dfmt != (int)df) {
            if (sws) sws_freeContext(sws);
            sws = sws_getContext(w, h, eff, w, h, df, SWS_BILINEAR, nullptr, nullptr, nullptr);
            sw = w; sh = h; sfmt = (int)srcfmt; dfmt = (int)df;
            if (sws && src_full) {
                // Full-range in and out; neutral brightness/contrast/saturation.
                const int* coef = sws_getCoefficients(SWS_CS_DEFAULT);
                sws_setColorspaceDetails(sws, coef, 1, coef, 1, 0, 1 << 16, 1 << 16);
            }
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
// back-pressures the grab loop. Writes an MCAP container (video + IMU) in the
// same schema/topic layout as a device-local recording (see mcap.hpp), so both
// targets replay through the same reader.
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
    std::string       last_dev_msg;    // Response.message from the last failed call

    // USB isoc StreamReader, WiFi UdpStreamReader, or MCAP replay behind one
    // shared consumer API (grab/current_video/drain_imu). shared_ptr + reader_mtx
    // so a concurrent update()/close()/reboot() can't free a reader while a grab()
    // is blocked in it: teardown swaps the pointer out under the lock and stop()
    // wakes the waiter, then the last snapshot holder frees it.
    std::shared_ptr<internal::StreamAssembler> reader;
    mutable std::mutex            reader_mtx;
    VideoDecoder                  decoder;
    // shared_ptr + data_mtx (below), same discipline as `reader`: a control-thread
    // reset frees the recorder only once no data-plane call still holds a snapshot.
    std::shared_ptr<HostRecorder> host_rec;
    std::string                   device_rec_name;   // active DEVICE_LOCAL session
    bool                          device_rec = false;

    // Raw activity facets, refreshed by refresh_device_state() alongside `state`.
    // They preserve the recording-vs-uploading distinction that DEVICE_STATE
    // collapses into STREAMING, keeping the illegal-state guards precise.
    bool                          dev_recording = false;  // device-side capture in flight (FSM COLLECT / reported_status RECORDING)
    bool                          dev_uploading = false;  // device-side upload in flight (reported_status UPLOADING)

    // Fault surfaced by the device firmware, refreshed alongside `state`. The 4-value
    // DEVICE_STATE has no FAULT value (SAFE projects to CLOSED per the state model),
    // so a client observes a device fault through these fields, not the state enum.
    bool                          device_fault_latch = false;
    std::string                   device_last_fault;

    Resolution stream_res{};   // validated at open(), applied when streaming starts

    // IMU drained by grab() while a host recording runs (recorder must see every
    // sample even if the app never calls retrieve_imu); served to the next
    // retrieve_imu ahead of fresh samples. Consumer-thread only.
    std::vector<ImuSample> imu_backlog;
    uint64_t               imu_backlog_dropped = 0;
    static constexpr size_t IMU_BACKLOG_CAP = 65536;

    // FLIP_MODE::AUTO latch: -1 unresolved, 0 upright, 1 mounted upside-down.
    // Resolved from gravity in the raw (IMAGE-frame) accelerometer stream.
    int flip_latched = -1;

    uint64_t last_frame_ts_ns = 0;
    uint32_t corr = 0;

    // Serializes control round trips against transport teardown (libusb_close or
    // BlueZ teardown under an in-flight request is UB). Multi-round-trip loops
    // (download_recording, ota_push, polls) re-take it per call(), so a concurrent
    // close() aborts them at the next chunk boundary instead of crashing.
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
    // concurrent grab() snapshot survives), then stop the local copy; the last
    // holder frees it. Safe to call with no recorder.
    void stop_host_rec() {
        std::shared_ptr<HostRecorder> hr;
        { std::lock_guard<std::mutex> lk(data_mtx); hr = std::move(host_rec); }
        if (hr) hr->stop();
    }

    // FLIP_MODE::AUTO: latch orientation from one near-static gravity reading.
    // IMAGE frame has +y down, so at rest accel-y reads ~ -g upright, ~ +g
    // upside-down. Consumer-thread only (writes flip_latched).
    void latch_flip_from_accel(const float a[3]) {
        if (init.flip_mode != FLIP_MODE::AUTO || flip_latched >= 0) return;
        double mag = std::sqrt((double)a[0] * a[0] + (double)a[1] * a[1] +
                               (double)a[2] * a[2]);
        if (std::fabs(mag - 9.81) > 0.8) return;
        if      (a[1] < -5.f) flip_latched = 0;
        else if (a[1] >  5.f) flip_latched = 1;
    }

    // One protobuf round trip: encode -> transport -> decode -> code check.
    // `ctx` selects the error translation; `expect` (optional) also requires
    // that response body tag.
    ERROR_CODE call(WireRequest& req, WireResponse& resp,
                    pb_size_t expect = 0, Ctx ctx = Ctx::CONTROL) {
        std::lock_guard<std::mutex> lk(ctl_mtx);
        ERROR_CODE ec = call_locked(req, resp, expect, ctx);
        // The device expires an idle authenticated session, because USB gives it
        // no client-disconnect event to end one on. A handle that sat idle can
        // therefore meet AUTH_REQUIRED mid-life through no fault of the caller.
        // Re-run the handshake once and retry, so the expiry is invisible instead
        // of surfacing as a spurious INVALID_PASSWORD.
        //
        // Only when this handle authenticated before: without that, a genuinely
        // wrong password would retry on every single call.
        if (ec == ERROR_CODE::INVALID_PASSWORD && authenticated && !reauthing) {
            reauthing = true;
            ERROR_CODE re = control_auth_locked(init.ble_password);
            reauthing = false;
            if (re == ERROR_CODE::SUCCESS) ec = call_locked(req, resp, expect, ctx);
        }
        return ec;
    }

    bool reauthing = false;   // guards the retry above against recursing

    // call()'s body. Assumes ctl_mtx is held, so the re-auth path can reuse it.
    ERROR_CODE call_locked(WireRequest& req, WireResponse& resp,
                           pb_size_t expect = 0, Ctx ctx = Ctx::CONTROL) {
        if (!connection) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
        req.corr_id = ++corr;
        uint8_t buf[8192];   // == proto::MAX_PAYLOAD (EFR_CTL_MSG_MAX): an
                             // OtaPushChunk request encodes to ~7.2 KB.
        pb_ostream_t os = pb_ostream_from_buffer(buf, sizeof buf);
        if (!pb_encode(&os, ef_v1_Request_fields, &req))
            return ERROR_CODE::UNKNOWN_FAILURE;
        std::string reply;
        uint8_t type = 0;
        last_dev_msg.clear();   // else a transport failure reports the PREVIOUS reason
        if (connection->request_raw(std::string((const char*)buf, os.bytes_written),
                                    reply, &type) != Status::SUCCESS)
            return ERROR_CODE::COMMUNICATION_ERROR;
        std::memset(&resp, 0, sizeof resp);
        pb_istream_t is = pb_istream_from_buffer((const uint8_t*)reply.data(), reply.size());
        if (!pb_decode(&is, ef_v1_Response_fields, &resp))
            return ERROR_CODE::COMMUNICATION_ERROR;
        if (resp.code != ef_v1_ErrorCode_SUCCESS) {
            // The device's message carries the specific reason a code collapsed. Keep it
            // for last_error_message(); an ERROR_CODE alone often is not actionable.
            last_dev_msg = resp.message;
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

    // StopStream, then poll get_state (bounded ~1 s) until the device leaves
    // STREAMING. Online (UDP) teardown is asynchronous, so a Configure right after
    // StopStream can race it and be rejected INVALID_STATE; settling first makes
    // the follow-up deterministic (one round trip, no sleep, when already idle).
    // A recording/upload owned by another client is not ours to interrupt.
    void stop_stream_and_settle() {
        refresh_device_state();
        if (dev_recording || dev_uploading) return;
        stop_stream_quiet();
        for (int i = 0; i < 10; ++i) {
            refresh_device_state();
            if (state != DEVICE_STATE::STREAMING) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
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

    // Device-truth state: ask the device FSM (get_state) and fold it into the
    // DEVICE_STATE cache. Best-effort: on failure the cache keeps its last value
    // (the public contract has no UNKNOWN state).
    void refresh_device_state() {
        if (!connection) return;
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_state_tag;
        WireResponse resp;
        if (call(req, resp, ef_v1_Response_state_info_tag) != ERROR_CODE::SUCCESS)
            return;
        const char* fsm = resp.body.state_info.state;         // authoritative device firmware state
        const char* dev = resp.body.state_info.device_state;  // reported_status (BLE projection)

        // Fault ride-along: the device firmware's fault_latch + last_fault arrive on the
        // same reply. Cache them so a client can observe a fault even though the
        // 4-value DEVICE_STATE deliberately has no FAULT value.
        device_fault_latch = resp.body.state_info.fault_latch;
        device_last_fault  = resp.body.state_info.last_fault;

        // Raw facets for the illegal-state guards. Recording shows as FSM COLLECT
        // (reported_status RECORDING echoes it); uploading only on reported_status.
        dev_recording = !std::strcmp(fsm, "COLLECT") || !std::strcmp(dev, "RECORDING");
        dev_uploading = !std::strcmp(dev, "UPLOADING");

        // Public 4-value projection of the authoritative FSM state (see docs
        // architecture/device-state-model). SAFE and a latched fault -> CLOSED;
        // transitional states (INIT/RESET/HEALTH_TEST/SLEEP) -> CLOSED; OTA ->
        // UPDATING; COLLECT/CAL or a live host stream -> STREAMING; IDLE -> IDLE.
        // CLOSED therefore means "present but not in a usable data state"; a client
        // disambiguates a fault via fault_latch/last_fault, not the state enum.
        auto r = reader_snapshot();
        const bool reader_live = (r && r->is_running());
        if      (device_fault_latch || !std::strcmp(fsm, "SAFE"))
                                            state = DEVICE_STATE::CLOSED;
        else if (!std::strcmp(fsm, "OTA"))  state = DEVICE_STATE::UPDATING;
        else if (reader_live || dev_recording || dev_uploading ||
                 !std::strcmp(fsm, "CAL"))  state = DEVICE_STATE::STREAMING;
        else if (!std::strcmp(fsm, "IDLE")) state = DEVICE_STATE::IDLE;
        else                                state = DEVICE_STATE::CLOSED;  // INIT/RESET/HEALTH_TEST/SLEEP
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
        // A bad Configure returns INVALID_PARAMETER; split it into codec / fps /
        // resolution from the device's reason string.
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
        // Reader must be listening BEFORE the device emits: isoc is fire-and-
        // forget and the first access unit carries the encoder's SPS/PPS. Miss
        // it and an encoded stream never decodes.
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
            if (ec != ERROR_CODE::SUCCESS) { r->stop(); stop_stream_quiet(); return ec; }
        }
        set_reader(std::move(r));
        return ERROR_CODE::SUCCESS;
    }

    // BLE + WiFi: configure + StartStream(ONLINE -> udp_host:udp_port over UDP)
    // + the host UDP reader bound on udp_port.
    ERROR_CODE start_streaming_online(const Resolution& res) {
        // A device left STREAMING by a crashed online session rejects Configure
        // with INVALID_STATE. stop_stream_and_settle() clears it, but teardown is
        // asynchronous, so retry across a few settle cycles; any other result
        // (success, or a genuine codec/resolution error) returns at once.
        ERROR_CODE ec = configure_session(res);
        for (int attempt = 0; attempt < 3 &&
                              ec == ERROR_CODE::INVALID_FUNCTION_CALL &&
                              !dev_recording && !dev_uploading; ++attempt) {
            stop_stream_and_settle();
            ec = configure_session(res);
        }
        if (ec != ERROR_CODE::SUCCESS) return ec;
        // Bind the UDP socket before StartStream, same reason the isoc reader
        // starts first: the opening access unit (SPS/PPS) is sent immediately
        // and never retransmitted.
        auto r = std::make_shared<internal::UdpStreamReader>();
        r->set_video_codec(codec_gate_id());   // drop-until-IDR resync classifier
        Status st = r->start(init.udp_port, /*video=*/true,
                             init.enable_imu, init.verbose);
        if (st != Status::SUCCESS) {
            stop_stream_quiet();   // never leave the device armed on our failure
            return ERROR_CODE::CANNOT_START_CAMERA_STREAM;
        }
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
            if (ec != ERROR_CODE::SUCCESS) { r->stop(); stop_stream_quiet(); return ec; }
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
        // Clear a stale stream left armed by a client that died mid-stream
        // (StopStream is harmless when idle). Done here, not at open(), so it
        // never touches another process's DEVICE_LOCAL recording.
        stop_stream_quiet();
        // udp_host picks WiFi/UDP for either control transport; BLE without one
        // has no data plane (isoc needs USB).
        ERROR_CODE ec;
        if (!init.udp_host.empty()) {
            ec = start_streaming_online(stream_res);
        } else if (init.input_type == INPUT_TYPE::STREAM) {
            return ERROR_CODE::INVALID_FUNCTION_CALL;
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
            r->stop();   // wakes any blocked grab(); freed by last snapshot holder
            // Only when a reader existed: StopStream == StopCollect on the device,
            // so an unconditional call would also stop a DEVICE_LOCAL recording.
            stop_stream_quiet();
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

    // Refresh the cached wireless snapshot from device truth (best-effort) so
    // get_device_information().wireless reflects a provisioning change: live
    // association from GetWifiStatus, saved networks from WifiList.
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
                // Back-compat: firmware predating WifiStatus.state (field 8)
                // leaves it empty, so derive the 3-state from the legacy
                // `connected` bool rather than show a blank state.
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

    // Every IMU drain goes through here so a host recording sees every sample
    // whether or not the app calls retrieve_imu. Samples stay raw IMAGE-frame;
    // the user-facing transform is in retrieve_imu. Consumer-thread only.
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

    // Control-plane auth: prove the password via PBKDF2 + HMAC-SHA256
    // challenge-response so it never crosses the wire. Runs once after link-up;
    // without it the device answers AUTH_REQUIRED to every gated verb.
    //
    // Not BLE-only: a locked USB link gates identically, so this is compiled
    // unconditionally and both transports use it.
    //
    // Records the outcome in `authenticated` so open() can succeed on a wrong
    // password (keeping info/state/storage reachable) without hiding from the
    // caller that gated verbs will refuse.
    //
    // Atomic because is_authenticated() reads it outside ctl_mtx while call()
    // writes it under the lock.
    std::atomic<bool> authenticated{false};

    ERROR_CODE control_auth(const std::string& pw) {
        std::lock_guard<std::mutex> lk(ctl_mtx);
        return control_auth_locked(pw);
    }

    // Assumes ctl_mtx is held. The handshake is two round trips that must not be
    // interleaved with another thread's, and call()'s retry path already holds
    // the lock when it needs to re-run this.
    ERROR_CODE control_auth_locked(const std::string& pw) {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_auth_challenge_tag;
        WireResponse resp;
        ERROR_CODE ec = call_locked(req, resp, ef_v1_Response_auth_challenge_tag);
        // Only AUTH_REQUIRED/AUTH_FAILED mean "wrong password". Everything else
        // (UNSUPPORTED, NOT_OPENED mid-OTA, transport loss) surfaces untranslated,
        // since no password was tested and INVALID_PASSWORD would mislead.
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
        ERROR_CODE rc = call_locked(areq, aresp);
        authenticated = (rc == ERROR_CODE::SUCCESS);
        return rc;
    }

    // Sideload a local .eff over the control link: OtaPushBegin announces
    // {name,total}, each OtaPushChunk carries <= 7168 B (fits the 8192 control-
    // frame cap), and the eof chunk triggers device-side reassembly + verify.
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
}  // namespace ef

#endif  // EF_DEVICE_IMPL_HPP
