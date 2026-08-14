////////////////////////////////////////////////////////////////////////////////
//
// File:      device_recording.cpp
// Purpose:   ef::Device recording + upload verbs.
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
#include "internal/recording_paths.hpp"

namespace ef {

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
        int rw, rh;
        { std::lock_guard<std::mutex> lk(impl_->info_mtx);
          rw = impl_->info.camera_configuration.resolution.width;
          rh = impl_->info.camera_configuration.resolution.height; }
        if (!rec->start(params.path, raw, fmt, rw, rh))
            return ERROR_CODE::SESSION_RECORDING_ERROR;
        { std::lock_guard<std::mutex> lk(impl_->data_mtx); impl_->host_rec = std::move(rec); }
        impl_->recording = params;
        return ERROR_CODE::SUCCESS;
    }

    // DEVICE_LOCAL: device records to eMMC, survives host disconnect. Host-side
    // illegal-state pre-check (decided without touching the device): a second
    // recording, or one during update/upload, is known-illegal locally. A concurrent
    // health sweep is caught device-side (BUSY/INVALID_STATE via Ctx::RECORDING).
    if (impl_->state == DEVICE_STATE::UPDATING)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    // A recording or upload already running is a transient holder, so say BUSY
    // ("retry once idle") rather than the catch-all wrong-state code, which gave
    // no hint that something else owns the device.
    if (impl_->device_rec || impl_->dev_recording || impl_->dev_uploading)
        return ERROR_CODE::DEVICE_BUSY;
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
        // Say so explicitly rather than letting the device infer intent from the
        // coordinates, which would drop a deliberate 0,0 as a default-constructed
        // struct.
        req.body.start_recording.location.state           = ef_v1_LocationState_LOCATION_SET;
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
    // Stop the device's recording session. DEVICE_LOCAL recordings are cross-process
    // (`record stop` runs elsewhere than `record start`), so this is NOT gated on
    // this handle's device_rec; an empty name stops the device's current session.
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_stop_recording_tag;
    std::snprintf(req.body.stop_recording.name,
                  sizeof req.body.stop_recording.name, "%s",
                  impl_->device_rec_name.c_str());   // "" => current session
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::RECORDING);
    impl_->device_rec = false;
    impl_->refresh_device_state();
    // The device rejects a stop with nothing recording (INVALID_FUNCTION_CALL);
    // surface that, unless we just stopped a host-file recording (a real stop).
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
    out->encrypted   = s.encrypted;
    // Out-of-range wire values (a future firmware's new reason) degrade to
    // UNSPECIFIED rather than aliasing an existing reason.
    out->stopped_reason =
        (s.stopped_reason >= ef_v1_StopReason_STOP_USER &&
         s.stopped_reason <= ef_v1_StopReason_STOP_DEVICE)
            ? (STOP_REASON)s.stopped_reason
            : STOP_REASON::UNSPECIFIED;
    out->partial = s.partial;
}

}  // namespace

ERROR_CODE Device::get_recording_status(RecordingStatus& out, const std::string& name) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    // Host-side recording: everything is local. Pin the recorder with a snapshot so
    // a concurrent close()/update()/disable_recording() can't free it mid-use.
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
    // "" = the active session: ours if one is running, else whichever the device
    // reports as recording.
    const std::string want = name.empty() ? impl_->device_rec_name : name;
    bool have = false;
    if (!want.empty()) {
        // A named session resolves off the device's filesystem, so it works for
        // sessions outside the 48 that list_recordings can carry.
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_recording_status_tag;
        std::snprintf(req.body.get_recording_status.name,
                      sizeof req.body.get_recording_status.name, "%s", want.c_str());
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_recording_status_tag,
                                    Ctx::RECORDING);
        if (ec == ERROR_CODE::SUCCESS) {
            recording_from_wire(resp.body.recording_status, &out);
            have = true;
        } else if (ec != ERROR_CODE::UNSUPPORTED) {
            return ec;                    // real failure, including NOT_FOUND
        }
        // UNSUPPORTED: firmware predating the verb, which is how firmware older than
        // COMMAND_NOT_FOUND reports an unrouted one. Newer firmware answers
        // COMMAND_NOT_FOUND instead, so a fallback added from here on must key on that.
        // Fall through to the list.
    }
    if (!have) {
        // Also the only way to answer "which session is live?", which no
        // name-keyed verb can. Carries only the 48 most recent sessions.
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_list_recordings_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_recording_list_tag,
                                    Ctx::RECORDING);
        if (ec != ERROR_CODE::SUCCESS) return ec;
        const WireRecordingList& L = resp.body.recording_list;
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
                    ERROR_CODE le = err_from(u.last_error, Ctx::UPLOAD_STATUS);
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

ERROR_CODE Device::download_recording(
    const std::string& name,
    const std::string& dest_path,
    std::string* saved_path,
    const std::function<void(const DownloadProgress&)>& on_progress) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    std::string out_path;
    {
        ERROR_CODE ec = internal::resolve_download_dest(name, dest_path, out_path);
        if (ec != ERROR_CODE::SUCCESS) return ec;
    }
    if (saved_path) *saved_path = out_path;

    // No name pre-check against list_recordings: that list carries only 48
    // sessions, so it refused names the device would have served. The
    // device now answers NOT_FOUND itself, and chunk 0 below is fetched before the
    // destination is opened, which is what keeps a bad name from leaving an empty
    // file behind.

    // Pull loop, mirror of the OTA sideload push: request a window at each offset,
    // device returns up to 7168 B per response, loop to eof. Rides the control
    // transport (USB or BLE) like every verb, so no WiFi needed.
    //
    // Chunks retry with backoff. A resume re-fetches chunk 0 and compares it
    // against the local prefix first, so a same-named re-recording is never
    // spliced onto stale bytes.
    const uint32_t kChunk = (uint32_t)sizeof(ef_v1_RecordingChunk{}.data.bytes);  // 7168
    const int kChunkRetries = 4;

    // Retry ONLY the transient stall (COMMUNICATION_ERROR): a permanent code
    // (recording gone, auth, closed device) re-fails identically, and retrying
    // INVALID_PASSWORD would re-run the auth handshake once per attempt.
    // Worst-case added latency at a real stall: 2.5 s of backoff.
    auto fetch = [&](uint64_t off, WireResponse& resp) -> ERROR_CODE {
        ERROR_CODE ec = ERROR_CODE::SUCCESS;
        for (int a = 0; a <= kChunkRetries; a++) {
            if (a) std::this_thread::sleep_for(std::chrono::milliseconds(250 * a));
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body = ef_v1_Request_download_recording_tag;
            std::snprintf(req.body.download_recording.name,
                          sizeof req.body.download_recording.name, "%s", name.c_str());
            req.body.download_recording.offset = off;
            req.body.download_recording.len    = kChunk;
            ec = impl_->call(req, resp, ef_v1_Response_recording_chunk_tag,
                             Ctx::RECORDING);
            if (ec != ERROR_CODE::COMMUNICATION_ERROR) return ec;
        }
        return ec;
    };

    // Resume decision: an existing partial resumes only when it proves identity
    // (a full first chunk matching the live file byte-for-byte; the static
    // metadata block carries session timestamps, so distinct recordings diverge
    // inside chunk 0) AND is not longer than the live file (a longer "partial"
    // is from some other file; appending would return SUCCESS on an over-long
    // splice). Anything else starts clean over the old file.
    // Chunk 0 is fetched FIRST, before the destination is created, so a name the
    // device rejects never leaves a 0-byte file behind. It replaces the round trip
    // the old list_recordings pre-check spent, and also serves the resume compare.
    uint64_t offset = 0;
    uint64_t total = 0;  // device-reported size, from the last chunk seen
    bool append = false;
    WireResponse first;
    {
        ERROR_CODE ec = fetch(0, first);
        if (ec != ERROR_CODE::SUCCESS) return ec;
    }
    if (std::FILE* old = std::fopen(out_path.c_str(), "rb")) {
        uint8_t head[sizeof(ef_v1_RecordingChunk{}.data.bytes)];
        size_t have = std::fread(head, 1, sizeof head, old);
        std::fseek(old, 0, SEEK_END);
        long old_size = std::ftell(old);
        std::fclose(old);
        if (have == sizeof head && old_size > 0) {
            const ef_v1_RecordingChunk& c = first.body.recording_chunk;
            if (c.data.size == sizeof head &&
                (uint64_t)old_size <= c.total &&
                std::memcmp(c.data.bytes, head, sizeof head) == 0) {
                offset = (uint64_t)old_size;
                append = true;
            }
        }
    }

    std::FILE* f = std::fopen(out_path.c_str(), append ? "ab" : "wb");
    if (!f) return ERROR_CODE::DESTINATION_NOT_WRITABLE;

    // What a resume already has, reported before any bytes are written so a caller
    // can tell resumed bytes from ones it watched arrive.
    DownloadProgress prog;
    prog.resumed_from = append ? offset : 0;
    if (on_progress) {
        prog.received = offset;
        prog.total    = first.body.recording_chunk.total;
        on_progress(prog);
    }

    ERROR_CODE result = ERROR_CODE::SUCCESS;
    for (;;) {
        WireResponse resp;
        ERROR_CODE ec = fetch(offset, resp);
        if (ec != ERROR_CODE::SUCCESS) { result = ec; break; }

        const ef_v1_RecordingChunk& c = resp.body.recording_chunk;
        total = c.total;
        if (c.data.size) {
            if (std::fwrite(c.data.bytes, 1, c.data.size, f) != c.data.size) {
                result = ERROR_CODE::DESTINATION_NOT_WRITABLE;   // host disk, not the recording
                break;
            }
            offset += c.data.size;
        }
        if (on_progress) {
            prog.received = offset;
            prog.total    = total;
            on_progress(prog);
        }
        if (c.eof) break;
        if (c.data.size == 0) { result = ERROR_CODE::COMMUNICATION_ERROR; break; }  // no progress, not eof
    }
    std::fclose(f);
    // SUCCESS means the local file is the whole recording. total==0 tolerated
    // for firmware that never fills the field.
    if (result == ERROR_CODE::SUCCESS && total && offset != total)
        result = ERROR_CODE::SESSION_RECORDING_ERROR;
    // The partial is kept on failure so a re-run resumes it (see above); a
    // failed download's file is not a valid .mcap until a run returns SUCCESS.
    return result;
}


ERROR_CODE Device::upload_recording(const std::string& name, const std::string& url,
                                    bool resumable) {
    if (!is_open())                 return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    if (name.empty() || url.empty()) return ERROR_CODE::INVALID_FUNCTION_CALL;
    if (!internal::is_http_url(url))          return ERROR_CODE::INVALID_FUNCTION_CALL;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_start_upload_tag;
    std::snprintf(req.body.start_upload.recording_name,
                  sizeof req.body.start_upload.recording_name, "%s", name.c_str());
    req.body.start_upload.has_spec  = true;
    req.body.start_upload.spec.dest = ef_v1_UploadDest_UPLOAD_CLOUD_URL;
    req.body.start_upload.spec.mode = ef_v1_UploadMode_UPLOAD_AFTER_STOP;
    req.body.start_upload.spec.resumable = resumable;
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

}  // namespace ef
