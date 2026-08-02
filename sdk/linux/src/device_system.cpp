////////////////////////////////////////////////////////////////////////////////
//
// File:      device_system.cpp
// Purpose:   ef::Device health, wireless, and system verbs.
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

// ---- health ------------------------------------------------------------------------

namespace {

// Map one wire sweep into the public HealthStatus (live states derived from the
// camera/imu probes). A skipped probe passes (tier not run, not failed) with
// "(skipped)" appended so the detail says why it's hollow.
HealthStatus health_from_wire(const WireHealth& s) {
    HealthStatus h;
    // The sweep verdict is authoritative: PASS unless some check FAILed (SKIP ignored).
    // `health_passed` is the inline gate only StartHealthCheck returns; it is NOT on the
    // GetHealthStatus poll path this comes from, so ANDing it in would force FAIL always.
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

        // Live attributes: whether camera/IMU can currently be read. Derive BEFORE
        // moving c into the vector, std::move(c) empties c.name, so a later read would
        // never match and leave both stuck at NOT_AVAILABLE.
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

    // Host-side illegal-state pre-check (decided without touching the device): no
    // sweep while recording/uploading/updating. A sweep already in flight is fine,
    // the poll below attaches to it (device returns BUSY).
    if (impl_->dev_recording || impl_->dev_uploading ||
        impl_->state == DEVICE_STATE::UPDATING || impl_->device_rec)
        return ERROR_CODE::INVALID_FUNCTION_CALL;

    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body                   = ef_v1_Request_start_health_check_tag;
        req.body.start_health_check.deep = deep;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp);
        // ONLY a wire BUSY means "a sweep is already running, poll it". err_from
        // collapses BUSY and INVALID_STATE into INVALID_FUNCTION_CALL, so check the
        // wire code: an INVALID_STATE rejection (recording/OTA started elsewhere) must
        // surface as the error, not return the previous sweep as a fresh pass.
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
            std::lock_guard<std::mutex> lk(impl_->info_mtx);
            impl_->health = health_from_wire(resp.body.health_status);
            out = impl_->health;
            return ERROR_CODE::SUCCESS;
        }
        if (std::chrono::steady_clock::now() >= deadline)
            return ERROR_CODE::UNKNOWN_FAILURE;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
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
        // Optimistic dedup'd update; refresh_wireless() then replaces the list with
        // device truth (WifiList), so a re-add / psk rotation never duplicates.
        {
            std::lock_guard<std::mutex> lk(impl_->info_mtx);
            auto& v = impl_->info.wireless.saved_networks;
            if (std::find(v.begin(), v.end(), ssid) == v.end()) v.push_back(ssid);
        }
        impl_->refresh_wireless();   // round-trips; must not hold info_mtx
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
    // Ctx::WIFI keeps a device-side BUSY distinct (DEVICE_BUSY, retryable) from
    // "not a saved network" (NOT_FOUND -> INVALID_FUNCTION_CALL); the CLI leans on
    // that split to word the failure.
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::WIFI);
    if (ec == ERROR_CODE::SUCCESS) {
        {
            std::lock_guard<std::mutex> lk(impl_->info_mtx);
            auto& v = impl_->info.wireless.saved_networks;
            for (auto it = v.begin(); it != v.end();) it = (*it == ssid) ? v.erase(it) : it + 1;
        }
        impl_->refresh_wireless();   // round-trips; must not hold info_mtx
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

ERROR_CODE Device::scan_wifi_networks(std::vector<WifiNetwork>& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_wifi_scan_tag;
    WireResponse resp;
    // Ctx::WIFI maps a device-side BUSY (recording/livestream) to a retryable
    // ERROR_CODE::BUSY; leave `out` untouched so the caller can retry in place.
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_wifi_scan_result_tag,
                                Ctx::WIFI);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    const ef_v1_WifiScanResult& r = resp.body.wifi_scan_result;
    out.clear();
    out.reserve(r.networks_count);
    for (pb_size_t i = 0; i < r.networks_count; i++)
        out.push_back({r.networks[i].ssid, r.networks[i].rssi, r.networks[i].secured});
    std::sort(out.begin(), out.end(),
              [](const WifiNetwork& a, const WifiNetwork& b) { return a.rssi > b.rssi; });
    return ERROR_CODE::SUCCESS;
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

ERROR_CODE Device::set_camera_calibration(const CalibrationParameters& c,
                                          int width, int height) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_set_calibration_tag;
    req.body.set_calibration.which_target = ef_v1_SetCalibration_camera_tag;
    ef_v1_CameraIntrinsics& cam = req.body.set_calibration.target.camera;
    cam.fx     = c.fx;
    cam.fy     = c.fy;
    cam.cx     = c.cx;
    cam.cy     = c.cy;
    cam.xi     = c.xi;
    cam.alpha  = c.alpha;
    cam.width  = (uint32_t)width;
    cam.height = (uint32_t)height;
    cam.rectify = c.rectify;   // on-device FEC rectification (default off)
    cam.fov_scale  = c.fov_scale;    // rectified-output FOV scale (default 1.0)
    // extrinsics_quat intentionally left empty (meaning undefined; see proto).
    WireResponse resp;
    // The device rejects a write outside IDLE (INVALID_STATE -> INVALID_FUNCTION_CALL).
    return impl_->call(req, resp);
}

ERROR_CODE Device::set_imu_calibration(const ImuCalibrationParameters& c) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_set_calibration_tag;
    req.body.set_calibration.which_target = ef_v1_SetCalibration_imu_tag;
    ef_v1_ImuCalibration& im = req.body.set_calibration.target.imu;
    im.accel_bias_count = 3;
    im.gyro_bias_count  = 3;
    for (int i = 0; i < 3; i++) { im.accel_bias[i] = c.accel_bias[i]; im.gyro_bias[i] = c.gyro_bias[i]; }
    im.accel_scale_misalign_count = 9;
    im.gyro_scale_misalign_count  = 9;
    for (int i = 0; i < 9; i++) {
        im.accel_scale_misalign[i] = c.accel_scale_misalign[i];
        im.gyro_scale_misalign[i]  = c.gyro_scale_misalign[i];
    }
    im.accel_noise_density    = c.accel_noise_density;
    im.gyro_noise_density     = c.gyro_noise_density;
    im.accel_bias_random_walk = c.accel_bias_random_walk;
    im.gyro_bias_random_walk  = c.gyro_bias_random_walk;
    im.accel_tau              = c.accel_tau;
    im.gyro_tau               = c.gyro_tau;
    im.imu_to_camera_count = 16;
    for (int i = 0; i < 16; i++) im.imu_to_camera[i] = c.imu_to_camera[i];
    im.time_offset_ns = c.time_offset_ns;
    WireResponse resp;
    // IDLE-only on the device; applied on the next capture session.
    return impl_->call(req, resp);
}

ERROR_CODE Device::reset_calibration(bool camera, bool imu) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                  = ef_v1_Request_reset_calibration_tag;
    req.body.reset_calibration.camera = camera;
    req.body.reset_calibration.imu    = imu;
    WireResponse resp;
    return impl_->call(req, resp);
}

ERROR_CODE Device::set_usb_lock(bool locked, bool session_only) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_set_usb_lock_tag;
    req.body.set_usb_lock.locked       = locked;
    req.body.set_usb_lock.session_only = session_only;
    WireResponse resp;
    return impl_->call(req, resp);
}

ERROR_CODE Device::set_encryption(bool enabled) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_set_encryption_tag;
    req.body.set_encryption.enabled = enabled;
    WireResponse resp;
    return impl_->call(req, resp);
}

// All three key verbs answer with the same EncryptionKey body, so they share one
// unpack; only the request differs.
static void unpack_key(const WireResponse& resp, EncryptionKey& out) {
    out.algorithm = static_cast<ENCRYPTION_ALGORITHM>(resp.body.encryption_key.algorithm);
    out.present   = resp.body.encryption_key.present;
    out.key_id    = resp.body.encryption_key.key_id;
    out.key.assign(resp.body.encryption_key.key.bytes,
                   resp.body.encryption_key.key.bytes + resp.body.encryption_key.key.size);
}

ERROR_CODE Device::get_encryption_key(EncryptionKey& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    out = EncryptionKey{};
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_get_encryption_key_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_encryption_key_tag);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    unpack_key(resp, out);
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::create_encryption_key(EncryptionKey& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    out = EncryptionKey{};
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_create_encryption_key_tag;
    // Leave algorithm at UNSPECIFIED: the device picks its default, which is what
    // a caller with no opinion should get.
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_encryption_key_tag);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    unpack_key(resp, out);
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::delete_encryption_key(const std::string& key_id, EncryptionKey& out) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    out = EncryptionKey{};
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_delete_encryption_key_tag;
    std::snprintf(req.body.delete_encryption_key.key_id,
                  sizeof req.body.delete_encryption_key.key_id, "%s", key_id.c_str());
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_encryption_key_tag);
    if (ec != ERROR_CODE::SUCCESS) return ec;
    unpack_key(resp, out);
    return ERROR_CODE::SUCCESS;
}

ERROR_CODE Device::factory_reset() {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_factory_reset_tag;
    WireResponse resp;
    return impl_->call(req, resp);
}

ERROR_CODE Device::reboot() {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_reboot_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp);
    // Only tear down when the device actually took the reboot: an accepted verb
    // (or a link that died mid-request, it may reboot before replying) means the
    // transport is gone. The device refuses reboot (session kept) while a capture
    // is active, so the caller sees that and the session stays usable.
    if (ec == ERROR_CODE::SUCCESS || ec == ERROR_CODE::COMMUNICATION_ERROR)
        close();
    return ec;
}

ERROR_CODE Device::set_configuration(int width, int height, int fps,
                                     COMPRESSION_MODE codec) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                        = ef_v1_Request_configure_tag;
    // mode left UNKNOWN => geometry/codec-only partial update that doesn't touch
    // the capture mode. Device applies it IDLE-only and refuses while
    // streaming/recording, so a live session is never disrupted.
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

ERROR_CODE Device::set_configuration(int width, int height, int fps,
                                     COMPRESSION_MODE codec, IMU_DATA imu_data) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body                        = ef_v1_Request_configure_tag;
    req.body.configure.has_video          = true;
    req.body.configure.video.width        = (uint32_t)width;
    req.body.configure.video.height       = (uint32_t)height;
    req.body.configure.video.fps          = (uint32_t)fps;
    req.body.configure.video.pixel_format = ef_v1_PixelFormat_NV12;
    req.body.configure.video.codec        = pb_codec(codec);
    req.body.configure.video.quality      = quality_for(codec);
    // IMU capture-mode select (proto ImuConfig.data). has_imu makes it opt-in on
    // the device side (the endpoint forwards "imu_calibrate" only when present),
    // so this overload is the only path that changes the persisted IMU mode.
    req.body.configure.has_imu            = true;
    req.body.configure.imu.data =
        imu_data == IMU_DATA::CALIBRATED ? ef_v1_ImuData_IMU_CALIBRATED :
        imu_data == IMU_DATA::BOTH       ? ef_v1_ImuData_IMU_BOTH       :
                                           ef_v1_ImuData_IMU_RAW;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::SESSION);
    return disambiguate_config_error(ec, resp);
}

}  // namespace ef
