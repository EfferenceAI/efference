////////////////////////////////////////////////////////////////////////////////
//
// File:      device.cpp
// Purpose:   ef::Device: construction, discovery, open/close.
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

// ---- construction ---------------------------------------------------------------

Device::Device() : impl_(new Impl) {}
Device::~Device() { close(); }
Device::Device(Device&&) noexcept = default;
// Release the target cleanly first: the defaulted move would destroy Impl
// directly, skipping StopStream/StopRecording and leaving an armed session.
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
    // Report what the device said: the update-check service keys on this string, so
    // asserting M1 would offer M1 firmware to a board that said otherwise.
    di.model_name = d.model[0] ? d.model : "";
    di.model      = MODEL::M1;                  // only value in the enum today
    di.hw_rev     = d.hw_rev[0] ? d.hw_rev : "";
    di.firmware_version     = d.firmware_version_int;
    di.firmware_version_str = d.firmware_version;   // "XX.XX.XX" from the device version file
    di.input_type       = transport;
    // Canonical serial (eMMC CID hex today), USB descriptor as fallback.
    // serial_number is the numeric form, parsed only if all-decimal (hex CID stays 0).
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
        cc.calibration.rectify = d.camera.rectify;
        // Absent/legacy device reports 0; keep the 1.0 default so it's never a no-scale.
        cc.calibration.fov_scale = (d.camera.fov_scale > 0.0) ? d.camera.fov_scale : 1.0;
        cc.calibration.width  = (int)d.camera.width;    // resolution intrinsics were calibrated at
        cc.calibration.height = (int)d.camera.height;
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
        sc.accelerometer.bias_random_walk = (float)d.imu.accel_bias_random_walk;
        sc.gyroscope.bias_random_walk     = (float)d.imu.gyro_bias_random_walk;
        sc.accelerometer.tau = d.imu.accel_tau;
        sc.gyroscope.tau     = d.imu.gyro_tau;
        sc.time_offset_ns    = (long long)d.imu.time_offset_ns;
        if (d.imu.accel_bias_count >= 3)
            for (int i = 0; i < 3; i++) sc.accel_bias[(size_t)i] = d.imu.accel_bias[i];
        if (d.imu.gyro_bias_count >= 3)
            for (int i = 0; i < 3; i++) sc.gyro_bias[(size_t)i] = d.imu.gyro_bias[i];
        if (d.imu.accel_scale_misalign_count >= 9)
            for (int i = 0; i < 9; i++) sc.accel_scale_misalign[(size_t)i] = d.imu.accel_scale_misalign[i];
        if (d.imu.gyro_scale_misalign_count >= 9)
            for (int i = 0; i < 9; i++) sc.gyro_scale_misalign[(size_t)i] = d.imu.gyro_scale_misalign[i];
        if (d.imu.imu_to_camera_count >= 16)
            for (int i = 0; i < 16; i++)
                sc.camera_imu_transform.m[(size_t)i] = d.imu.imu_to_camera[i];
    }

    di.usb_locked         = d.usb_locked;
    di.session_unlocked   = d.session_unlocked;
    di.encryption_enabled = d.encryption_enabled;
    di.encryption_key_present = d.encryption_key_present;
    di.encryption_key_id      = d.encryption_key_id;
    di.encryption_algorithm   = static_cast<ENCRYPTION_ALGORITHM>(d.encryption_algorithm);
    // Absent on older firmware, which decodes to false: exactly right, since such a
    // device does not enforce the tier.
    di.admin_gate                 = d.admin_gate;
    di.admin_provisioned          = d.admin_provisioned;
    di.key_unprotected            = d.key_unprotected;
    di.recordings_volume_attached = d.recordings_volume_attached;
    di.recordings_volume_files    = d.recordings_volume_files;

    di.wireless.wifi_mac_address = d.wifi_mac;   // vendor storage; "" if unprovisioned
    di.wireless.bt_mac_address   = d.bt_mac;
    di.wireless.ble_connected    = d.ble_connected;
    if (d.has_wifi) {
        // The whole WifiStatus: this reply is ungated where GetWifiStatus is not,
        // so on a locked link it is the only source of the last four.
        di.wireless.wifi_connected     = d.wifi.connected;
        di.wireless.wifi_ssid          = d.wifi.ssid;
        di.wireless.wifi_ip_address    = d.wifi.ip;
        di.wireless.wifi_rssi          = d.wifi.rssi;
        di.wireless.internet_reachable = d.wifi.internet;
        di.wireless.wifi_link_speed    = d.wifi.link_speed;
        di.wireless.wifi_freq_mhz      = d.wifi.freq_mhz;
        di.wireless.wifi_security      = d.wifi.security;
        di.wireless.wifi_health        = static_cast<WIFI_HEALTH>(d.wifi.health);
        // Same back-compat derivation refresh_wireless uses: firmware predating
        // WifiStatus.state leaves it empty.
        di.wireless.wifi_state =
            d.wifi.state[0] ? d.wifi.state
                            : (d.wifi.connected ? "connected" : "disconnected");
    }

    if (d.has_caps) {
        // Rebuild, never append: `di` is fresh per call but `caps` is the caller's
        // long-lived menu, and open() alone calls this twice on a locked link.
        if (caps) { caps->modes.clear(); caps->codecs.clear(); }
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
                sm.binning           = d.caps.modes[i].binning;   // "none"/"2x2"
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

    {   // Same reason get_init_parameters() locks: these are std::strings another
        // thread may be copying, and a rotation writes them under this mutex.
        std::lock_guard<std::mutex> lk(impl_->ctl_mtx);
        impl_->init            = params;
        // The latches belong with it: a reopen is a fresh link with a fresh credential
        // set, so nothing the previous one learned may survive (Device::open reuses
        // impl_). Here rather than in each transport arm, so a third arm cannot forget.
        impl_->admin_pw_bad    = false;
        impl_->operator_pw_bad = false;
        impl_->ever_authed     = false;
        // `authenticated` too, or is_authenticated() reports the PREVIOUS link's session
        // on a reopen: an unlocked device never calls control_auth, so nothing else
        // would ever clear it and the stale true would survive for the handle's life.
        impl_->authenticated   = false;
    }
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
        { std::lock_guard<std::mutex> lk(impl_->info_mtx);
          impl_->info            = DeviceInformation{};
          impl_->info.input_type = INPUT_TYPE::MCAP;
          impl_->info.camera_configuration.resolution  = {r->width(), r->height()};
          impl_->info.camera_configuration.compression = impl_->init.compression; }
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
        if (st == Status::DEVICE_NOT_FOUND) {
            // A fresh adapter can miss the advert in the first discovery window;
            // one clean rebuild-and-retry absorbs that without masking a device
            // that is genuinely gone.
            ble.reset(new internal::BleConnection(params.verbose));
            st = ble->open(params.ble_address);
        }
        if (st != Status::SUCCESS) return open_err(st);
        {
            std::lock_guard<std::mutex> lk(impl_->ctl_mtx);
            impl_->connection = std::move(ble);
        }
        // Authenticate before any gated verb, or the device answers
        // AUTH_REQUIRED to everything.
        //
        // A wrong password does not fail the open, matching the USB path: the
        // gated verbs report INVALID_PASSWORD on their own, which is what an
        // operator needs to diagnose a device
        // whose password they have lost. Gated verbs report INVALID_PASSWORD.
        (void)impl_->control_auth(params.ble_password);
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

    // Don't clear a stale stream here: device StopStream maps to StopCollect,
    // which would also kill a DEVICE_LOCAL recording another process started
    // (those survive host disconnect). ensure_streaming() clears it instead.

    // ---- identity snapshot (cached; get_device_information() serves this) ----
    {
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_get_device_information_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_device_information_tag);
        if (ec != ERROR_CODE::SUCCESS) { impl_->close_transport(); return ec; }
        { std::lock_guard<std::mutex> lk(impl_->info_mtx);
          info_from_wire(resp.body.device_information, params.input_type, usb_serial,
                         &impl_->info, &impl_->caps); }

        // A locked USB link gates exactly like BLE. GetDeviceInformation is
        // ungated, so this is the earliest we can know, and authenticating here
        // means every later verb sees an authed session. Unlocked USB (the
        // factory default) skips this entirely.
        //
        // A wrong password does NOT fail the open: an operator diagnosing a device
        // whose password they have lost still needs to identify it, and failing
        // here would hide that behind the very credential they are recovering.
        // Gated verbs report INVALID_PASSWORD on their own.
        if (params.input_type != INPUT_TYPE::STREAM &&
            resp.body.device_information.usb_locked) {
            if (impl_->control_auth(params.ble_password) == ERROR_CODE::SUCCESS) {
                // Re-read now that the session is authed. The snapshot above was
                // necessarily taken pre-auth (it is what tells us the link is
                // locked), so it is missing every gated field -- encryption_key_id
                // today. Without this, a locked device never reports its key_id
                // however correct the operator's password is.
                WireRequest req2 = ef_v1_Request_init_zero;
                req2.which_body = ef_v1_Request_get_device_information_tag;
                WireResponse resp2;
                if (impl_->call(req2, resp2, ef_v1_Response_device_information_tag)
                        == ERROR_CODE::SUCCESS) {
                    std::lock_guard<std::mutex> lk(impl_->info_mtx);
                    info_from_wire(resp2.body.device_information, params.input_type,
                                   usb_serial, &impl_->info, &impl_->caps);
                }
            }
        }
    }

    // ---- validate the session configuration against the advertised menu ------
    // Validate off a snapshot: holding info_mtx through the loops would pin it
    // across the close_transport() calls below, which take it themselves.
    Caps menu;
    { std::lock_guard<std::mutex> lk(impl_->info_mtx); menu = impl_->caps; }
    Resolution res = get_resolution(params.resolution);
    if (params.resolution == RESOLUTION::AUTO) {
        res = {1920, 1200};   // device native (HD1200) when caps offer no match
        for (const auto& m : menu.modes)
            if (m.usable && m.fps == params.fps) { res = {m.w, m.h}; break; }
    } else if (!menu.modes.empty()) {
        bool res_ok = false, fps_ok = false;
        for (const auto& m : menu.modes) {
            if (m.usable && m.w == res.width && m.h == res.height) {
                res_ok = true;
                if (m.fps == params.fps) fps_ok = true;
            }
        }
        if (!res_ok) { impl_->close_transport(); return ERROR_CODE::INVALID_RESOLUTION; }
        if (!fps_ok) { impl_->close_transport(); return ERROR_CODE::INVALID_FPS; }
    }
    if (!menu.codecs.empty()) {
        const char* want = (pb_codec(params.compression) == ef_v1_Codec_H264) ? "H264"
                         : (pb_codec(params.compression) == ef_v1_Codec_H265) ? "H265"
                                                                              : "RAW";
        bool ok = false;
        for (const auto& c : menu.codecs)
            if (c == want) { ok = true; break; }
        if (!ok) { impl_->close_transport(); return ERROR_CODE::UNSUPPORTED_COMPRESSION; }
    }
    // Raw NV12 (~830 Mbit/s @1200p30) overruns the WiFi/UDP online link, so reject
    // it at open rather than on the first grab. Raw over USB isoc (SuperSpeed) is
    // fine, so this is gated on a udp_host, not on the codec alone. The device
    // enforces the same rule; this fails fast with a clear code before any stream.
    if (!params.udp_host.empty() && params.compression == COMPRESSION_MODE::RAW) {
        impl_->close_transport();
        return ERROR_CODE::INSUFFICIENT_WIFI_BANDWIDTH;
    }

    // ---- align the device clock with the host (best-effort, non-fatal) --------
    impl_->set_time_quiet();

    // ---- seed the cached wireless view (saved networks come from WifiList; the
    // identity snapshot only carries the live association) ----------------------
    impl_->refresh_wireless();

    // Data plane starts on first grab() (ensure_streaming), so open() leaves the
    // device IDLE, never blocking control-only work behind a COLLECT session.
    // BLE without a udp_host has no data plane at all.
    impl_->stream_res = res;
    impl_->state      = DEVICE_STATE::IDLE;
    impl_->refresh_device_state();
    return ERROR_CODE::SUCCESS;
}

void Device::close() {
    if (!impl_) return;
    if (impl_->state == DEVICE_STATE::CLOSED) {
        // A failed update() reconnect can strand a host recorder with the transport
        // gone; release it so a later open() doesn't append to the old file.
        impl_->stop_host_rec();
        return;
    }
    // A DEVICE_LOCAL recording survives host disconnect by design: closing the
    // handle must NOT stop it (only disable_recording() does). Tear down host-side
    // resources (recorder + reader) only.
    impl_->stop_host_rec();
    impl_->stop_streaming();
    impl_->close_transport();
    {
        std::lock_guard<std::mutex> lk(impl_->info_mtx);
        impl_->info   = DeviceInformation{};
        impl_->health = HealthStatus{};
        impl_->caps   = Caps{};
    }
    { std::lock_guard<std::mutex> lk(impl_->data_mtx);
      impl_->imu_backlog.clear();
      impl_->imu_backlog_dropped = 0; }
    impl_->flip_latched        = -1;
}

bool         Device::is_open() const { return impl_ && impl_->state != DEVICE_STATE::CLOSED; }
bool         Device::is_authenticated() const { return impl_ && impl_->authenticated; }
DEVICE_STATE Device::get_state() const {
    return impl_ ? impl_->state.load() : DEVICE_STATE::CLOSED;
}

ERROR_CODE Device::refresh_state() {
    // An MCAP replay has no device to ask, and its state is a property of the file
    // rather than something to overwrite from a wire that is not there.
    //
    // Deliberately NOT gated on is_open(), unlike its siblings. is_open() is
    // `cached state != CLOSED`, and CLOSED is what the device's INIT, RESET,
    // HEALTH_TEST and SLEEP all map to — so gating on it makes this call refuse
    // exactly the states it exists to poll a device out of, and a health check
    // would leave get_state() pinned at CLOSED over a healthy link. The real
    // precondition is having a transport, which refresh_device_state checks.
    if (!impl_ || impl_->init.input_type == INPUT_TYPE::MCAP)
        return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    return impl_->refresh_device_state();
}
bool Device::poll_fault(std::string* reason) {
    if (!impl_) { if (reason) reason->clear(); return false; }
    impl_->refresh_device_state();  // live: also refreshes the cached DEVICE_STATE
    // reason carries the most recent anomaly cause whether or not it latched, so an
    // unlatched abnormal stop (disk_full, capture_stopped) surfaces too. The bool
    // return (fault_latch) distinguishes a locked-in fault needing recovery from an
    // informational last-anomaly. Empty once a new session starts clean.
    if (reason) *reason = impl_->device_last_fault;
    return impl_->device_fault_latch;
}

// Cached get_* stay safe on a moved-from Device (null impl_): they serve the type's
// defaults, matching the CLOSED state such a handle reports.
//
// get_init_parameters is the one that takes a lock, because a successful password
// rotation rewrites init's credential strings under ctl_mtx; copying them unlocked
// is a data race on a std::string, not merely a stale read.
InitParameters      Device::get_init_parameters() const      {
    if (!impl_) return InitParameters{};
    std::lock_guard<std::mutex> lk(impl_->ctl_mtx);
    return impl_->init;
}
RuntimeParameters   Device::get_runtime_parameters() const   { return impl_ ? impl_->runtime : RuntimeParameters{}; }
RecordingParameters Device::get_recording_parameters() const { return impl_ ? impl_->recording : RecordingParameters{}; }
DeviceInformation   Device::get_device_information() const {
    if (!impl_) return DeviceInformation{};
    std::lock_guard<std::mutex> lk(impl_->info_mtx);
    return impl_->info;
}

ERROR_CODE Device::refresh_device_information() {
    // Guard like every sibling verb rather than letting call() bounce it. MCAP
    // needs its own test: a replay handle reports STREAMING and so passes
    // is_open(), but it has no device to ask and must not have its view rewritten.
    if (!is_open() || impl_->init.input_type == INPUT_TYPE::MCAP)
        return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_get_device_information_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, ef_v1_Response_device_information_tag);
    if (ec != ERROR_CODE::SUCCESS) {
        // Only on silence (see link_failed): the return code alone is not enough,
        // since a caller that ignores it re-reads the stale answer.
        if (Impl::link_failed(ec)) {
            std::lock_guard<std::mutex> lk(impl_->info_mtx);
            impl_->forget_wifi_association();
            impl_->info.wireless.ble_connected = false;
            impl_->info.recordings_volume_attached = false;
            impl_->info.recordings_volume_files    = 0;
        }
        return ec;
    }
    // By value, not a reference into the struct being overwritten: defensive
    // against info_from_wire's read-before-assign order changing.
    std::unique_lock<std::mutex> lk(impl_->info_mtx);
    const std::string serial_fallback = impl_->info.serial;
    // info_from_wire blanks the fields the wifi verbs own. Carry the saved list:
    // refresh_wireless() below deliberately does not restore it on a failed list.
    std::vector<std::string> saved_nets = impl_->info.wireless.saved_networks;
    info_from_wire(resp.body.device_information, impl_->init.input_type,
                   serial_fallback, &impl_->info, &impl_->caps);
    impl_->info.wireless.saved_networks = std::move(saved_nets);
    lk.unlock();               // refresh_wireless() round-trips; see the lock order
    impl_->refresh_wireless();
    return ERROR_CODE::SUCCESS;
}
HealthStatus        Device::get_health_status() const {
    if (!impl_) return HealthStatus{};
    std::lock_guard<std::mutex> lk(impl_->info_mtx);
    return impl_->health;   // owns a vector<HealthCheck>; close() frees it
}

}  // namespace ef
