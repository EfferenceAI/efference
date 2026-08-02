////////////////////////////////////////////////////////////////////////////////
//
// File:      device_ota.cpp
// Purpose:   ef::Device firmware-update (OTA) verbs.
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
            u.active = true; u.state = UPDATE_STATE::CHECKING; break;
        case ef_v1_OtaPhase_OTA_DOWNLOADING:
            u.active = true; u.state = UPDATE_STATE::DOWNLOADING; break;
        case ef_v1_OtaPhase_OTA_VERIFYING:
            u.active = true; u.state = UPDATE_STATE::VERIFYING; break;
        case ef_v1_OtaPhase_OTA_READY:
            u.active = true; u.state = UPDATE_STATE::READY_TO_APPLY; break;
        case ef_v1_OtaPhase_OTA_APPLYING:
            u.active = true; u.state = UPDATE_STATE::APPLYING; break;
        case ef_v1_OtaPhase_OTA_REBOOTING:
            u.active = true; u.state = UPDATE_STATE::REBOOTING; break;
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

// One POST, using the info cached at open(); the device is not consulted.
ERROR_CODE Device::check_update(UpdateAvailability& out) {
    out = UpdateAvailability{};
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    // An MCAP replay has no device behind it, so info is default-constructed and there
    // is nothing to ask the service about. Same guard update() applies.
    if (impl_->init.input_type == INPUT_TYPE::MCAP)
        return ERROR_CODE::INVALID_FUNCTION_CALL;
    DeviceInformation snap;
    { std::lock_guard<std::mutex> lk(impl_->info_mtx); snap = impl_->info; }
    return check_for_update(snap, out);
}

ERROR_CODE Device::check_update(bool& available) {
    UpdateAvailability a;
    ERROR_CODE ec = check_update(a);
    available = a.available;
    return ec;
}

// Drive the whole A/B update: acquire (URL download or local-file sideload) ->
// verify -> apply -> reboot -> reconnect -> confirm the new version. Blocking.
ERROR_CODE Device::update(const std::string& url,
                          const std::function<void(const UpdateStatus&)>& on_progress) {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;

    // Host-side illegal-state pre-check (decided without touching the device):
    // legal from IDLE, or with only a live host stream (paused below); never while
    // recording or uploading (both fold into STREAMING, so check raw facets), and
    // never on an MCAP replay session (there is no device).
    if ((impl_->state != DEVICE_STATE::IDLE &&
         impl_->state != DEVICE_STATE::STREAMING) ||
        impl_->device_rec || impl_->dev_recording || impl_->dev_uploading ||
        impl_->init.input_type == INPUT_TYPE::MCAP)
        return ERROR_CODE::INVALID_FUNCTION_CALL;

    uint32_t old_version;
    { std::lock_guard<std::mutex> lk(impl_->info_mtx); old_version = impl_->info.firmware_version; }

    // A url that names an existing local file means sideload: push the .eff
    // over the control link instead of asking the device to download it.
    struct stat st{};
    const bool sideload = !url.empty() && ::stat(url.c_str(), &st) == 0 &&
                          S_ISREG(st.st_mode);

    // Sideload is USB-only: ~7 KB chunks means ~56k round trips for a 400 MB bundle, with
    // no resume, so over a BLE control link it takes hours and restarts from zero. Both
    // transports push over the bulk control endpoints, so what decides this is the
    // link's speed, not whether it has a stream endpoint.
    if (sideload && impl_->init.input_type == INPUT_TYPE::STREAM)
        return ERROR_CODE::INVALID_FUNCTION_CALL;

    // Resolve first, so an up-to-date device never has its stream paused for nothing.
    std::string fetch_url = url;
    if (!sideload && url.empty()) {
        UpdateAvailability avail;
        DeviceInformation snap2;
        { std::lock_guard<std::mutex> lk(impl_->info_mtx); snap2 = impl_->info; }
        ERROR_CODE ec = check_for_update(snap2, avail);
        // Carry the service's reason across, or the caller sees a bare
        // COMMUNICATION_ERROR here while check_update() explains itself.
        impl_->last_dev_msg = avail.service_error;
        if (ec != ERROR_CODE::SUCCESS) return ec;
        if (!avail.available)          return ERROR_CODE::DEVICE_UP_TO_DATE;
        fetch_url = avail.url;
    }

    // A host-file recording can't span the update (transport close + reboot):
    // finish the .mcap now instead of concatenating two firmware sessions or
    // orphaning the writer if the post-apply reconnect fails.
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
        // A host->device push; report it as UPLOADING, not the device's DOWNLOADING.
        auto push_progress = [&](uint64_t sent, uint64_t total) {
            if (!on_progress) return;
            UpdateStatus u;
            u.active   = true;
            u.state    = UPDATE_STATE::UPLOADING;
            u.progress = total ? (int)((sent * 100) / total) : -1;
            u.message  = "sending update.eff over the wire";
            on_progress(u);
        };
        ERROR_CODE ec = impl_->ota_push(url, push_progress);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
    } else {
        // Attach to an in-flight OTA rather than starting a competing download the device
        // would refuse. Only pre-apply phases can still reach READY; assumes same target.
        UpdateStatus cur;
        const bool in_flight =
            get_update_status(cur) == ERROR_CODE::SUCCESS && cur.active &&
            (cur.state == UPDATE_STATE::CHECKING ||
             cur.state == UPDATE_STATE::DOWNLOADING ||
             cur.state == UPDATE_STATE::VERIFYING ||
             cur.state == UPDATE_STATE::READY_TO_APPLY);
        if (!in_flight) {
            // Only gate on WiFi when actually starting a download. An in-flight one is
            // already past this, and its own retry loop waits out a blip, so checking
            // earlier would refuse the re-run that attaches to it.
            impl_->refresh_wireless();
            bool up; { std::lock_guard<std::mutex> lk(impl_->info_mtx);
                       up = impl_->info.wireless.wifi_connected; }
            if (!up) return bail(ERROR_CODE::WIFI_NOT_CONNECTED);
            // The wire field is a fixed array and snprintf truncates silently, so refuse
            // a URL that would not survive rather than sending a mangled one.
            WireRequest req = ef_v1_Request_init_zero;
            req.which_body = ef_v1_Request_ota_download_tag;
            if (fetch_url.size() >= sizeof req.body.ota_download.base_url)
                return bail(ERROR_CODE::INVALID_FUNCTION_CALL);
            std::snprintf(req.body.ota_download.base_url,
                          sizeof req.body.ota_download.base_url, "%s", fetch_url.c_str());
            WireResponse resp;
            ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
            if (ec != ERROR_CODE::SUCCESS) return bail(ec);
        }
    }

    // 2. Device-side verify, reporting progress until READY. The acquire verbs are
    // async and the OTA status file may hold a stale terminal phase from a prior
    // attempt (see check_update), so a FAILED_TO_UPDATE is believed only once this
    // op is seen active, or after a start grace with no activity.
    auto deadline    = std::chrono::steady_clock::now() + std::chrono::minutes(15);
    auto start_grace = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    bool seen_active = false;
    // Poll BLE gently: the same radio is carrying the WiFi download this poll is
    // waiting on, so polling hard would slow the thing being measured. USB has no
    // such contention and polls once a second.
    const bool ble_link = (impl_->init.input_type == INPUT_TYPE::STREAM);
    const auto poll_interval = std::chrono::seconds(ble_link ? 5 : 1);
    for (;;) {
        UpdateStatus u;
        ERROR_CODE ec = get_update_status(u);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
        // A stale error from a prior attempt (before this op goes active) is ignored
        // below; don't render it either, or the caller shows a spurious FAIL.
        const bool terminal = (u.last_error == ERROR_CODE::FAILED_TO_UPDATE ||
                               u.last_error == ERROR_CODE::DEVICE_UP_TO_DATE);
        const bool ignoring = terminal && !seen_active &&
                              std::chrono::steady_clock::now() < start_grace;
        if (on_progress && !ignoring) on_progress(u);
        if (u.active) seen_active = true;
        if (u.active && u.state == UPDATE_STATE::READY_TO_APPLY) break;
        if ((u.last_error == ERROR_CODE::FAILED_TO_UPDATE ||
             u.last_error == ERROR_CODE::DEVICE_UP_TO_DATE) &&
            (seen_active || std::chrono::steady_clock::now() >= start_grace))
            return bail(u.last_error);
        if (std::chrono::steady_clock::now() >= deadline) return bail(ERROR_CODE::FAILED_TO_UPDATE);
        std::this_thread::sleep_for(poll_interval);
    }

    // Report a phase for the host-driven window (apply/reboot/reconnect), where
    // there is no device status to poll.
    auto report = [&](UPDATE_STATE st, int progress, const std::string& msg) {
        if (!on_progress) return;
        UpdateStatus u;
        u.active = true; u.state = st; u.progress = progress; u.message = msg;
        on_progress(u);
    };

    // 3. Apply, the device writes the boot slot and reboots underneath the host.
    {
        report(UPDATE_STATE::APPLYING, -1, "writing boot slot");
        WireRequest req = ef_v1_Request_init_zero;
        req.which_body = ef_v1_Request_ota_apply_tag;
        WireResponse resp;
        ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
        if (ec != ERROR_CODE::SUCCESS) return bail(ec);
    }

    // 4. Reconnect after the reboot, then confirm what's running. Report REBOOTING/
    // RECONNECTING so the caller sees the window advance, not a frozen bar.
    report(UPDATE_STATE::REBOOTING, -1, "device restarting");
    const InitParameters saved = impl_->init;
    impl_->close_transport();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    ERROR_CODE reopened = ERROR_CODE::DEVICE_NOT_DETECTED;
    const auto reconnect_start = std::chrono::steady_clock::now();
    const auto reconnect_deadline = reconnect_start + std::chrono::minutes(3);
    while (std::chrono::steady_clock::now() < reconnect_deadline) {
        // Keep the message static; the caller's progress callback owns elapsed time.
        report(UPDATE_STATE::RECONNECTING, -1, "waiting for device");
        reopened = open(saved);
        if (reopened == ERROR_CODE::SUCCESS) break;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
    // Distinguish the three outcomes so the caller can show why it failed.
    if (reopened != ERROR_CODE::SUCCESS) {
        report(UPDATE_STATE::RECONNECTING, -1,
               "device did not come back after reboot (possible crash)");
        return ERROR_CODE::FAILED_TO_UPDATE;
    }
    // For the rollback check only; don't render it (a stale post-boot "rebooting"
    // phase would draw a duplicate line).
    UpdateStatus u;
    get_update_status(u);
    if (u.last_error == ERROR_CODE::FAILED_TO_UPDATE ||
        [&]{ std::lock_guard<std::mutex> lk(impl_->info_mtx);
             return impl_->info.firmware_version; }() <= old_version) {
        report(UPDATE_STATE::RECONNECTING, -1,
               "device rebooted but is still on the old version (update rejected / rolled back)");
        return ERROR_CODE::FAILED_TO_UPDATE;
    }
    return ERROR_CODE::SUCCESS;
}

const std::string& Device::last_error_message() const { return impl_->last_dev_msg; }

ERROR_CODE Device::abort_update() {
    if (!is_open()) return ERROR_CODE::DEVICE_NOT_INITIALIZED;
    WireRequest req = ef_v1_Request_init_zero;
    req.which_body = ef_v1_Request_ota_abort_tag;
    WireResponse resp;
    ERROR_CODE ec = impl_->call(req, resp, 0, Ctx::UPDATE);
    impl_->refresh_device_state();
    return ec;
}

}  // namespace ef
