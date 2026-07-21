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
    // Poll the check (manifest fetch + signature verify) to completion. It runs
    // DETACHED and acks before the status file is rewritten, so early polls can read
    // a PRE-check phase (OTA_UNKNOWN, or a stale terminal phase from a prior attempt).
    // Trust nothing until CHECKING is seen, or a start grace elapses without it (a
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

    const uint32_t old_version = impl_->info.firmware_version;

    // A url that names an existing local file means sideload: push the .eff
    // over the control link instead of asking the device to download it.
    struct stat st{};
    const bool sideload = !url.empty() && ::stat(url.c_str(), &st) == 0 &&
                          S_ISREG(st.st_mode);

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
        // Only the "" = configured-server path is gated by check_update(): OtaCheck
        // consults the device's CONFIGURED server (proto carries no URL), so gating an
        // explicit caller URL on it could skip a reachable LAN mirror. Explicit URL
        // goes straight to ota_download.
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

    // 2. Device-side verify, reporting progress until READY. The acquire verbs are
    // async and the OTA status file may hold a stale terminal phase from a prior
    // attempt (see check_update), so a FAILED_TO_UPDATE is believed only once this
    // op is seen active, or after a start grace with no activity.
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

}  // namespace ef
