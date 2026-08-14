////////////////////////////////////////////////////////////////////////////////
//
// File:      Device.hpp
// Purpose:   Public SDK entry point, the ef::Device handle.
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
//
// Contract:
//   - open() claims the device, validates the config, and syncs the device clock
//     to the host. The data plane starts lazily on the first grab() for a live
//     transport (USB, or STREAM = BLE control + WiFi/UDP data when udp_host is
//     set); MCAP replay reads from the file. Control-only BLE (no udp_host) stays
//     IDLE with no data plane to grab().
//   - Calls that touch the wire return ERROR_CODE; data comes back via `out`
//     params. get_* methods without an ERROR_CODE never block, returning values
//     cached at open() or the last completed call.
//   - Illegal calls for the current DEVICE_STATE return INVALID_FUNCTION_CALL
//     without touching the device.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef EF_DEVICE_HPP
#define EF_DEVICE_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Core.hpp"
#include "Parameters.hpp"

namespace ef {

class Device {
public:
    Device();
    ~Device();
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    static std::vector<DeviceProperties> get_device_list(bool scan_ble = false,
                                                         uint32_t scan_ms = 3000);

    ERROR_CODE     open(InitParameters params = InitParameters());
    InitParameters get_init_parameters() const;
    bool           is_open() const;
    // Whether this session holds a proven credential. Meaningful only when the device
    // reports itself locked; open() succeeds even with a wrong password, since
    // info/state/storage still answer. Check this before a gated verb.
    bool           is_authenticated() const;
    void           close();

    // Last known state. Free and never blocks; open() and every verb that moves the
    // device keep it current, so poll it as often as you like.
    DEVICE_STATE get_state() const;

    // Re-read state from the device, for a caller that has been idle and wants to
    // see a change it did not cause. On failure the cached value is kept, so an
    // error means get_state() is stale, not that the device is gone.
    ERROR_CODE refresh_state();

    // Live fault poll. Refreshes the device state from the device firmware's state machine (so a
    // subsequent get_state() is fresh too). Returns true iff a fault is LATCHED (the
    // device needs a health-gated recovery); a latched device reports
    // DEVICE_STATE::CLOSED, which is how a client tells a fault from a not-open device. When `reason` is non-null it receives
    // the most recent anomaly cause -- populated for a latched fault AND for an unlatched
    // abnormal session end (e.g. disk_full, capture_stopped), and empty once a new
    // session starts clean -- so check `reason` even when the return is false. Best-effort:
    // on a communication failure it returns the last-known cached values.
    bool poll_fault(std::string* reason = nullptr);

    DeviceInformation get_device_information() const;

    // Retake the snapshot get_device_information() serves; it is otherwise taken
    // at open() and by the wifi verbs, so a handle held across a WiFi drop keeps
    // reporting what it saw then. Re-read after calling: earlier returns are copies.
    // Const accessors stay safe from other threads; two mutating calls are unordered.
    ERROR_CODE refresh_device_information();

    // The first grab() starts the data plane; open() alone streams nothing.
    // GRAB_TIMEOUT is backpressure, not a fault: call again. On MCAP replay,
    // END_OF_BUFFER means the file ended.
    ERROR_CODE grab(RuntimeParameters params = RuntimeParameters());
    RuntimeParameters get_runtime_parameters() const;

    // Both read what the last grab() latched. The Mat borrows that frame and is
    // valid only until the next grab().
    ERROR_CODE retrieve_image(Mat& mat, VIEW view = VIEW::NV12);
    ERROR_CODE retrieve_imu(SensorsData& data,
                            TIME_REFERENCE ref = TIME_REFERENCE::IMAGE);

    // Frame accounting for the open stream. Counts reset when streaming starts.
    // INVALID_FUNCTION_CALL before the first grab(), with `out` left untouched.
    ERROR_CODE get_stream_stats(StreamStats& out) const;

    Timestamp get_timestamp(TIME_REFERENCE ref = TIME_REFERENCE::CURRENT) const;

    ERROR_CODE health_check(HealthStatus& out, bool deep = false);
    HealthStatus get_health_status() const;

    ERROR_CODE enable_recording(RecordingParameters params);
    // Stops the active recording. Host-file always stops cleanly (SUCCESS);
    // device-local returns the device's result (INVALID_FUNCTION_CALL if idle).
    ERROR_CODE disable_recording();
    RecordingParameters get_recording_parameters() const;

    // An empty name asks for the session recording now. A named one resolves on
    // the device, so it reaches sessions older than list_recordings can carry.
    ERROR_CODE get_recording_status(RecordingStatus& out,
                                    const std::string& name = "");
    // The 48 most recent sessions, oldest first.
    ERROR_CODE list_recordings(std::vector<RecordingStatus>& out);

    // Free / total bytes on the device's recording store. Queryable any time,
    // no active recording required.
    ERROR_CODE get_storage(uint64_t& free_bytes, uint64_t& total_bytes);
    ERROR_CODE delete_recording(const std::string& name);

    // Pull a device recording over the control link (USB/BLE, no WiFi needed).
    // dest_path names the output file, or an existing directory to write
    // <name>.mcap into. saved_path reports the file this call targets after that
    // resolution; it is filled in even when the call then fails, so it is the
    // path to look at, not proof one was written. A failed run leaves a partial
    // file there, not a valid .mcap until a run returns SUCCESS. A re-run resumes
    // it after verifying it belongs to this recording.
    // on_progress reports the pull once the destination is open: once before any
    // bytes are written, then as they arrive, on the calling thread. A call that
    // fails before that never invokes it, and nothing is reported after a failure,
    // so the return value is the outcome rather than the last progress seen. Keep
    // it cheap, do not throw, and do not call back into this Device.
    ERROR_CODE download_recording(
        const std::string& name,
        const std::string& dest_path,
        std::string* saved_path = nullptr,
        const std::function<void(const DownloadProgress&)>& on_progress = {});

    // Hand the device a URL to upload a recording to, over WiFi. Returns once the
    // URL is attached, not when the transfer finishes; poll get_recording_status(),
    // whose upload, upload_bytes_sent/total and last_error track it.
    // Returns DEVICE_BUSY if an upload of this recording is already running.
    // Set resumable when url is a resumable-session URI, so an interrupted
    // transfer continues instead of restarting. Minting that URI is the caller's job.
    ERROR_CODE upload_recording(const std::string& name, const std::string& url,
                                bool resumable = false);
    // Abort the upload of a recording: a transfer in flight is cut off, and the URL
    // is detached so it is not retried. Succeeds whether or not one was running.
    // With a resumable URI the destination keeps what it already committed, so
    // re-attaching the same URI continues rather than restarting.
    ERROR_CODE stop_upload(const std::string& name);

    // Ask the update-check service what this device should be running. A host-side
    // network call (see check_for_update below); the device is only read, never asked.
    ERROR_CODE check_update(bool& available);
    ERROR_CODE check_update(UpdateAvailability& out);
    // url: "" = ask the update-check service and use what it returns; an http(s) URL
    // = download exactly that, no service call; a local .eff path = push it over the
    // wire, which requires USB. A bundle that is not strictly newer than the running
    // version is rejected on-device by the signed-manifest anti-rollback gate,
    // whichever route it arrived by.
    ERROR_CODE update(const std::string& url = "",
                      const std::function<void(const UpdateStatus&)>& on_progress = {});
    ERROR_CODE abort_update();
    ERROR_CODE get_update_status(UpdateStatus& out);
    // Device's own words for the last failure, when it sent any. The ERROR_CODE is a
    // category; this is the specific reason and is often the only actionable part.
    const std::string& last_error_message() const;

    // country: e.g. "US". Leave it empty and the device reads the regulatory
    // domain out of nearby beacons, which is what decides whether channels 12-13
    // and the 5 GHz band are usable at all.
    // band: AUTO leaves any stored band pin alone; use wifi_select to clear one.
    ERROR_CODE wifi_add(const std::string& ssid, const std::string& psk,
                        const std::string& country = "",
                        BAND band = BAND::AUTO);
    ERROR_CODE wifi_remove(const std::string& ssid);
    // Prefer a saved network, optionally pinning it to one radio of a dual-band
    // AP. A band the AP does not offer is refused without disturbing the link.
    ERROR_CODE wifi_select(const std::string& ssid, BAND band = BAND::AUTO);

    // Access points in range, strongest first. DEVICE_BUSY while recording or
    // livestreaming (a scan would disrupt the link); retry once idle.
    ERROR_CODE scan_wifi_networks(std::vector<WifiNetwork>& out);

    // Rekey the device control password (shared by BLE and USB). The old password
    // is required, except on an unlocked USB link, where it may be "".
    //
    // An ADMINISTRATOR grant also substitutes for the old password. Call
    // authenticate_admin() immediately before this to use it; the alternative for a
    // forgotten password is factory_reset(), which destroys the encryption key.
    ERROR_CODE set_ble_password(const std::string& old_password,
                                const std::string& new_password);

    // Clear every BLE pairing, for a phone that paired before and will no longer
    // connect. USB only, since it drops the bond of the link it would answer on.
    // Clears the DEVICE side only: each phone must also forget the device before
    // it will pair again. Passwords are unchanged.
    ERROR_CODE forget_ble_bonds();

    // Rekey the ADMINISTRATOR password, the separate credential that guards reading
    // and destroying the encryption key.
    //
    // There is no factory default for it. Pass an empty `old_password` to set one on a
    // device that has none; once set, changing it requires the current value, on either
    // transport. A lost administrator password is recoverable only by factory_reset(),
    // which destroys the encryption key anyway.
    //
    // A new password shorter than 8 characters is refused by the DEVICE with
    // INVALID_PARAMETER; the host does not pre-check it, so read last_error_message().

    ERROR_CODE set_admin_password(const std::string& old_password,
                                  const std::string& new_password);

    // REMOVE the administrator credential, returning the device to "no administrator
    // password": the encryption-key verbs and set_encryption drop back to the control
    // password, and any installed key becomes readable by whoever holds it. A
    // DOWNGRADE, so the device demands the current password on top of the admin grant.
    ERROR_CODE clear_admin_password(const std::string& current_password);

    // Prove the administrator credential NOW, instead of waiting for a verb to demand
    // it. Needed only for set_ble_password()'s rescue, which cannot escalate by itself.
    //
    // The grant is SINGLE-USE and the device spends it on the first admin-gated verb.
    // Call the verb you want immediately afterwards, with nothing in between, or you
    // will spend the grant on something else. The intended sequence is exactly:
    //     authenticate_admin(admin_pw);
    //     set_ble_password("", new_worker_pw);
    //
    // INVALID_PASSWORD means refused, and the handle keeps whatever it already held.
    ERROR_CODE authenticate_admin(const std::string& admin_password);

    // Lock or unlock the USB control plane. Unlocked (factory default) USB carries
    // control-password privilege; the administrator verbs still refuse it. Locked, it
    // gates exactly like BLE and open() authenticates with
    // InitParameters::ble_password. Toggling requires the current password.
    //
    // session_only opens a LOCKED device for this power session without changing
    // the stored policy: gated verbs answer with no password until set_usb_lock(
    // true, ...) or the device loses power, and it still reports usb_locked.
    // Refused when the device is not locked. Authenticating never does this
    // implicitly.
    ERROR_CODE set_usb_lock(bool locked, bool session_only = false);

    // Turn at-rest video encryption on/off for SUBSEQUENT recordings; existing
    // ones keep whatever they were written with. Refused with INVALID_FUNCTION_CALL
    // if no key exists, so "enabled" never means "recording in the clear".
    // Requires the ADMINISTRATOR password when the device has one.
    ERROR_CODE set_encryption(bool enabled);

    // Read the video-encryption key. Requires the ADMINISTRATOR password when the device
    // has one, on either transport and independently of the lock; with none set, the
    // control password is enough and DeviceInformation reports the key as unprotected.
    // `present` is false when no key exists, which is not an error.
    ERROR_CODE get_encryption_key(EncryptionKey& out);

    // Generate the device's encryption key and return it. This is the ONLY time
    // the key is handed out in full at creation, so the caller must keep it: it is
    // what decrypts every recording made from here on.
    //
    // Refused with INVALID_FUNCTION_CALL when a key already exists, because replacing
    // one would make every recording written under it permanently undecryptable.
    // Rotation is delete_encryption_key() then create, two deliberate steps.
    // Requires the ADMINISTRATOR password when the device has one.
    ERROR_CODE create_encryption_key(EncryptionKey& out);

    // Install a key the caller already holds, so a fleet can share one archive key.
    // `key` must be exactly 32 bytes. Refused when a key already exists or the device
    // is not idle; rotation is delete_encryption_key() then set, then set_encryption().
    //
    // `out` names the key (key_id, algorithm, present); its `key` is EMPTY, since the
    // caller supplied the bytes.
    // Requires the ADMINISTRATOR password when the device has one.
    ERROR_CODE set_encryption_key(const std::vector<uint8_t>& key, EncryptionKey& out);

    // Destroy the device's encryption key. `key_id` must match the installed key
    // (see DeviceInformation::encryption_key_id); INVALID_FUNCTION_CALL if it does not.
    //
    // `out` carries the destroyed key, with present == false. That is the last
    // chance to keep it for ciphertext already recorded under it; afterwards
    // nothing on the device can decrypt those recordings, wherever they now live.
    //
    // Refused unless the device is IDLE: a running session holds the key in
    // memory and would keep encrypting under it after the call reported it gone.
    // Requires the ADMINISTRATOR password, on the same terms as get_encryption_key().
    ERROR_CODE delete_encryption_key(const std::string& key_id, EncryptionKey& out);

    // Restore factory settings: the control password back to its default, the
    // administrator password REMOVED (it has no default to restore), USB unlocked,
    // encryption off, and wifi, calibration, capture config, recordings and runtime
    // state cleared. Over BLE it needs the password like any other verb.
    //
    // WARNING: DESTROYS the encryption key. Every recording made under it becomes
    // permanently undecryptable, including copies already uploaded elsewhere.
    // It does NOT return the key first; use delete_encryption_key() to keep a copy.
    //
    // Refused while a capture is active.
    ERROR_CODE factory_reset();

    ERROR_CODE sync_time();   // align the device clock with the host clock
    // Read the device's current wall clock (epoch). Round-trips TimeSync; the
    // returned Timestamp is the device CLOCK_REALTIME at the time of the reply.
    ERROR_CODE get_device_time(Timestamp& out);

    // Store the device location; every subsequent recording carries it until
    // changed. IDLE only. For a one-off, use RecordingParameters::location.
    // All four values are replaced, so omitting covariance_diag stores 0, which
    // reports the fix as accuracy-unknown rather than keeping a previous value.
    ERROR_CODE set_location(double latitude, double longitude,
                            double altitude = 0.0, double covariance_diag = 0.0);
    // Drop the stored location; recordings then carry none. IDLE only. Returns
    // UNSUPPORTED on firmware that cannot hold an empty location.
    ERROR_CODE clear_location();
    // Read the stored location. Location::is_set is false when none is stored.
    ERROR_CODE get_location(Location& out);

    ERROR_CODE reboot();

    // Persist the capture config (resolution/fps/codec). IDLE only; refused while
    // streaming/recording, so it never disrupts a live session. Rejected values
    // report INVALID_RESOLUTION/INVALID_FPS/UNSUPPORTED_COMPRESSION; enabled modes
    // come from get_device_information().capabilities.
    ERROR_CODE set_configuration(int width, int height, int fps,
                                 COMPRESSION_MODE codec);
    // Overload that also selects on-device IMU handling for the session (proto
    // ImuConfig.data): RAW records uncalibrated (default), CALIBRATED applies the
    // stored M*S*(x-b) per sample, BOTH emits both. Geometry/codec as above.
    ERROR_CODE set_configuration(int width, int height, int fps,
                                 COMPRESSION_MODE codec, IMU_DATA imu_data);

    // Persist the camera intrinsics (Double Sphere). IDLE only; applied on the
    // next capture session. The values come from the OpenCV calibration tool.
    // Read them back via get_device_information().camera_configuration.calibration.
    ERROR_CODE set_camera_calibration(const CalibrationParameters& calibration,
                                      int width, int height);
    // Persist the IMU field calibration (bias + M*S + noise/temporal). IDLE only;
    // applied on the next capture session. All fields round-trip via
    // get_device_information().sensors_configuration: accel_bias/gyro_bias,
    // accel_scale_misalign/gyro_scale_misalign, per-sensor noise_density/
    // bias_random_walk/tau, camera_imu_transform, and time_offset_ns. Values come
    // from the host-side IMU calibration solve.
    ERROR_CODE set_imu_calibration(const ImuCalibrationParameters& calibration);
    // Restore factory-default calibration for the selected sensor(s). IDLE only.
    // The camera factory default is currently all-zeros (uncalibrated).
    ERROR_CODE reset_calibration(bool camera, bool imu = false);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---- update-check service ------------------------------------------------------------
// Maps {model, running version, board, unit} to the bundle a device should install next.

// $EF_UPDATE_CHECK_URL if set, else the compiled-in default.
std::string update_check_url();

// One POST, no retries. Empty `service_url` means update_check_url(). SUCCESS with
// available == false covers both "current" and "nothing published for this model";
// COMMUNICATION_ERROR means unreachable or unparsable.
ERROR_CODE check_for_update(const DeviceInformation& info, UpdateAvailability& out,
                            const std::string& channel = "stable",
                            const std::string& service_url = "");

}  // namespace ef

#endif  // EF_DEVICE_HPP
