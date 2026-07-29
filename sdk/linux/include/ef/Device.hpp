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
    // Whether this session proved the control password. Meaningful only when the
    // device reports itself locked (DeviceInformation::usb_locked, or any BLE
    // link). open() deliberately SUCCEEDS with a wrong password, because the
    // device still answers info/state/storage and those are what an operator
    // needs when recovering a lost password; check this to know whether gated
    // verbs will work before calling one.
    bool           is_authenticated() const;
    void           close();

    DEVICE_STATE get_state() const;

    // Live fault poll. Refreshes the device state from the device firmware's state machine (so a
    // subsequent get_state() is fresh too). Returns true iff a fault is LATCHED (the
    // device is in SAFE and needs a health-gated recovery); a latched device projects
    // DEVICE_STATE::CLOSED (the 4-value enum has no FAULT value), which is how a client
    // tells a fault-CLOSED from a not-open CLOSED. When `reason` is non-null it receives
    // the most recent anomaly cause -- populated for a latched fault AND for an unlatched
    // abnormal session end (e.g. disk_full, capture_stopped), and empty once a new
    // session starts clean -- so check `reason` even when the return is false. Best-effort:
    // on a communication failure it returns the last-known cached values.
    bool poll_fault(std::string* reason = nullptr);

    DeviceInformation get_device_information() const;

    ERROR_CODE grab(RuntimeParameters params = RuntimeParameters());
    RuntimeParameters get_runtime_parameters() const;

    ERROR_CODE retrieve_image(Mat& mat, VIEW view = VIEW::NV12);
    ERROR_CODE retrieve_imu(SensorsData& data,
                            TIME_REFERENCE ref = TIME_REFERENCE::IMAGE);

    Timestamp get_timestamp(TIME_REFERENCE ref = TIME_REFERENCE::CURRENT) const;

    ERROR_CODE health_check(HealthStatus& out, bool deep = false);
    HealthStatus get_health_status() const;

    ERROR_CODE enable_recording(RecordingParameters params);
    // Stops the active recording. Host-file always stops cleanly (SUCCESS);
    // device-local returns the device's result (INVALID_FUNCTION_CALL if idle).
    ERROR_CODE disable_recording();
    RecordingParameters get_recording_parameters() const;

    ERROR_CODE get_recording_status(RecordingStatus& out,
                                    const std::string& name = "");
    ERROR_CODE list_recordings(std::vector<RecordingStatus>& out);

    // Free / total bytes on the recording store (the /userdata ext4 partition,
    // via statvfs). Queryable any time, no active recording required.
    ERROR_CODE get_storage(uint64_t& free_bytes, uint64_t& total_bytes);
    ERROR_CODE delete_recording(const std::string& name);

    // Pull a device recording over the control link (USB/BLE, no WiFi needed).
    // A failed run leaves the partial file at dest_path — not a valid .mcap
    // until a run returns SUCCESS — and a re-run resumes it after verifying it
    // belongs to this recording.
    ERROR_CODE download_recording(const std::string& name,
                                  const std::string& dest_path);

    ERROR_CODE upload_recording(const std::string& name, const std::string& url);
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

    ERROR_CODE wifi_add(const std::string& ssid, const std::string& psk,
                        const std::string& country = "");  // e.g. "US" unlocks 5 GHz
    ERROR_CODE wifi_remove(const std::string& ssid);
    ERROR_CODE wifi_select(const std::string& ssid);

    // Access points in range, strongest first. DEVICE_BUSY while recording or
    // livestreaming (a scan would disrupt the link); retry once idle.
    ERROR_CODE scan_wifi_networks(std::vector<WifiNetwork>& out);

    // Rekey the device control password (shared by BLE and USB). The old password
    // is required, except on an UNLOCKED USB link where it may be "" (physical
    // access resets a forgotten one). Once USB is locked that shortcut is gone
    // and factory_reset() is the escape.
    ERROR_CODE set_ble_password(const std::string& old_password,
                                const std::string& new_password);

    // Lock or unlock the USB control plane. Unlocked (factory default) USB has
    // full privileges; locked, it gates exactly like BLE and open() authenticates
    // with InitParameters::ble_password. Toggling requires the current password.
    //
    // session_only opens a LOCKED device for this power session without changing
    // the stored policy: gated verbs answer with no password until set_usb_lock(
    // true, ...) or the device loses power, and it still reports usb_locked.
    // Refused when the device is not locked. Authenticating never does this
    // implicitly.
    ERROR_CODE set_usb_lock(bool locked, bool session_only = false);

    // Turn at-rest video encryption on/off for SUBSEQUENT recordings; existing
    // ones keep whatever they were written with. Refused with INVALID_PARAMETER
    // if no key exists, so "enabled" never means "recording in the clear".
    ERROR_CODE set_encryption(bool enabled);

    // Read the video-encryption key. Gated on the same password as the lock.
    // `present` is false when no key exists, which is not an error.
    ERROR_CODE get_encryption_key(EncryptionKey& out);

    // Generate the device's encryption key and return it. This is the ONLY time
    // the key is handed out in full at creation, so the caller must keep it: it is
    // what decrypts every recording made from here on.
    //
    // Refused with INVALID_PARAMETER when a key already exists, because replacing
    // one would make every recording written under it permanently undecryptable.
    // Rotation is delete_encryption_key() then create, two deliberate steps.
    ERROR_CODE create_encryption_key(EncryptionKey& out);

    // Destroy the device's encryption key. `key_id` must match the installed key
    // (see DeviceInformation::encryption_key_id), which is what stops a caller
    // destroying a key it never identified; INVALID_PARAMETER if it does not.
    //
    // `out` carries the destroyed key, with present == false. That is the last
    // chance to keep it for ciphertext already recorded under it; afterwards
    // nothing on the device can decrypt those recordings, wherever they now live.
    //
    // Refused unless the device is IDLE: a running session holds the key in
    // memory and would keep encrypting under it after the call reported it gone.
    ERROR_CODE delete_encryption_key(const std::string& key_id, EncryptionKey& out);

    // Restore factory settings: password to default, USB unlocked, encryption
    // off, and wifi, calibration, capture config, recordings and runtime state
    // cleared. Ungated on USB only, so physical possession is the escape when the
    // password is lost; over BLE it needs the password like any other verb.
    //
    // ⚠ DESTROYS the encryption key. Every recording ever made under it becomes
    // permanently undecryptable, including copies already uploaded elsewhere.
    // Unlike delete_encryption_key() it does NOT return the key first: this verb
    // answers unauthenticated over USB, and handing a key to an unauthenticated
    // caller is exactly the exposure destroying it removes.
    //
    // Preserves the ADB-enable marker. Refused while a capture is active.
    ERROR_CODE factory_reset();

    ERROR_CODE sync_time();   // align the device clock with the host clock
    // Read the device's current wall clock (epoch). Round-trips TimeSync; the
    // returned Timestamp is the device CLOCK_REALTIME at the time of the reply.
    ERROR_CODE get_device_time(Timestamp& out);

    // Persist the device location to session_meta.json (replaces the default);
    // every subsequent recording uses it until changed. For a one-off, use
    // RecordingParameters::location instead.
    ERROR_CODE set_location(double latitude, double longitude,
                            double altitude = 0.0, double covariance_diag = 0.0);
    // Read the device's current effective location (session_meta.json if present,
    // otherwise the compiled default).
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
// The answer is not trusted: the device still ECDSA-verifies the bundle and refuses
// anything not strictly newer, so a wrong answer wastes a download, nothing more.

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
