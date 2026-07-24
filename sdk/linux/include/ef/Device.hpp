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
    ERROR_CODE download_recording(const std::string& name,
                                  const std::string& dest_path);

    ERROR_CODE upload_recording(const std::string& name, const std::string& url);
    ERROR_CODE stop_upload(const std::string& name);

    ERROR_CODE check_update(bool& available);
    // url: "" = the device's configured update server, an http(s) URL, or a
    // local .eff file path; a local file is sideloaded over the control link.
    ERROR_CODE update(const std::string& url = "",
                      const std::function<void(const UpdateStatus&)>& on_progress = {});
    ERROR_CODE abort_update();
    ERROR_CODE get_update_status(UpdateStatus& out);

    ERROR_CODE wifi_add(const std::string& ssid, const std::string& psk,
                        const std::string& country = "");  // e.g. "US" unlocks 5 GHz
    ERROR_CODE wifi_remove(const std::string& ssid);
    ERROR_CODE wifi_select(const std::string& ssid);

    // Access points in range, strongest first. DEVICE_BUSY while recording or
    // livestreaming (a scan would disrupt the link); retry once idle.
    ERROR_CODE scan_wifi_networks(std::vector<WifiNetwork>& out);

    // Rekey the BLE control password. Over BLE the old password is required;
    // over USB old_password may be "" (physical access resets a forgotten one).
    ERROR_CODE set_ble_password(const std::string& old_password,
                                const std::string& new_password);

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

}  // namespace ef

#endif  // EF_DEVICE_HPP
