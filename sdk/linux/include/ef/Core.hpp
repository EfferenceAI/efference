////////////////////////////////////////////////////////////////////////////////
//
// File:      Core.hpp
// Purpose:   Public SDK data types returned by the device.
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

#ifndef EF_CORE_HPP
#define EF_CORE_HPP

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "Enums.hpp"

namespace ef {

class Timestamp {
public:
    uint64_t data_ns = 0;

    uint64_t nanoseconds()  const { return data_ns; }
    uint64_t microseconds() const { return data_ns / 1000ULL; }
    uint64_t milliseconds() const { return data_ns / 1000000ULL; }
    uint64_t seconds()      const { return data_ns / 1000000000ULL; }
};

struct Resolution {
    int width  = 0;
    int height = 0;
};

// Device geographic location, written into each recording's location metadata.
// covariance_diag fills the diagonal of the 3x3 position covariance (0 = unknown).
struct Location {
    double latitude        = 0;
    double longitude       = 0;
    double altitude        = 0;
    double covariance_diag = 0;
};

struct Matrix4x4 {
    std::array<double, 16> m{};
};


struct CalibrationParameters {
    LENS_DISTORTION_MODEL model = LENS_DISTORTION_MODEL::DS;
    double fx = 0, fy = 0;      // focal length (px)
    double cx = 0, cy = 0;      // principal point (px)
    double xi = 0, alpha = 0;   // double-sphere parameters
    int    width = 0, height = 0;   // resolution the intrinsics were calibrated at
    // Perform rectification ON-DEVICE via FEC hardware (undistort frames using
    // these intrinsics). Default off; the intrinsics are always published as
    // recording metadata regardless, so a host consumer can rectify instead.
    bool   rectify = false;
    // Rectified-output FOV scale (rectify path). 1.0 = match lens focal; <1
    // widens (more periphery + black corners), >1 zooms the center.
    double fov_scale = 1.0;
};

struct CameraConfiguration {
    Resolution resolution;
    int fps = 0;
    COMPRESSION_MODE compression = COMPRESSION_MODE::RAW;
    CalibrationParameters calibration;
};

// One selectable capture mode. Only the capture modes the device has enabled
// are listed here; disabled modes are filtered out.
struct SupportedMode {
    Resolution  resolution;
    int         fps = 0;
    std::string binning;   // "none" or "2x2" (how the mode derives from the full sensor)
};

// The device's enabled capability menu: the resolution/fps modes and codecs
// a client may select.
struct Capabilities {
    std::vector<SupportedMode> modes;
    std::vector<std::string>   codecs;
};

// One inertial sensor (SENSOR_TYPE::ACCELEROMETER or GYROSCOPE).
struct SensorParameters {
    SENSOR_TYPE  type = SENSOR_TYPE::ACCELEROMETER;
    SENSOR_UNIT unit = SENSOR_UNIT::M_SEC_2;
    int sampling_rate = 0;      // Hz
    int range         = 0;      // ±g (accel) / ±deg/s (gyro)
    float noise_density = 0.f;
    float bias_random_walk = 0.f;   // 0 = not estimated (no Allan-variance run)
    double tau = 0.0;               // bias correlation time, s (0 = not estimated)
    SENSOR_STATE state = SENSOR_STATE::NOT_AVAILABLE;
};

struct SensorsConfiguration {
    SensorParameters accelerometer;
    SensorParameters gyroscope;
    Matrix4x4        camera_imu_transform;   // camera frame -> IMU frame
    std::array<double, 3> accel_bias{};      // b, m/s^2 (all-zero = uncalibrated)
    std::array<double, 3> gyro_bias{};       // b, rad/s
    // 3x3 scale-misalignment (M*S), row-major; identity = uncalibrated. Mirrors
    // ImuCalibrationParameters so a set_imu_calibration round-trips through read-back.
    std::array<double, 9> accel_scale_misalign{ {1,0,0, 0,1,0, 0,0,1} };
    std::array<double, 9> gyro_scale_misalign{  {1,0,0, 0,1,0, 0,0,1} };
    long long time_offset_ns = 0;            // IMU->camera temporal offset
};

// Full IMU field calibration written to the device (set_imu_calibration). Model:
// a* = M*S*(x - b), with the 3x3 scale-misalignment folding M and S together,
// row-major, per sensor. Defaults are the identity map (a no-op), so a partially
// filled struct is safe. The params are recorded as MCAP metadata regardless;
// the device only bakes them into the samples when the capture mode is
// IMU_DATA::CALIBRATED. Values come from the host-side IMU calibration solve.
struct ImuCalibrationParameters {
    std::array<double, 3>  accel_bias{};   // b, m/s^2
    std::array<double, 3>  gyro_bias{};    // b, rad/s
    std::array<double, 9>  accel_scale_misalign{ {1,0,0, 0,1,0, 0,0,1} };  // M*S, row-major
    std::array<double, 9>  gyro_scale_misalign { {1,0,0, 0,1,0, 0,0,1} };  // bias-only => identity
    double accel_noise_density    = 0.0;
    double gyro_noise_density     = 0.0;
    double accel_bias_random_walk = 0.0;   // Allan +1/2 slope
    double gyro_bias_random_walk  = 0.0;
    double accel_tau = 0.0;                 // Gauss-Markov correlation time, s
    double gyro_tau  = 0.0;
    std::array<double, 16> imu_to_camera{ {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1} };  // 4x4 identity
    int64_t time_offset_ns = 0;             // IMU->camera temporal offset (0 = HW-sync only)
};

struct WirelessConfiguration {
    std::string wifi_mac_address;
    std::string bt_mac_address;
    bool        bt_paired = false;
    bool        wifi_connected = false;
    std::string wifi_ssid;
    std::string wifi_ip_address;
    int         wifi_rssi = 0;
    bool        internet_reachable = false;
    // Live association detail (best-effort; empty/0 = unknown or older firmware).
    std::string wifi_state;           // "connected" / "connecting" / "disconnected"
    int         wifi_link_speed = 0;  // negotiated PHY link rate, Mbps
    int         wifi_freq_mhz = 0;    // channel frequency, MHz (2.4 vs 5 GHz derivable)
    std::string wifi_security;        // "WPA2" / "WPA3" / "WPA2/WPA3" / "Open"
    std::vector<std::string> saved_networks;
};

// One access point seen by scan_wifi_networks().
struct WifiNetwork {
    std::string ssid;
    int         rssi = 0;        // signal level, dBm (higher = stronger)
    bool        secured = false; // true = encrypted (WPA/WPA2/WPA3)
};

// For device discovery
struct DeviceProperties {
    INPUT_TYPE input_type = INPUT_TYPE::USB;
    int device_id  = 0;
    std::string serial;             // USB descriptor serial
    std::string ble_address;        // BLE MAC (input_type == STREAM)
    std::string ble_name;           // advertised name ("Efference M1")
};

struct DeviceInformation {
    std::string serial;             // full device serial (may be non-numeric)
    unsigned int serial_number = 0; // numeric form; 0 when the serial is not numeric
    MODEL model = MODEL::M1;
    std::string model_name;         // model exactly as the device reports it ("M1")
    // Board revision the device reports. Currently baked per firmware build, so every
    // unit on a given image returns the same value ("1.00"); it identifies a board spin
    // only once provisioned per-unit. Treat "1.00" as "unknown revision".
    std::string hw_rev;
    unsigned int firmware_version = 0;      // OTA monotonic int (anti-rollback)
    std::string  firmware_version_str;      // human-readable "XX.XX.XX" (display only)
    INPUT_TYPE input_type = INPUT_TYPE::USB;
    CameraConfiguration camera_configuration;
    SensorsConfiguration sensors_configuration;
    WirelessConfiguration wireless;
    Capabilities         capabilities;   // enabled capture modes + codecs (device CAPS)
};


// One frame of image data plus the metadata to interpret it. Owns its buffer
// (deep copy/assign). Fields are private to preserve layout invariants; Device
// is the sole producer, consumers read through the accessors.
class Mat {
public:
    Mat() = default;

    // Allocate a CPU buffer for (width x height) of the given layout (step/size
    // derived from type). MEM::GPU returns UNSUPPORTED; a zero/negative extent
    // returns INVALID_FUNCTION_CALL.
    ERROR_CODE alloc(int width, int height, MAT_TYPE type, MEM mem = MEM::CPU);

    // Release the buffer; the Mat becomes uninitialised (isInit() == false).
    void free();

    // Deep copy into dst (full pixel copy). MEM::GPU returns UNSUPPORTED.
    ERROR_CODE copyTo(Mat& dst, MEM mem = MEM::CPU) const;

    // Raw pointer to the pixel bytes. CPU -> the owning buffer; GPU -> nullptr
    // (GPU memory is not supported).
    uint8_t*       getPtr(MEM mem = MEM::CPU);
    const uint8_t* getPtr(MEM mem = MEM::CPU) const;

    // --- metadata (read-only for consumers) ---
    bool       isInit()         const { return !data_.empty(); }
    int        getWidth()       const { return resolution_.width; }
    int        getHeight()      const { return resolution_.height; }
    Resolution getResolution()  const { return resolution_; }
    int        getStep()        const { return step_; }          // bytes per row (plane 0)
    size_t     getSizeInBytes() const { return data_.size(); }
    MAT_TYPE   getDataType()    const { return type_; }
    VIEW       getView()        const { return view_; }
    MEM        getMemoryType()  const { return memory_; }
    uint32_t   getFrameId()     const { return frame_id_; }
    Timestamp  getTimestamp()   const { return timestamp_; }

private:
    friend class Device;   // sole producer; fills the fields below directly
    void flip180();        // in-place 180-degree rotation (host-side, upside-down camera)

    uint32_t   frame_id_ = 0;
    Timestamp  timestamp_;                // capture time, device clock
    Resolution resolution_;
    MAT_TYPE   type_   = MAT_TYPE::NV12;
    VIEW       view_   = VIEW::NV12;
    MEM        memory_ = MEM::CPU;
    int        step_   = 0;               // bytes per row (first plane)
    std::vector<uint8_t> data_;
};

struct ImuSample {
    Timestamp timestamp;                 // device clock
    uint64_t  sequence = 0;
    float     acceleration[3]     = {0, 0, 0};   // m/s^2
    float     angular_velocity[3] = {0, 0, 0};   // rad/s
    float     temperature_c       = 0.f;
};

// All IMU samples since the previous grab(), filled by retrieve_imu().
struct SensorsData {
    std::vector<ImuSample> samples;
    uint64_t     dropped      = 0;
    MOTION_STATE motion_state = MOTION_STATE::STATIC;   // ZUPT / free-fall
};


struct HealthCheck {
    std::string name;
    bool        passed = false;
    std::string detail;
};

struct HealthStatus {
    // Live availability: whether the camera and IMU are currently readable.
    CAMERA_STATE camera = CAMERA_STATE::NOT_AVAILABLE;
    SENSOR_STATE imu    = SENSOR_STATE::NOT_AVAILABLE;

    // Saved: last completed sweep.
    bool                     passed = false;   // no probe failed
    bool                     deep   = false;   // stress tier was included
    std::vector<HealthCheck> checks;
    Timestamp                timestamp;        // when the sweep completed
};

struct RecordingStatus {
    std::string      name;
    RECORDING_TARGET target    = RECORDING_TARGET::HOST_FILE;
    bool             recording = false;
    uint64_t         bytes       = 0;
    uint64_t         frames      = 0;
    uint64_t         duration_ms = 0;

    // Device storage (free/total space for continued recording).
    uint64_t storage_free_bytes  = 0;
    uint64_t storage_total_bytes = 0;

    // Upload of this session (upload_recording()).
    UPLOAD_STATE upload             = UPLOAD_STATE::OFF;
    uint64_t     upload_bytes_sent  = 0;
    uint64_t     upload_bytes_total = 0;
    ERROR_CODE   last_error         = ERROR_CODE::SUCCESS;
};

// What the update-check service says this device should be running. Resolved on the
// HOST: the device holds no update policy and never contacts the service itself.
struct UpdateAvailability {
    bool         available = false;
    unsigned int target_version = 0;     // version_int of the bundle AT `url`, which is
                                         // not necessarily the newest release: the
                                         // service may route an old device through an
                                         // intermediate hop.
    std::string  target_version_str;     // display only ("00.09.13")
    std::string  url;                    // empty unless `available`
    std::string  notes;                  // optional one-line service message
    // Why no answer was usable: the service's own {error, details}, or the transport
    // failure. The ERROR_CODE cannot tell "unreachable" from "it refused our request".
    std::string  service_error;
};

struct UpdateStatus {
    bool         active   = false;
    UPDATE_STATE state    = UPDATE_STATE::DOWNLOADING;
    int          progress = -1;              // 0-100, -1 indeterminate
    std::string  message;
    std::string  running_version;
    uint32_t     running_version_int = 0;
    uint32_t     target_version_int  = 0;
    ERROR_CODE   last_error = ERROR_CODE::SUCCESS;
};

}  // namespace ef

#endif  // EF_CORE_HPP
