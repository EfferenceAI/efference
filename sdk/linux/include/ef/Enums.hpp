////////////////////////////////////////////////////////////////////////////////
//
// File:      Enums.hpp
// Purpose:   Public SDK enumerations.
// Author:    Gianluca Bencomo
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

#ifndef EF_ENUMS_HPP
#define EF_ENUMS_HPP

namespace ef {

enum class ERROR_CODE : int {
    SUCCESS = 0,                        // Operation succeeded.

    // --- 1-19: informational; non-fatal, the session survives ---
    SENSOR_CONFIGURATION_CHANGED = 10,  // Config changed while streaming; the SDK reconnects, dropping frames across the change.
    CONFIGURATION_FALLBACK = 11,        // Target config failed; a fallback config was applied.
    DEVICE_REBOOTING = 12,              // Device is rebooting and cannot be accessed.
    FAILED_TO_UPDATE = 13,              // Firmware update failed.
    DEVICE_UP_TO_DATE = 14,             // Device is already up to date.

    // --- 20-39: setup & call sequencing ---
    DEVICE_NOT_DETECTED = 20,           // No device was detected.
    DEVICE_NOT_INITIALIZED = 21,        // SDK not initialized (missing Device::open()).
    DEVICE_NOT_AVAILABLE = 22,          // Device detected but could not be opened.
    INVALID_FIRMWARE = 23,              // Corrupted or unrecognized firmware image.
    INVALID_FUNCTION_CALL = 24,         // Call not valid in the current DEVICE_STATE.
    INVALID_PASSWORD = 25,              // A credential was rejected: the control password, or the administrator password on a key verb.
    INSUFFICIENT_PERMISSIONS = 26,      // Cannot access the device; add the udev rule / grant USB access.
    UNSUPPORTED = 27,                   // Operation not supported in this build.
    DEVICE_BUSY = 28,                   // Device busy (e.g. recording or livestreaming); retry once it is idle.

    // --- 40-59: sensor parameters & calibration ---
    INVALID_RESOLUTION = 40,            // Resolution not in the device's supported set.
    INVALID_FPS = 41,                   // FPS not valid for the selected resolution.
    UNSUPPORTED_COMPRESSION = 42,       // Codec not in the supported set.
    CALIBRATION_FILE_NOT_AVAILABLE = 43,// Calibration file missing on the device.
    INVALID_CALIBRATION_FILE = 44,      // Calibration file present but unreadable.
    POTENTIAL_CALIBRATION_ISSUE = 45,   // Runtime values suggest a calibration problem.

    // --- 60-79: hardware & transport ---
    LOW_USB_BANDWIDTH = 60,             // Insufficient USB bandwidth (e.g. a USB 2.0 port).
    CANNOT_START_CAMERA_STREAM = 61,    // Camera stream failed to start (may already be in use).
    COMMUNICATION_ERROR = 62,           // A control-plane round trip failed (USB or BLE I/O).
    WIFI_NOT_CONNECTED = 63,            // Operation requires WiFi but the device is not connected (e.g. upload before provisioning).
    INSUFFICIENT_WIFI_BANDWIDTH = 64,   // Selected format is too large for the WiFi/UDP online link (e.g. raw NV12 ~830 Mbit/s); use an encoded codec, or stream raw over USB.

    // --- 80-99: streaming & recording ---
    CORRUPTED_FRAME = 80,               // Frame decoded with invalid colors; a hardware/driver fault.
    SESSION_RECORDING_ERROR = 81,       // Recording failed.
    END_OF_BUFFER = 82,                 // Reached the end of a recorded session.
    GRAB_TIMEOUT = 83,                  // No frame within grab_timeout_ms. Non-fatal: keep looping.
    STORAGE_FULL = 84,                  // Device storage below the free-space floor; recording refused.
    RECORDING_NOT_FOUND = 85,           // No recording with that name exists on the device.
    RECORDING_ALREADY_EXISTS = 86,      // A recording with that name already exists; delete it or pick a new name.
    USB_REQUIRED = 87,                  // Needs the USB link itself; no password satisfies it over BLE.
    DESTINATION_NOT_WRITABLE = 88,      // The local destination could not be written; the recording on the device is fine.

    UNKNOWN_FAILURE = 100               // Unclassified failure.
};

// Explicit values: consumers persist/serialize these (recordings, config, IPC),
// so the numbering is ABI. Append new members; never renumber or reorder.

enum class MODEL {
    M1 = 0,
};

enum class INPUT_TYPE {
    USB    = 0,  // USB input mode
    STREAM = 1,  // WiFi or BT
    MCAP   = 2,  // MCAP file replay (offline)
};

enum class RESOLUTION {
    HD1200 = 0,  // 1920x1200
    HD1080 = 1,  // 1920x1080
    SVGA   = 2,  // 960x600 (2x2 binning)
    AUTO   = 3,  // device native (HD1200)
};

enum class COMPRESSION_MODE {
    RAW     = 0,  // uncompressed
    H264    = 1,
    H264_HQ = 2,  // high quality (near-lossless)
    H265    = 3,
    H265_HQ = 4,  // high quality (near-lossless)
};

// On-device IMU sample handling for a capture session.
// RAW: record uncalibrated samples + carry the full ImuCalibration params as
// metadata (data-collection default; consumer applies M*S*(x-b)). CALIBRATED:
// the device applies M*S*(x-b) per sample. BOTH: emit both.
enum class IMU_DATA {
    RAW        = 0,
    CALIBRATED = 1,
    BOTH       = 2,
};

enum class SENSOR_TYPE {
    ACCELEROMETER = 0,
    GYROSCOPE     = 1,
};

enum class SENSOR_UNIT {
    M_SEC_2 = 0,
    DEG_SEC = 1,
    CELSIUS = 2,
    HERTZ   = 3,
};

enum class LENS_DISTORTION_MODEL {
    DS = 0,  // Double sphere camera model
};

enum class FLIP_MODE {
    ON   = 0,  // flip image
    OFF  = 1,  // don't flip
    AUTO = 2,  // look at IMU accel and flip based on accelerometer direction
};

enum class VIEW {
    GRAY = 0,  // U8_C1
    BGR  = 1,  // U8_C3
    RGB  = 2,  // U8_C3
    BGRA = 3,  // U8_C4
    RGBA = 4,  // U8_C4
    NV12 = 5,  // NV12
};

// Device lifecycle state. No UNKNOWN: an unreadable state surfaces as an
// ERROR_CODE on the call that hit it; get_state() keeps the last known value.
enum class DEVICE_STATE {
    CLOSED    = 0,  // Device is detected but not open.
    IDLE      = 1,  // Device is open but not moving data.
    STREAMING = 2,  // Device is moving data: live host stream, recording, upload, or calibration.
    UPDATING  = 3,  // Device is connected, but unavailable, due to updating.
};

// IMU availability (set by health_check()).
enum class SENSOR_STATE {
    AVAILABLE     = 0,
    NOT_AVAILABLE = 1,
};

// Camera availability (set by health_check()).
enum class CAMERA_STATE {
    AVAILABLE     = 0,
    NOT_AVAILABLE = 1,
};

enum class TIME_REFERENCE {
    IMAGE   = 0,  // time of frame capture (device clock)
    CURRENT = 1,  // time of the call (host clock)
};

// Coordinate convention for IMU and pose data.
enum class COORDINATE_SYSTEM {
    IMAGE                   = 0,
    LEFT_HANDED_Y_UP        = 1,
    RIGHT_HANDED_Y_UP       = 2,
    RIGHT_HANDED_Z_UP       = 3,
    LEFT_HANDED_Z_UP        = 4,
    RIGHT_HANDED_Z_UP_X_FWD = 5,
};

// Destination memory for retrieved data.
enum class MEM {
    CPU  = 0,
    GPU  = 1,
    BOTH = 2,
};

// Motion classification from the IMU (ZUPT / free-fall detection).
enum class MOTION_STATE {
    STATIC  = 0,
    MOVING  = 1,
    FALLING = 2,
};

// Where enable_recording() writes: an MCAP on the host, or the device's eMMC
// (a device-local recording keeps running if the host disconnects).
enum class RECORDING_TARGET {
    HOST_FILE    = 0,
    DEVICE_LOCAL = 1,
};

// Upload state; failures surface as an ERROR_CODE.
enum class UPLOAD_STATE {
    RUNNING = 0,
    OFF     = 1,
};

// Why a device-local recording session ended. UNSPECIFIED for in-progress
// sessions and recordings made by firmware that predates the field.
enum class STOP_REASON {
    UNSPECIFIED = 0,
    USER        = 1,   // explicit stop
    DISK_FULL   = 2,   // storage reserve reached; finalized cleanly
    WRITE_ERROR = 3,   // sink write error ended the session
    INTERRUPTED = 4,   // power loss / crash; recovered at next boot
};

// Firmware update phases. New values are appended; 0-3 keep their values.
enum class UPDATE_STATE {
    DOWNLOADING    = 0,
    VERIFYING      = 1,
    READY_TO_APPLY = 2,
    APPLYING       = 3,
    CHECKING       = 4,   // reading the update manifest
    UPLOADING      = 5,   // host pushing a local .eff over the wire
    REBOOTING      = 6,   // booting into the new slot
    RECONNECTING   = 7,   // waiting for the device after reboot
};

enum class MAT_TYPE {
    U8_C1 = 0,
    U8_C3 = 1,
    U8_C4 = 2,
    NV12  = 3,
};

const char* to_string(ERROR_CODE);
const char* to_string(MODEL);
const char* to_string(INPUT_TYPE);
const char* to_string(RESOLUTION);
const char* to_string(COMPRESSION_MODE);
const char* to_string(IMU_DATA);
const char* to_string(SENSOR_TYPE);
const char* to_string(SENSOR_UNIT);
const char* to_string(LENS_DISTORTION_MODEL);
const char* to_string(FLIP_MODE);
const char* to_string(VIEW);
const char* to_string(DEVICE_STATE);
const char* to_string(SENSOR_STATE);
const char* to_string(CAMERA_STATE);
const char* to_string(TIME_REFERENCE);
const char* to_string(COORDINATE_SYSTEM);
const char* to_string(MEM);
const char* to_string(MOTION_STATE);
const char* to_string(RECORDING_TARGET);
const char* to_string(UPLOAD_STATE);
const char* to_string(STOP_REASON);
const char* to_string(UPDATE_STATE);
const char* to_string(MAT_TYPE);

} // namespace ef

#endif // EF_ENUMS_HPP
