////////////////////////////////////////////////////////////////////////////////
//
// File:      Enums.hpp
// Purpose:   [PURPOSE]
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

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace ef {

enum class ERROR_CODE : int {
    // --- Success Code ---
    SUCCESS = 0, // Standard code for successful behavior.

    // --- 1-19: Information & State Management ---
    // This block of errors are meant to be non-critical, providing information to the user
    // about changes in configuration, rebooting, and other events that cause discontinuity 
    // in sensor data / function but do not kill a session.
    SENSOR_CONFIGURATION_CHANGED = 10, // Sensor configuration was changed externally while streaming. SDK automatically reconnects but frames will be dropped during change.
    CONFIGURATION_FALLBACK = 11, // Target configuration setup was unsuccessful but fallback configuration was successful.
    CAMERA_REBOOTING = 12, // The camera is currently rebooting.

    // --- 20-39: Device Setup & Sequencing ---
    // This block of errors are related to setup on the SDK side. Users will incorrectly use our code
    // and these errors are meant to be descriptive and push towards proper code usage. 
    DEVICE_NOT_DETECTED = 20, // No device was detected.
    DEVICE_NOT_INITIALIZED = 21, // The Efference SDK is not initialized. Probably a missing call to Device.open().
    DEVICE_NOT_AVAILABLE = 22,  // Device is detected but cannot be opened.
    INVALID_FIRMWARE = 23, // Corrupted image.
    INVALID_FUNCTION_CALL = 24, // The call of the function is not valid in the current context. Could be a missing call of Device.open().

    // --- 40-59: Sensor Parameters & Calibration ---
    // These errors are related to image sensor and IMU setup, calibration, and configuration.
    INVALID_RESOLUTION = 40, // In case of invalid resolution parameter, such as an upsize beyond the original image size.
    INVALID_FPS = 41, // FPS selected that does not belong to the valid set for each resolutuon.
    UNSUPPORTED_COMPRESSION = 42, // Compression selected outside the set of valid selections.
    CALIBRATION_FILE_NOT_AVAILABLE = 43, // If the calibration file was removed from userspace on the device and we try to call it.
    INVALID_CALIBRATION_FILE = 44, // If the calibration file is nonsense and cannot be loaded.
    POTENTIAL_CALIBRATION_ISSUE = 45, // Real-time detected if we see stange values blowing up.

    // --- 60-79: Hardware & Transport ---
    // Here track things that can go wrong with DRAM, SoC, eMMC, PMIC, USB, and hardware 
    // related issues for image sensor, IMU, WiFi, and BT.
    LOW_USB_BANDWIDTH = 60, // Insufficient bandwidth for the correct use of the device. This issue can occur when you use multiple cameras or a USB 2.0 port.
    CANNOT_START_CAMERA_STREAM = 61, // Cannot start the camera stream. Make sure your camera is not already used by another process or blocked by firewall or antivirus.

    // --- 80-99: Streaming & Recording  ---
    // This block of errors are related to USB-C/WiFi/BT streaming and recording. They are 
    // not neccessarily hardware errors but events that occur sometimes, such as dropped frames.
    CORRUPTED_FRAME = 70, // The image is corrupted with invalid colors (green/purple images). This indicates a serious hardware or driver issue.
    SESSION_RECORDING_ERROR = 71, // If the recording fails for whatever reason. 
    END_OF_BUFFER = 72, // Recording stopped because we reached the end of a recording session with the buffer.

    // --- We Have No Idea What's Going On Code ---
    UNKNOWN_FAILURE = 100 // Standard code for unknown, unsuccessful behavior.
};

enum class MODEL {  
    M1,
};

enum class INPUT_TYPE {  
    USB,
    STREAM, // WiFi or BT
    MCAP, // our standard for recording data
};
enum class RESOLUTION  { 
    HD1200,
    HD1080,
    SVGA,
    AUTO
};

enum class COMPRESSION_MODE { 
    LOSSLESS,
    H264,
    H264_LOSSLESS,
    H265,
    H265_LOSSLESS,
 };

enum class SENSOR_TYPE {
    ACCELEROMETER,
    GYROSCOPE
};

enum class SENSORS_UNIT {
    M_SEC_2,
    DEG_SEC,
    CELSIUS,
    HERTZ
};

enum class LENS_DISTORTION_MODEL {
    DS
};


} // namespace ef

#endif // EF_ENUMS_HPP