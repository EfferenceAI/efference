////////////////////////////////////////////////////////////////////////////////
//
// File:      Device.hpp
// Purpose:   [PURPOSE]
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

#ifndef EF_DEVICE_HPP
#define EF_DEVICE_HPP

#include "Enums.hpp"
#include "Parameters.hpp"
#include <functional>
#include <memory>
#include <string>

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

// Invoked exactly once when a stream ends, just before stream_mcap() returns,
// so an event-driven caller is notified of the reason. The SDK has already
// stopped — it does NOT resume; resuming is the caller's explicit decision.
using StreamEndCallback = std::function<void(const StreamResult&)>;

// Sink for raw MCAP bytes — invoked on the reader thread with each ep3 chunk,
// in order. It MUST be fast (append + return); doing heavy work here stalls the
// reader and risks a device-side ring overrun (§8 — keep persist/decode/view on
// their own threads). Used by the sink overload of stream_mcap (e.g. a live
// viewer feeding a parser/decoder while also archiving).
using StreamSink = std::function<void(const uint8_t* data, size_t len)>;

// --- device handle -------------------------------------------

struct InitParameters {
    int verbose   = 0;  // >0 → log transport detail to stderr
    int device_id = 0;  // index among attached M1s (0 = first)
};

class Device {
public:
    Device();
    ~Device();
    Device(const Device&)            = delete;
    Device& operator=(const Device&) = delete;
    Device(Device&&) noexcept;
    Device& operator=(Device&&) noexcept;

    // Discover + claim the device. Returns ALREADY_OPEN if called twice.
    ERROR_CODE open(const InitParameters& params = InitParameters{});

    // READ. Frames get_device_information over USB, parses the reply into the
    // struct. Throws ef::Exception on transport / protocol / parse failure.
    DeviceInformation get_device_information();

    // WRITE. Apply a (partial) session configuration. IDLE-only on the device
    // (ADR-0009) — applied at the next session start. Returns the device's
    // verdict: ConfigureResult::applied + config_version on success, or
    // applied==false + reason on rejection (bad mode, not idle, unknown key).
    // Throws ef::Exception only on transport / protocol failure, never on a
    // plain rejection. Validate intended modes against
    // get_device_information().caps.modes (usable==true) to avoid round-trips.
    ConfigureResult configure(const Configuration& cfg);

    // --- MCAP data plane (M3) -----------------------------------
    // Stream a byte-exact MCAP byte stream from the device to `path`, blocking
    // until `should_stop()` returns true, the duration elapses, or the device
    // signals a terminal end (drop / producer restart / host gone).
    //
    // Mechanics: start_stream over the control channel arms capture (usb_sdk
    // mode); the device pushes the MCAP onto its 2nd bulk IN (ep3). This call
    // drains ep3 to `path` at line rate on the calling thread — it is NOT
    // coupled to any viewer (§8); a live viewer must read `path` independently
    // (e.g. tail it), so a slow viewer can never back-pressure the reader into
    // a device-side overrun. On a clean stop it performs the TWO-PHASE stop
    // (finalize → drain the footer off ep3 → teardown) so `path` is a complete,
    // Foxglove-openable .mcap. On a terminal end `path` is a valid truncated
    // PREFIX (no footer) and the reason is reported — the SDK does NOT resume
    // (§7); `on_end` (if set) and the returned StreamResult carry the reason.
    //
    // Does not throw; failures surface as a STREAM_END in the result.
    StreamResult stream_mcap(const std::string& path,
                             const std::function<bool()>& should_stop,
                             const StreamEndCallback& on_end = {});

    // Convenience: stream for `seconds` (<= 0 means until a terminal end).
    StreamResult stream_mcap(const std::string& path, double seconds,
                             const StreamEndCallback& on_end = {});

    // Same as the file overloads, but delivers the raw MCAP bytes to `sink`
    // (on the reader thread, in order) instead of writing a file — for a live
    // consumer (parser / decoder / viewer) that also wants the two-phase stop +
    // terminal semantics. The caller archives too by writing the bytes itself.
    StreamResult stream_mcap(const StreamSink& sink,
                             const std::function<bool()>& should_stop,
                             const StreamEndCallback& on_end = {});

    // Release the interface + handle. Safe to call when not open.
    void close();

    bool is_open() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// --- enum -> string (for printing / logging) ------------------------

const char* to_string(ERROR_CODE);
const char* to_string(STREAM_END);
const char* to_string(MODEL);
const char* to_string(INPUT_TYPE);
const char* to_string(RESOLUTION);
const char* to_string(COMPRESSION_MODE);
const char* to_string(RATE_CONTROL);
const char* to_string(EXPOSURE_MODE);
const char* to_string(WHITE_BALANCE_MODE);
const char* to_string(GAMMA_MODE);
const char* to_string(NOISE_REDUCTION_MODE);
const char* to_string(DISTORTION_MODE);

}  // namespace ef

#endif  // EF_DEVICE_HPP