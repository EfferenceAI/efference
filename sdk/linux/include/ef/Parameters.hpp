////////////////////////////////////////////////////////////////////////////////
//
// File:      Parameters.hpp
// Purpose:   Public SDK parameter classes passed into ef::Device.
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

#ifndef EF_PARAMETERS_HPP
#define EF_PARAMETERS_HPP

#include <cstdint>
#include <string>

#include "Core.hpp"
#include "Enums.hpp"

namespace ef {

// Passed to open(): transport + the session configuration. Validated against
// the device's advertised capabilities at open(); an unsupported combination
// fails with INVALID_RESOLUTION / INVALID_FPS / UNSUPPORTED_COMPRESSION.
class InitParameters {
public:
    // ---- transport ----------------------------------------------------------
    INPUT_TYPE  input_type = INPUT_TYPE::USB;
    std::string ble_address;    // required when input_type == STREAM (BLE MAC)
    std::string mcap_path;      // required when input_type == MCAP (replay)
    int device_id = 0;  // which M1 when several are attached over USB

    // STREAM only: BLE carries control while the device pushes video+IMU to
    // this host over WiFi/UDP. udp_host is this machine's IP as the device can
    // reach it (the device cannot infer it over BLE). Leave udp_host empty for
    // a control-only BLE session (device sits in IDLE).
    std::string udp_host;
    uint16_t    udp_port = 5005;
    std::string ble_password = "123456";   // BLE control password (factory default)

    // ---- session configuration (applied at open(); capture auto-starts) -----
    RESOLUTION       resolution  = RESOLUTION::AUTO;
    int              fps         = 30;
    COMPRESSION_MODE compression = COMPRESSION_MODE::H265;
    FLIP_MODE        flip_mode   = FLIP_MODE::OFF;
    bool             enable_imu  = true;

    // Coordinate convention for IMU data and pose.
    COORDINATE_SYSTEM coordinate_system = COORDINATE_SYSTEM::IMAGE;

    // ---- grab behavior -------------------------------------------------------
    uint32_t grab_timeout_ms     = 1000;   // grab() blocks at most this long
    bool     drop_partial_frames = true;   // discard lossy frames silently

    int verbose = 0;                       // 0 = quiet, 1 = connection trace
};

// Passed to grab(): per-call knobs.
class RuntimeParameters {
public:
    uint32_t timeout_ms     = 0;       // 0 = use InitParameters::grab_timeout_ms
    bool     return_partial = false;   // deliver lossy frames as CORRUPTED_FRAME
};

// Passed to enable_recording(): where the session goes.
class RecordingParameters {
public:
    RECORDING_TARGET target = RECORDING_TARGET::HOST_FILE;

    // HOST_FILE: path of the .mcap written on the host.
    std::string path;

    // DEVICE_LOCAL: session name on the device ("" = device picks
    // "rec_<utc>_<seq>"). The recording survives host disconnect.
    std::string name;

    // Optional per-session location override: when set, this recording's
    // LocationFix uses `location` instead of the device's persistent/default
    // location. Does not change the persistent default (use Device::set_location
    // for that).
    bool     has_location = false;
    Location location;
};

}  // namespace ef

#endif  // EF_PARAMETERS_HPP
