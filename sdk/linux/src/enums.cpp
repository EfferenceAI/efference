////////////////////////////////////////////////////////////////////////////////
//
// File:      enums.cpp
// Purpose:   ef enum to_string() overloads.
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

// ---- enum -> string ---------------------------------------------------------------------

const char* to_string(ERROR_CODE e) {
    switch (e) {
        case ERROR_CODE::SUCCESS:                        return "SUCCESS";
        case ERROR_CODE::SENSOR_CONFIGURATION_CHANGED:   return "SENSOR_CONFIGURATION_CHANGED";
        case ERROR_CODE::CONFIGURATION_FALLBACK:         return "CONFIGURATION_FALLBACK";
        case ERROR_CODE::DEVICE_REBOOTING:               return "DEVICE_REBOOTING";
        case ERROR_CODE::FAILED_TO_UPDATE:               return "FAILED_TO_UPDATE";
        case ERROR_CODE::DEVICE_UP_TO_DATE:              return "DEVICE_UP_TO_DATE";
        case ERROR_CODE::DEVICE_NOT_DETECTED:            return "DEVICE_NOT_DETECTED";
        case ERROR_CODE::DEVICE_NOT_INITIALIZED:         return "DEVICE_NOT_INITIALIZED";
        case ERROR_CODE::DEVICE_NOT_AVAILABLE:           return "DEVICE_NOT_AVAILABLE";
        case ERROR_CODE::INVALID_FIRMWARE:               return "INVALID_FIRMWARE";
        case ERROR_CODE::INVALID_FUNCTION_CALL:          return "INVALID_FUNCTION_CALL";
        case ERROR_CODE::INVALID_PASSWORD:               return "INVALID_PASSWORD";
        case ERROR_CODE::INSUFFICIENT_PERMISSIONS:       return "INSUFFICIENT_PERMISSIONS";
        case ERROR_CODE::UNSUPPORTED:                    return "UNSUPPORTED";
        case ERROR_CODE::DEVICE_BUSY:                    return "DEVICE_BUSY";
        case ERROR_CODE::INVALID_RESOLUTION:             return "INVALID_RESOLUTION";
        case ERROR_CODE::INVALID_FPS:                    return "INVALID_FPS";
        case ERROR_CODE::UNSUPPORTED_COMPRESSION:        return "UNSUPPORTED_COMPRESSION";
        case ERROR_CODE::CALIBRATION_FILE_NOT_AVAILABLE: return "CALIBRATION_FILE_NOT_AVAILABLE";
        case ERROR_CODE::INVALID_CALIBRATION_FILE:       return "INVALID_CALIBRATION_FILE";
        case ERROR_CODE::POTENTIAL_CALIBRATION_ISSUE:    return "POTENTIAL_CALIBRATION_ISSUE";
        case ERROR_CODE::LOW_USB_BANDWIDTH:              return "LOW_USB_BANDWIDTH";
        case ERROR_CODE::CANNOT_START_CAMERA_STREAM:     return "CANNOT_START_CAMERA_STREAM";
        case ERROR_CODE::COMMUNICATION_ERROR:            return "COMMUNICATION_ERROR";
        case ERROR_CODE::WIFI_NOT_CONNECTED:             return "WIFI_NOT_CONNECTED";
        case ERROR_CODE::INSUFFICIENT_WIFI_BANDWIDTH:    return "INSUFFICIENT_WIFI_BANDWIDTH";
        case ERROR_CODE::CORRUPTED_FRAME:                return "CORRUPTED_FRAME";
        case ERROR_CODE::SESSION_RECORDING_ERROR:        return "SESSION_RECORDING_ERROR";
        case ERROR_CODE::END_OF_BUFFER:                  return "END_OF_BUFFER";
        case ERROR_CODE::GRAB_TIMEOUT:                   return "GRAB_TIMEOUT";
        case ERROR_CODE::STORAGE_FULL:                   return "STORAGE_FULL";
        case ERROR_CODE::RECORDING_NOT_FOUND:            return "RECORDING_NOT_FOUND";
        case ERROR_CODE::RECORDING_ALREADY_EXISTS:       return "RECORDING_ALREADY_EXISTS";
        case ERROR_CODE::UNKNOWN_FAILURE:                return "UNKNOWN_FAILURE";
    }
    return "UNKNOWN_FAILURE";
}

const char* to_string(MODEL m) {
    switch (m) { case MODEL::M1: return "M1"; }
    return "M1";
}

const char* to_string(INPUT_TYPE t) {
    switch (t) {
        case INPUT_TYPE::USB:    return "USB";
        case INPUT_TYPE::STREAM: return "STREAM";
        case INPUT_TYPE::MCAP:   return "MCAP";
    }
    return "USB";
}

const char* to_string(RESOLUTION r) {
    switch (r) {
        case RESOLUTION::HD1200: return "HD1200";
        case RESOLUTION::HD1080: return "HD1080";
        case RESOLUTION::SVGA:   return "SVGA";
        case RESOLUTION::AUTO:   return "AUTO";
    }
    return "AUTO";
}

const char* to_string(COMPRESSION_MODE c) {
    switch (c) {
        case COMPRESSION_MODE::RAW:      return "RAW";
        case COMPRESSION_MODE::H264:     return "H264";
        case COMPRESSION_MODE::H264_HQ:  return "H264_HQ";
        case COMPRESSION_MODE::H265:     return "H265";
        case COMPRESSION_MODE::H265_HQ:  return "H265_HQ";
    }
    return "RAW";
}

const char* to_string(SENSOR_TYPE t) {
    return t == SENSOR_TYPE::GYROSCOPE ? "GYROSCOPE" : "ACCELEROMETER";
}

const char* to_string(SENSOR_UNIT u) {
    switch (u) {
        case SENSOR_UNIT::M_SEC_2: return "M_SEC_2";
        case SENSOR_UNIT::DEG_SEC: return "DEG_SEC";
        case SENSOR_UNIT::CELSIUS: return "CELSIUS";
        case SENSOR_UNIT::HERTZ:   return "HERTZ";
    }
    return "M_SEC_2";
}

const char* to_string(LENS_DISTORTION_MODEL m) {
    switch (m) { case LENS_DISTORTION_MODEL::DS: return "DS"; }
    return "DS";
}

const char* to_string(FLIP_MODE f) {
    switch (f) {
        case FLIP_MODE::ON:   return "ON";
        case FLIP_MODE::OFF:  return "OFF";
        case FLIP_MODE::AUTO: return "AUTO";
    }
    return "OFF";
}

const char* to_string(VIEW v) {
    switch (v) {
        case VIEW::GRAY: return "GRAY";
        case VIEW::BGR:  return "BGR";
        case VIEW::RGB:  return "RGB";
        case VIEW::BGRA: return "BGRA";
        case VIEW::RGBA: return "RGBA";
        case VIEW::NV12: return "NV12";
    }
    return "BGR";
}

const char* to_string(DEVICE_STATE s) {
    switch (s) {
        case DEVICE_STATE::CLOSED:       return "CLOSED";
        case DEVICE_STATE::IDLE:         return "IDLE";
        case DEVICE_STATE::STREAMING:    return "STREAMING";
        case DEVICE_STATE::UPDATING:     return "UPDATING";
    }
    return "CLOSED";
}

const char* to_string(SENSOR_STATE s) {
    return s == SENSOR_STATE::AVAILABLE ? "AVAILABLE" : "NOT_AVAILABLE";
}

const char* to_string(CAMERA_STATE s) {
    return s == CAMERA_STATE::AVAILABLE ? "AVAILABLE" : "NOT_AVAILABLE";
}

const char* to_string(TIME_REFERENCE r) {
    return r == TIME_REFERENCE::IMAGE ? "IMAGE" : "CURRENT";
}

const char* to_string(COORDINATE_SYSTEM c) {
    switch (c) {
        case COORDINATE_SYSTEM::IMAGE:                   return "IMAGE";
        case COORDINATE_SYSTEM::LEFT_HANDED_Y_UP:        return "LEFT_HANDED_Y_UP";
        case COORDINATE_SYSTEM::RIGHT_HANDED_Y_UP:       return "RIGHT_HANDED_Y_UP";
        case COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP:       return "RIGHT_HANDED_Z_UP";
        case COORDINATE_SYSTEM::LEFT_HANDED_Z_UP:        return "LEFT_HANDED_Z_UP";
        case COORDINATE_SYSTEM::RIGHT_HANDED_Z_UP_X_FWD: return "RIGHT_HANDED_Z_UP_X_FWD";
    }
    return "IMAGE";
}

const char* to_string(MEM m) {
    switch (m) {
        case MEM::CPU:  return "CPU";
        case MEM::GPU:  return "GPU";
        case MEM::BOTH: return "BOTH";
    }
    return "CPU";
}

const char* to_string(MOTION_STATE s) {
    switch (s) {
        case MOTION_STATE::STATIC:  return "STATIC";
        case MOTION_STATE::MOVING:  return "MOVING";
        case MOTION_STATE::FALLING: return "FALLING";
    }
    return "STATIC";
}

const char* to_string(RECORDING_TARGET t) {
    return t == RECORDING_TARGET::DEVICE_LOCAL ? "DEVICE_LOCAL" : "HOST_FILE";
}

const char* to_string(UPLOAD_STATE s) {
    return s == UPLOAD_STATE::RUNNING ? "RUNNING" : "OFF";
}

const char* to_string(STOP_REASON r) {
    switch (r) {
        case STOP_REASON::USER:        return "USER";
        case STOP_REASON::DISK_FULL:   return "DISK_FULL";
        case STOP_REASON::WRITE_ERROR: return "WRITE_ERROR";
        case STOP_REASON::INTERRUPTED: return "INTERRUPTED";
        default:                       return "UNSPECIFIED";
    }
}

const char* to_string(UPDATE_STATE s) {
    switch (s) {
        case UPDATE_STATE::DOWNLOADING:    return "DOWNLOADING";
        case UPDATE_STATE::VERIFYING:      return "VERIFYING";
        case UPDATE_STATE::READY_TO_APPLY: return "READY_TO_APPLY";
        case UPDATE_STATE::APPLYING:       return "APPLYING";
        case UPDATE_STATE::CHECKING:       return "CHECKING";
        case UPDATE_STATE::UPLOADING:      return "UPLOADING";
        case UPDATE_STATE::REBOOTING:      return "REBOOTING";
        case UPDATE_STATE::RECONNECTING:   return "RECONNECTING";
    }
    return "DOWNLOADING";
}

const char* to_string(MAT_TYPE t) {
    switch (t) {
        case MAT_TYPE::U8_C1: return "U8_C1";
        case MAT_TYPE::U8_C3: return "U8_C3";
        case MAT_TYPE::U8_C4: return "U8_C4";
        case MAT_TYPE::NV12:  return "NV12";
    }
    return "NV12";
}

const char* to_string(IMU_DATA d) {
    switch (d) {
        case IMU_DATA::RAW:        return "raw";
        case IMU_DATA::CALIBRATED: return "calibrated";
        case IMU_DATA::BOTH:       return "both";
    }
    return "raw";
}

}  // namespace ef
