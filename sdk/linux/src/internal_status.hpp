////////////////////////////////////////////////////////////////////////////////
//
// File:      internal_status.hpp
// Purpose:   Internal transport/reader result codes (ef::Status).
// Author:    Calvin Nguyen
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

#ifndef EF_INTERNAL_STATUS_HPP
#define EF_INTERNAL_STATUS_HPP

namespace ef {

// The transport / reader internals return ef::Status, the fine-grained internal
// result set. Status left the public headers when the API became ERROR_CODE-only,
// so it lives here and device.cpp translates it to ERROR_CODE; nothing above that
// layer speaks Status.
enum class Status : int {
    SUCCESS             = 0,
    DEVICE_NOT_FOUND    = 1,
    INTERFACE_NOT_FOUND = 2,
    INSUFFICIENT_PERMISSIONS = 3,
    CLAIM_FAILED        = 4,
    USB_ERROR           = 5,
    ALREADY_OPEN        = 6,
    NOT_OPEN            = 7,
    BLE_ERROR           = 8,
    BLE_NOT_SUPPORTED   = 9,
    NOT_ENABLED         = 10,
    TIMEOUT             = 11,
    CORRUPTED_FRAME     = 12,
    END_OF_STREAM       = 13,
    DECODE_FAILED       = 14,
    NO_STREAM           = 15,
    CONFIGURE_REJECTED  = 16,
    STORAGE_FULL        = 17,
    RECORDING_NOT_FOUND = 18,
    FILE_NOT_FOUND      = 19,
    INVALID_PARAMETER   = 20,
    UNSUPPORTED         = 21,
    BUSY                = 22,
    INVALID_STATE       = 23,
    WIFI_NOT_CONNECTED  = 24,
    ORCHESTRATOR_UNREACHABLE = 25,
    CAMERA_UNAVAILABLE  = 26,
    BANDWIDTH_EXCEEDED  = 27,
    AUTH_REQUIRED       = 28,
    AUTH_FAILED         = 29,
    DEVICE_ERROR        = 30,
};

}  // namespace ef

#endif  // EF_INTERNAL_STATUS_HPP
