////////////////////////////////////////////////////////////////////////////////
//
// File:      usb_connection.hpp
// Purpose:   libusb control connection (internal).
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

#ifndef EF_USB_CONNECTION_HPP
#define EF_USB_CONNECTION_HPP

#include <cstdint>
#include <string>

#include <libusb-1.0/libusb.h>

#include "connection.hpp"
#include "internal_status.hpp"

namespace ef {
namespace internal {

class UsbConnection : public Connection {
public:
    UsbConnection() = default;
    ~UsbConnection() override { close(); }
    UsbConnection(const UsbConnection&)            = delete;
    UsbConnection& operator=(const UsbConnection&) = delete;

    // USB-specific bring-up (not part of the Connection interface): discover the
    // M1, claim the vendor interface, latch the endpoint addresses.
    Status open(int device_index, int verbose);

    void close() override;
    bool is_open() const override { return handle_ != nullptr; }

    // {req,args} JSON round-trip → response payload in `out`. Returns Status.
    Status request(const std::string& req, const std::string& args,
                   std::string& out) override;

    // Raw framed round-trip for the protobuf control plane (see Connection).
    Status request_raw(const std::string& payload, std::string& out,
                       uint8_t* out_type) override;

    // USB iSerialNumber string descriptor (empty if none). Used as a fallback
    // serial source when the firmware reply omits serial_number.
    const std::string& serial_descriptor() const { return serial_; }

private:
    libusb_context*       ctx_     = nullptr;
    libusb_device_handle* handle_  = nullptr;
    int                   iface_    = -1;
    uint8_t               ep_out_   = 0;
    uint8_t               ep_in_    = 0;  // bulk IN: control responses
    uint32_t              corr_     = 0;
    int                   verbose_ = 0;
    std::string           serial_;
};

}  // namespace internal
}  // namespace ef

#endif  // EF_USB_CONNECTION_HPP
