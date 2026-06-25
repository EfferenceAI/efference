// usb_transport.hpp — libusb-1.0 transport for the SDK Endpoint wire protocol.
//
// Discovers the M1 (39c5:0001), auto-finds the FF/EF/03 vendor interface and
// its bulk OUT/IN endpoints (their addresses shift when ADB is also enumerated,
// so we read them from the descriptors, never hard-code), claims it, and does
// single-in-flight framed request/response.
#ifndef EF_USB_TRANSPORT_HPP
#define EF_USB_TRANSPORT_HPP

#include <cstdint>
#include <string>

#include <libusb-1.0/libusb.h>

#include "ef/device.hpp"

namespace ef {
namespace detail {

class UsbTransport {
public:
    UsbTransport() = default;
    ~UsbTransport() { close(); }
    UsbTransport(const UsbTransport&)            = delete;
    UsbTransport& operator=(const UsbTransport&) = delete;

    ERROR_CODE open(int device_index, int verbose);
    void       close();
    bool       is_open() const { return handle_ != nullptr; }

    // Frame {req,args} → bulk OUT → bulk IN → return the response payload JSON.
    // Throws ef::Exception on transport/protocol failure (incl. ERROR frames).
    std::string request(const std::string& req, const std::string& args);

    // USB iSerialNumber string descriptor (empty if none). Used as a fallback
    // serial source when the firmware reply omits serial_number.
    const std::string& serial_descriptor() const { return serial_; }

private:
    libusb_context*       ctx_     = nullptr;
    libusb_device_handle* handle_  = nullptr;
    int                   iface_   = -1;
    uint8_t               ep_out_  = 0;
    uint8_t               ep_in_   = 0;
    uint32_t              corr_    = 0;
    int                   verbose_ = 0;
    std::string           serial_;
};

}  // namespace detail
}  // namespace ef

#endif  // EF_USB_TRANSPORT_HPP
