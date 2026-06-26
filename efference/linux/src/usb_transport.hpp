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

    // True if the device exposed a 2nd bulk IN (the MCAP stream endpoint, ep3).
    // Pre-M0 firmware has only the control IN, so streaming is unavailable.
    bool has_stream() const { return ep_stream_ != 0; }

    // Read up to `len` raw bytes off the stream IN (ep3). Returns the libusb rc
    // (0 on success or timeout); *got holds the byte count (0 on timeout). Does
    // NOT throw — the stream drain loop interprets timeouts vs errors itself.
    int read_stream(uint8_t* buf, int len, unsigned timeout_ms, int* got);

    // USB iSerialNumber string descriptor (empty if none). Used as a fallback
    // serial source when the firmware reply omits serial_number.
    const std::string& serial_descriptor() const { return serial_; }

private:
    libusb_context*       ctx_     = nullptr;
    libusb_device_handle* handle_  = nullptr;
    int                   iface_    = -1;
    uint8_t               ep_out_   = 0;
    uint8_t               ep_in_    = 0;  // 1st bulk IN: control responses (ep2)
    uint8_t               ep_stream_ = 0; // 2nd bulk IN: MCAP byte stream (ep3)
    uint32_t              corr_     = 0;
    int                   verbose_ = 0;
    std::string           serial_;
};

}  // namespace detail
}  // namespace ef

#endif  // EF_USB_TRANSPORT_HPP
