#include "usb_transport.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#include "proto.hpp"

namespace ef {
namespace detail {

namespace {
constexpr uint16_t kVid          = 0x39c5;
constexpr uint16_t kPid          = 0x0001;
constexpr uint8_t  kIfClass      = 0xFF;
constexpr uint8_t  kIfSubClass   = 0xEF;
constexpr uint8_t  kIfProtoSdk   = 0x03;
constexpr unsigned kTimeoutMs    = 2000;

// Locate the FF/EF/03 interface + its bulk endpoint addresses. The control IN
// (ep2) is the FIRST bulk IN; the MCAP stream IN (ep3, M0+) is the SECOND, in
// descriptor order — matching the firmware's descriptor layout. ep_stream is 0
// on pre-M0 firmware that exposes only the control IN. Addresses are read from
// the descriptors (they shift when ADB also enumerates), never hard-coded.
int find_sdk_iface(libusb_device* dev, int* iface, uint8_t* ep_out,
                   uint8_t* ep_in, uint8_t* ep_stream) {
    libusb_config_descriptor* cfg = nullptr;
    if (libusb_get_active_config_descriptor(dev, &cfg) != 0) return -1;
    int found = -1;
    for (int i = 0; i < cfg->bNumInterfaces && found < 0; i++) {
        const libusb_interface& itf = cfg->interface[i];
        for (int a = 0; a < itf.num_altsetting; a++) {
            const libusb_interface_descriptor& id = itf.altsetting[a];
            if (id.bInterfaceClass != kIfClass ||
                id.bInterfaceSubClass != kIfSubClass ||
                id.bInterfaceProtocol != kIfProtoSdk)
                continue;
            uint8_t o = 0, in = 0, strm = 0;
            for (int e = 0; e < id.bNumEndpoints; e++) {
                const libusb_endpoint_descriptor& ep = id.endpoint[e];
                if ((ep.bmAttributes & 0x03) != LIBUSB_TRANSFER_TYPE_BULK) continue;
                if (ep.bEndpointAddress & LIBUSB_ENDPOINT_IN) {
                    if      (!in)   in   = ep.bEndpointAddress;  // 1st IN = control
                    else if (!strm) strm = ep.bEndpointAddress;  // 2nd IN = stream
                } else {
                    o = ep.bEndpointAddress;
                }
            }
            if (o && in) {
                *iface     = id.bInterfaceNumber;
                *ep_out    = o;
                *ep_in     = in;
                *ep_stream = strm;
                found      = 0;
            }
        }
    }
    libusb_free_config_descriptor(cfg);
    return found;
}
}  // namespace

ERROR_CODE UsbTransport::open(int device_index, int verbose) {
    if (handle_) return ERROR_CODE::ALREADY_OPEN;
    verbose_ = verbose;

    if (libusb_init(&ctx_) < 0) { ctx_ = nullptr; return ERROR_CODE::USB_ERROR; }

    libusb_device** list = nullptr;
    ssize_t cnt = libusb_get_device_list(ctx_, &list);
    if (cnt < 0) { libusb_exit(ctx_); ctx_ = nullptr; return ERROR_CODE::USB_ERROR; }

    libusb_device* chosen = nullptr;
    int match = 0;
    for (ssize_t k = 0; k < cnt; k++) {
        libusb_device_descriptor dd{};
        if (libusb_get_device_descriptor(list[k], &dd) != 0) continue;
        if (dd.idVendor == kVid && dd.idProduct == kPid) {
            if (match == device_index) { chosen = list[k]; break; }
            match++;
        }
    }
    if (!chosen) {
        libusb_free_device_list(list, 1);
        libusb_exit(ctx_); ctx_ = nullptr;
        return ERROR_CODE::DEVICE_NOT_FOUND;
    }

    int rc = libusb_open(chosen, &handle_);
    if (rc == 0) {
        // Read iSerialNumber while we still hold the device reference.
        libusb_device_descriptor dd{};
        if (libusb_get_device_descriptor(chosen, &dd) == 0 && dd.iSerialNumber) {
            unsigned char tmp[256];
            int sl = libusb_get_string_descriptor_ascii(handle_, dd.iSerialNumber,
                                                         tmp, sizeof tmp);
            if (sl > 0) serial_.assign((const char*)tmp, (size_t)sl);
        }
    }
    libusb_free_device_list(list, 1);

    if (rc != 0) {
        handle_ = nullptr;
        libusb_exit(ctx_); ctx_ = nullptr;
        return (rc == LIBUSB_ERROR_ACCESS) ? ERROR_CODE::ACCESS_DENIED
                                           : ERROR_CODE::USB_ERROR;
    }

    if (find_sdk_iface(libusb_get_device(handle_), &iface_, &ep_out_, &ep_in_,
                       &ep_stream_) != 0) {
        close();
        return ERROR_CODE::INTERFACE_NOT_FOUND;
    }
    if (verbose_)
        fprintf(stderr,
                "[ef] SDK iface=%d ep_out=0x%02x ep_in=0x%02x ep_stream=0x%02x serial=\"%s\"\n",
                iface_, ep_out_, ep_in_, ep_stream_, serial_.c_str());

    libusb_set_auto_detach_kernel_driver(handle_, 1);
    rc = libusb_claim_interface(handle_, iface_);
    if (rc < 0) {
        ERROR_CODE ec = (rc == LIBUSB_ERROR_ACCESS) ? ERROR_CODE::ACCESS_DENIED
                                                    : ERROR_CODE::CLAIM_FAILED;
        close();
        return ec;
    }
    return ERROR_CODE::SUCCESS;
}

void UsbTransport::close() {
    if (handle_) {
        if (iface_ >= 0) libusb_release_interface(handle_, iface_);
        libusb_close(handle_);
        handle_ = nullptr;
    }
    if (ctx_) { libusb_exit(ctx_); ctx_ = nullptr; }
    iface_  = -1;
    ep_out_ = ep_in_ = ep_stream_ = 0;
}

int UsbTransport::read_stream(uint8_t* buf, int len, unsigned timeout_ms, int* got) {
    *got = 0;
    if (!handle_ || !ep_stream_) return LIBUSB_ERROR_NOT_FOUND;
    int rc = libusb_bulk_transfer(handle_, ep_stream_, buf, len, got, timeout_ms);
    // A timeout with bytes already received is success for our drain loop.
    if (rc == LIBUSB_ERROR_TIMEOUT) return 0;
    return rc;
}

std::string UsbTransport::request(const std::string& req, const std::string& args) {
    if (!handle_) throw Exception("transport not open");

    std::string payload = "{\"req\":\"" + req + "\",\"args\":" +
                          (args.empty() ? std::string("{}") : args) + "}";
    if (payload.size() > proto::MAX_PAYLOAD) throw Exception("request payload too large");

    std::vector<uint8_t> frame(proto::HDR_LEN + payload.size());
    uint32_t corr = ++corr_;
    frame[0] = proto::MAGIC;
    frame[1] = proto::VERSION;
    frame[2] = proto::REQUEST;
    frame[3] = 0;
    proto::put_le32(&frame[4], corr);
    proto::put_le32(&frame[8], (uint32_t)payload.size());
    std::memcpy(&frame[proto::HDR_LEN], payload.data(), payload.size());

    int sent = 0;
    int rc = libusb_bulk_transfer(handle_, ep_out_, frame.data(), (int)frame.size(),
                                  &sent, kTimeoutMs);
    if (rc < 0) throw Exception(std::string("bulk OUT failed: ") + libusb_strerror(rc));
    if (verbose_) fprintf(stderr, "[ef] sent %d B (req=%s corr=%u)\n", sent, req.c_str(), corr);

    // Accumulate the response: read the header, then the declared payload.
    std::vector<uint8_t> buf(proto::MAX_FRAME);
    size_t   have    = 0;
    uint32_t need    = proto::HDR_LEN;
    uint32_t plen    = 0;
    bool     got_hdr = false;
    while (have < need) {
        int got = 0;
        rc = libusb_bulk_transfer(handle_, ep_in_, buf.data() + have,
                                  (int)(buf.size() - have), &got, kTimeoutMs);
        if (rc < 0) throw Exception(std::string("bulk IN failed: ") + libusb_strerror(rc));
        if (got == 0) continue;  // ZLP
        have += (size_t)got;
        if (!got_hdr && have >= proto::HDR_LEN) {
            if (buf[0] != proto::MAGIC || buf[1] != proto::VERSION)
                throw Exception("malformed response header");
            plen = proto::get_le32(&buf[8]);
            if (plen > proto::MAX_PAYLOAD) throw Exception("response payload too large");
            need    = proto::HDR_LEN + plen;
            got_hdr = true;
        }
    }

    uint8_t type = buf[2];
    uint32_t corr_in = proto::get_le32(&buf[4]);
    if (verbose_)
        fprintf(stderr, "[ef] reply type=%u corr=%u len=%u\n", type, corr_in, plen);
    std::string out((const char*)buf.data() + proto::HDR_LEN, plen);

    if (type == proto::ERROR)   throw Exception("device error frame: " + out);
    if (type != proto::RESPONSE) throw Exception("unexpected frame type");
    return out;
}

}  // namespace detail
}  // namespace ef
