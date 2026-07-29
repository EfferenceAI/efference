////////////////////////////////////////////////////////////////////////////////
//
// File:      usb_connection.cpp
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

#include "usb_connection.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

#include "proto.hpp"

namespace ef {
namespace internal {

namespace {
constexpr uint16_t kVid          = 0x39c5;
constexpr uint16_t kPid          = 0x0001;
constexpr uint8_t  kIfClass      = 0xFF;
constexpr uint8_t  kIfSubClass   = 0xEF;
constexpr uint8_t  kIfProtoSdk   = 0x03;
constexpr unsigned kTimeoutMs    = 2000;

// Locate the FF/EF/03 interface + its bulk control endpoints. Video and IMU ride
// the isochronous endpoints on a second interface and are not opened here. Read
// from descriptors (they shift when ADB enumerates), never hard-coded.
int find_sdk_iface(libusb_device* dev, int* iface, uint8_t* ep_out, uint8_t* ep_in) {
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
            uint8_t o = 0, in = 0;
            for (int e = 0; e < id.bNumEndpoints; e++) {
                const libusb_endpoint_descriptor& ep = id.endpoint[e];
                if ((ep.bmAttributes & 0x03) != LIBUSB_TRANSFER_TYPE_BULK) continue;
                if (ep.bEndpointAddress & LIBUSB_ENDPOINT_IN) {
                    if (!in) in = ep.bEndpointAddress;   // 1st bulk IN = control
                } else {
                    o = ep.bEndpointAddress;
                }
            }
            if (o && in) {
                *iface  = id.bInterfaceNumber;
                *ep_out = o;
                *ep_in  = in;
                found   = 0;
            }
        }
    }
    libusb_free_config_descriptor(cfg);
    return found;
}
}  // namespace

Status UsbConnection::open(int device_index, int verbose) {
    if (handle_) return Status::ALREADY_OPEN;
    verbose_ = verbose;

    if (libusb_init(&ctx_) < 0) { ctx_ = nullptr; return Status::USB_ERROR; }

    libusb_device** list = nullptr;
    ssize_t cnt = libusb_get_device_list(ctx_, &list);
    if (cnt < 0) { libusb_exit(ctx_); ctx_ = nullptr; return Status::USB_ERROR; }

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
        return Status::DEVICE_NOT_FOUND;
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
        return (rc == LIBUSB_ERROR_ACCESS) ? Status::INSUFFICIENT_PERMISSIONS
                                           : Status::USB_ERROR;
    }

    if (find_sdk_iface(libusb_get_device(handle_), &iface_, &ep_out_, &ep_in_) != 0) {
        close();
        return Status::INTERFACE_NOT_FOUND;
    }
    if (verbose_)
        fprintf(stderr, "[ef] SDK iface=%d ep_out=0x%02x ep_in=0x%02x serial=\"%s\"\n",
                iface_, ep_out_, ep_in_, serial_.c_str());

    libusb_set_auto_detach_kernel_driver(handle_, 1);
    rc = libusb_claim_interface(handle_, iface_);
    if (rc < 0) {
        // EBUSY = another process holds the SDK interface (one control client per
        // cable); distinct from the device being absent or unreadable.
        Status ec = (rc == LIBUSB_ERROR_ACCESS) ? Status::INSUFFICIENT_PERMISSIONS
                  : (rc == LIBUSB_ERROR_BUSY)   ? Status::BUSY
                                                : Status::CLAIM_FAILED;
        close();
        return ec;
    }

    // Drain bytes the endpoint left queued from a previous process (unread reply or
    // stale EVENT): corr restarts at 0 each open, so a leftover frame could collide
    // with a fresh request's corr id and be mistaken for its reply. Short-timeout
    // reads until empty.
    {
        uint8_t sink[4096];
        for (int i = 0; i < 64; ++i) {
            int got = 0;
            int drc = libusb_bulk_transfer(handle_, ep_in_, sink, (int)sizeof sink, &got, 20);
            if (drc != 0 || got == 0) break;   // timeout/empty/error -> done draining
        }
    }
    return Status::SUCCESS;
}

void UsbConnection::close() {
    if (handle_) {
        if (iface_ >= 0) libusb_release_interface(handle_, iface_);
        libusb_close(handle_);
        handle_ = nullptr;
    }
    if (ctx_) { libusb_exit(ctx_); ctx_ = nullptr; }
    iface_  = -1;
    ep_out_ = ep_in_ = 0;
}

Status UsbConnection::request_raw(const std::string& payload, std::string& out,
                                  uint8_t* out_type) {
    out.clear();
    if (!handle_) return Status::NOT_OPEN;
    if (payload.size() > proto::MAX_PAYLOAD) return Status::USB_ERROR;

    std::vector<uint8_t> frame(proto::HDR_LEN + payload.size());
    uint32_t corr = ++corr_;
    frame[0] = proto::MAGIC;
    frame[1] = proto::VERSION;
    frame[2] = proto::REQUEST;
    frame[3] = 0;
    proto::put_le32(&frame[4], corr);
    proto::put_le32(&frame[8], (uint32_t)payload.size());
    if (!payload.empty())
        std::memcpy(&frame[proto::HDR_LEN], payload.data(), payload.size());

    int sent = 0;
    int rc = libusb_bulk_transfer(handle_, ep_out_, frame.data(), (int)frame.size(),
                                  &sent, kTimeoutMs);
    if (rc < 0) return Status::USB_ERROR;
    if (verbose_) fprintf(stderr, "[ef] sent %d B (corr=%u)\n", sent, corr);

    // Read frames until the RESPONSE/ERROR whose corr_id matches THIS request. A
    // stale reply (prior timed-out call) or unsolicited EVENT could otherwise return
    // the wrong payload. `acc` may hold multiple coalesced frames, so extract from it
    // before reading more.
    std::vector<uint8_t> acc;
    acc.reserve(proto::MAX_FRAME);
    uint8_t chunk[4096];
    for (int guard = 0; guard < 256; ++guard) {
        // Extract every complete frame currently buffered.
        while (acc.size() >= proto::HDR_LEN) {
            if (acc[0] != proto::MAGIC || acc[1] != proto::VERSION)
                return Status::USB_ERROR;
            uint32_t plen = proto::get_le32(&acc[8]);
            if (plen > proto::MAX_PAYLOAD) return Status::USB_ERROR;
            size_t need = proto::HDR_LEN + plen;
            if (acc.size() < need) break;             // wait for the rest
            uint8_t  type  = acc[2];
            uint32_t rcorr = proto::get_le32(&acc[4]);
            bool matches = (type == proto::RESPONSE || type == proto::ERROR) &&
                           (rcorr == corr);
            if (verbose_)
                fprintf(stderr, "[ef] reply type=%u corr=%u len=%u%s\n",
                        type, rcorr, plen, matches ? "" : " (stale/event, skipping)");
            if (matches) {
                if (out_type) *out_type = type;
                out.assign((const char*)acc.data() + proto::HDR_LEN, plen);
                return Status::SUCCESS;
            }
            acc.erase(acc.begin(), acc.begin() + need);  // drop it, keep the rest
        }
        // Need more bytes from the wire.
        int got = 0;
        rc = libusb_bulk_transfer(handle_, ep_in_, chunk, (int)sizeof chunk, &got, kTimeoutMs);
        if (rc < 0) return Status::USB_ERROR;
        if (got > 0) acc.insert(acc.end(), chunk, chunk + (size_t)got);
    }
    return Status::USB_ERROR;   // too many stale frames without a match
}

Status UsbConnection::request(const std::string& req, const std::string& args,
                             std::string& out) {
    std::string payload = "{\"req\":\"" + req + "\",\"args\":" +
                          (args.empty() ? std::string("{}") : args) + "}";
    uint8_t type = 0;
    Status rc = request_raw(payload, out, &type);
    if (rc != Status::SUCCESS)    return rc;
    if (type == proto::ERROR)     return Status::CONFIGURE_REJECTED;
    if (type != proto::RESPONSE)  return Status::USB_ERROR;
    return Status::SUCCESS;
}

}  // namespace internal
}  // namespace ef
