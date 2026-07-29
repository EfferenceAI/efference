////////////////////////////////////////////////////////////////////////////////
//
// File:      stream_reader.cpp
// Purpose:   Isochronous (USB) data-plane reader (internal).
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

#include "stream_reader.hpp"

#include <cstdio>
#include <cstdlib>

namespace ef {
namespace internal {

namespace {
constexpr uint16_t kVid = 0x39c5;
constexpr uint16_t kPid = 0x0001;

// isoc transfer pool sizing.
constexpr int kVidTransfers = 16, kVidPkts = 32;
constexpr int kImuTransfers = 4,  kImuPkts = 8;
}  // namespace

// ---- discovery + bring-up --------------------------------------------------

Status StreamReader::start(int device_index, bool want_video, bool want_imu,
                               int verbose) {
    verbose_ = verbose;
    if (libusb_init(&ctx_) < 0) { ctx_ = nullptr; return Status::USB_ERROR; }

    // find the Nth M1
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
    if (!chosen) { libusb_free_device_list(list, 1); libusb_exit(ctx_); ctx_ = nullptr;
                   return Status::DEVICE_NOT_FOUND; }

    int rc = libusb_open(chosen, &handle_);
    libusb_device* dev = libusb_ref_device(chosen);
    libusb_free_device_list(list, 1);
    if (rc != 0) {
        libusb_unref_device(dev);
        handle_ = nullptr; libusb_exit(ctx_); ctx_ = nullptr;
        return (rc == LIBUSB_ERROR_ACCESS) ? Status::INSUFFICIENT_PERMISSIONS : Status::USB_ERROR;
    }

    // Streaming interface = vendor-class (0xFF) alt with isoc IN endpoints
    // (1st = video, 2nd = IMU). isoc lives on alt 0 (FunctionFS has no get_alt).
    libusb_config_descriptor* cfg = nullptr;
    if (libusb_get_active_config_descriptor(dev, &cfg) != 0) {
        libusb_unref_device(dev); libusb_close(handle_); handle_ = nullptr;
        libusb_exit(ctx_); ctx_ = nullptr; return Status::USB_ERROR;
    }
    for (int i = 0; i < cfg->bNumInterfaces && iface_ < 0; i++) {
        const libusb_interface& itf = cfg->interface[i];
        for (int a = 0; a < itf.num_altsetting; a++) {
            const libusb_interface_descriptor& id = itf.altsetting[a];
            if (id.bInterfaceClass != 0xFF) continue;
            uint8_t v = 0, m = 0;
            for (int e = 0; e < id.bNumEndpoints; e++) {
                const libusb_endpoint_descriptor& ep = id.endpoint[e];
                bool iso = (ep.bmAttributes & 0x03) == LIBUSB_TRANSFER_TYPE_ISOCHRONOUS;
                bool in  = ep.bEndpointAddress & LIBUSB_ENDPOINT_IN;
                if (iso && in) { if (!v) v = ep.bEndpointAddress; else if (!m) m = ep.bEndpointAddress; }
            }
            if (v) { iface_ = id.bInterfaceNumber; video_ep_ = v; imu_ep_ = m; break; }
        }
    }
    int imu_iso_sz = 0;
    if (iface_ >= 0) {
        iso_sz_ = libusb_get_max_iso_packet_size(dev, video_ep_);
        // The IMU endpoint's isoc packet is tiny (~1 KB, one ef_stream IMU batch) and
        // MUST get its own size: reusing the video ep's 32 KB size overruns ep4's max
        // packet, so the host controller never completes IMU transfers (timeouts, imu=0).
        if (imu_ep_) imu_iso_sz = libusb_get_max_iso_packet_size(dev, imu_ep_);
    }
    libusb_free_config_descriptor(cfg);
    libusb_unref_device(dev);
    if (iso_sz_ <= 0) iso_sz_ = 32768;
    if (imu_iso_sz <= 0) imu_iso_sz = 4096;

    if (iface_ < 0 || !video_ep_) {
        libusb_close(handle_); handle_ = nullptr; libusb_exit(ctx_); ctx_ = nullptr;
        return Status::NO_STREAM;
    }
    if (verbose_)
        fprintf(stderr, "[ef] isoc iface=%d video=0x%02x imu=0x%02x iso_sz=%d imu_iso_sz=%d\n",
                iface_, video_ep_, imu_ep_, iso_sz_, imu_iso_sz);

    libusb_set_auto_detach_kernel_driver(handle_, 1);
    rc = libusb_claim_interface(handle_, iface_);
    if (rc < 0) {
        // Same EBUSY split as the control interface (usb_connection.cpp): a
        // second client is "busy", not "unavailable".
        Status ec = (rc == LIBUSB_ERROR_ACCESS) ? Status::INSUFFICIENT_PERMISSIONS
                  : (rc == LIBUSB_ERROR_BUSY)   ? Status::BUSY
                                                : Status::CLAIM_FAILED;
        libusb_close(handle_); handle_ = nullptr; libusb_exit(ctx_); ctx_ = nullptr;
        iface_ = -1; return ec;
    }

    // allocate + submit transfers
    auto add = [&](uint8_t ep, int npkts, int ntransfers, int isz) {
        int buflen = npkts * isz;
        for (int i = 0; i < ntransfers; i++) {
            libusb_transfer* t = libusb_alloc_transfer(npkts);
            uint8_t* buf = (uint8_t*)std::malloc(buflen);
            libusb_fill_iso_transfer(t, handle_, ep, buf, buflen, npkts,
                                     &StreamReader::on_xfer, this, 1000);
            libusb_set_iso_packet_lengths(t, isz);
            if (libusb_submit_transfer(t) == 0) xfers_.push_back(t);
            else { std::free(buf); libusb_free_transfer(t); }
        }
    };
    if (want_video) add(video_ep_, kVidPkts, kVidTransfers, iso_sz_);
    if (want_imu && imu_ep_)  add(imu_ep_, kImuPkts, kImuTransfers, imu_iso_sz);
    if (xfers_.empty()) {
        libusb_release_interface(handle_, iface_); libusb_close(handle_); handle_ = nullptr;
        libusb_exit(ctx_); ctx_ = nullptr; iface_ = -1; return Status::USB_ERROR;
    }

    running_.store(true);
    stop_.store(false);
    thread_ = std::thread(&StreamReader::event_loop, this);
    return Status::SUCCESS;
}

void StreamReader::stop() {
    if (!running_.load() && !ctx_) return;
    stop_.store(true);
    mark_ended();
    for (auto* t : xfers_) libusb_cancel_transfer(t);
    if (thread_.joinable()) thread_.join();
    for (auto* t : xfers_) { std::free(t->buffer); libusb_free_transfer(t); }
    xfers_.clear();
    if (handle_) {
        if (iface_ >= 0) libusb_release_interface(handle_, iface_);
        libusb_close(handle_); handle_ = nullptr;
    }
    if (ctx_) { libusb_exit(ctx_); ctx_ = nullptr; }
    iface_ = -1; video_ep_ = imu_ep_ = 0;
    running_.store(false);
}

void StreamReader::event_loop() {
    while (!stop_.load()) {
        struct timeval tv { 0, 100000 };
        libusb_handle_events_timeout_completed(ctx_, &tv, nullptr);
    }
    for (int i = 0; i < 10; i++) {   // drain cancellations
        struct timeval tv { 0, 50000 };
        libusb_handle_events_timeout_completed(ctx_, &tv, nullptr);
    }
}

// ---- transfer completion (feeds the shared assembler) ----------------------

void LIBUSB_CALL StreamReader::on_xfer(libusb_transfer* t) {
    static_cast<StreamReader*>(t->user_data)->handle_iso(t);
}

void StreamReader::handle_iso(libusb_transfer* t) {
    if (t->status == LIBUSB_TRANSFER_CANCELLED) return;
    if (t->status == LIBUSB_TRANSFER_NO_DEVICE || t->status == LIBUSB_TRANSFER_ERROR) {
        mark_ended();
        return;
    }
    for (int i = 0; i < t->num_iso_packets; i++) {
        libusb_iso_packet_descriptor* d = &t->iso_packet_desc[i];
        if (d->status == LIBUSB_TRANSFER_COMPLETED && d->actual_length > 0)
            on_packet(libusb_get_iso_packet_buffer_simple(t, i), (int)d->actual_length);
    }
    if (!stop_.load()) {
        if (libusb_submit_transfer(t) != 0) mark_ended();
    }
}

}  // namespace internal
}  // namespace ef
