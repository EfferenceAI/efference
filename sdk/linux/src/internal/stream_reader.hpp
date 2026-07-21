////////////////////////////////////////////////////////////////////////////////
//
// File:      stream_reader.hpp
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

#ifndef EF_STREAM_READER_HPP
#define EF_STREAM_READER_HPP

#include <cstdint>
#include <vector>

#include <libusb-1.0/libusb.h>

#include "stream_assembler.hpp"

namespace ef {
namespace internal {

// USB isoc data-plane reader. Owns a second libusb handle on the M1's isoc
// streaming interface + an event thread; feeds each isoc packet to the shared
// StreamAssembler (reassembly / quad buffer / IMU ring / consumer API live there).
class StreamReader : public StreamAssembler {
public:
    StreamReader() = default;
    ~StreamReader() override { stop(); }

    // Claim the isoc interface, submit transfers, start the event thread.
    // NO_STREAM if the device exposes no isoc IN endpoint.
    Status start(int device_index, bool want_video, bool want_imu, int verbose);

    // Cancel transfers, join the thread, release the interface. Idempotent.
    void stop() override;

private:
    static void LIBUSB_CALL on_xfer(libusb_transfer* t);
    void handle_iso(libusb_transfer* t);
    void event_loop();

    libusb_context*       ctx_    = nullptr;
    libusb_device_handle* handle_ = nullptr;
    int                   iface_  = -1;
    uint8_t               video_ep_ = 0, imu_ep_ = 0;
    int                   iso_sz_ = 0;
    int                   verbose_ = 0;

    std::vector<libusb_transfer*> xfers_;      // video + imu, all in one pool
};

}  // namespace internal
}  // namespace ef

#endif  // EF_STREAM_READER_HPP
