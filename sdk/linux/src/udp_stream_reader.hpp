////////////////////////////////////////////////////////////////////////////////
//
// File:      udp_stream_reader.hpp
// Purpose:   WiFi/UDP data-plane reader (internal).
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

#ifndef EF_UDP_STREAM_READER_HPP
#define EF_UDP_STREAM_READER_HPP

#include <cstdint>

#include <netinet/in.h>

#include "stream_assembler.hpp"

namespace ef {
namespace internal {

// WiFi data-plane reader. Binds a UDP socket and feeds each received datagram to
// the shared StreamAssembler. The device multiplexes video + IMU onto one UDP
// flow (it demuxes by the ef_stream packet type), so a single bound port carries
// both. The control plane (which told the device to stream here) is USB or BLE
// and is independent of this reader.
class UdpStreamReader : public StreamAssembler {
public:
    UdpStreamReader() = default;
    ~UdpStreamReader() override { stop(); }

    // Bind a UDP socket on `port` (all interfaces) and start the recv thread.
    // want_video/want_imu are advisory: the reader accepts whatever the device
    // sends and demuxes by packet type. Returns SUCCESS or NO_STREAM.
    Status start(uint16_t port, bool want_video, bool want_imu, int verbose);

    // Close the socket, join the recv thread. Idempotent.
    void stop() override;

private:
    void recv_loop();

    // On a wire seq-gap, send a Picture-Loss-Indication back to the sender so the
    // device forces a keyframe (rate-limited to ~5 Hz to match the device gate).
    // Runs on the recv thread (same as recv_loop), so peer_/last_pli_ns_ need no lock.
    void on_loss() override;

    int         fd_      = -1;
    int         verbose_ = 0;
    sockaddr_in peer_{};             // sender addr, captured by recvfrom
    bool        have_peer_ = false;
    uint64_t    last_pli_ns_ = 0;    // steady-clock ns of the last PLI sent
    // Test aid (EF_UDP_TEST_PLI_MS): emit a PLI every N ms regardless of loss, so
    // the force-IDR round-trip can be exercised on a clean link (no tc netem). 0=off.
    int         test_pli_ms_ = 0;
    uint64_t    last_test_pli_ns_ = 0;
};

}  // namespace internal
}  // namespace ef

#endif  // EF_UDP_STREAM_READER_HPP
