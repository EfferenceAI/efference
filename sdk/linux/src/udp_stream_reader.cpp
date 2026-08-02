////////////////////////////////////////////////////////////////////////////////
//
// File:      udp_stream_reader.cpp
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

#include "udp_stream_reader.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>

namespace ef {
namespace internal {

namespace {
// Picture-Loss-Indication: 4-byte datagram sent back to the device on a wire
// seq-gap so it forces a keyframe. Matches the device firmware PLI constant.
constexpr uint8_t kPli[4] = {0xEF, 'P', 'L', 'I'};
constexpr uint64_t kPliMinGapNs = 200ull * 1000 * 1000;   // <=5 Hz, matches device

uint64_t mono_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}
}  // namespace

Status UdpStreamReader::start(uint16_t port, bool /*want_video*/,
                              bool /*want_imu*/, int verbose) {
    verbose_ = verbose;
    if (const char* t = getenv("EF_UDP_TEST_PLI_MS")) test_pli_ms_ = atoi(t);
    fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd_ < 0) return Status::NO_STREAM;

    int rcv = 8 * 1024 * 1024;
    setsockopt(fd_, SOL_SOCKET, SO_RCVBUF, &rcv, sizeof rcv);        // best-effort
    int one = 1;
    setsockopt(fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    // Periodic timeout so the recv loop notices stop_ without needing the socket
    // to be shut down out from under a blocking recv.
    struct timeval to { 0, 200 * 1000 };   // 200 ms
    setsockopt(fd_, SOL_SOCKET, SO_RCVTIMEO, &to, sizeof to);

    struct sockaddr_in a;
    std::memset(&a, 0, sizeof a);
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port        = htons(port);
    if (bind(fd_, (struct sockaddr*)&a, sizeof a) < 0) {
        if (verbose_) fprintf(stderr, "[ef] udp reader bind :%u failed: %s\n",
                              (unsigned)port, strerror(errno));
        ::close(fd_);
        fd_ = -1;
        return Status::NO_STREAM;
    }
    if (verbose_)
        fprintf(stderr, "[ef] udp reader bound on :%u\n", (unsigned)port);

    running_.store(true);
    stop_.store(false);
    thread_ = std::thread(&UdpStreamReader::recv_loop, this);
    return Status::SUCCESS;
}

void UdpStreamReader::stop() {
    if (!running_.load() && fd_ < 0) return;
    stop_.store(true);
    mark_ended();
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    running_.store(false);
}

void UdpStreamReader::recv_loop() {
    // A datagram is one video fragment (44 + <=1400) or one IMU batch (<=~1 KB);
    // 64 KB is the UDP max and covers any config without truncation.
    std::vector<uint8_t> buf(65536);
    while (!stop_.load()) {
        // EF_UDP_TEST_PLI_MS: unsupported debug knob, off unless set. Emits periodic
        // PLIs to exercise force-IDR on a clean link.
        if (test_pli_ms_ > 0 && have_peer_) {
            uint64_t now = mono_ns();
            if (!last_test_pli_ns_ ||
                now - last_test_pli_ns_ >= (uint64_t)test_pli_ms_ * 1000000ull) {
                last_test_pli_ns_ = now;
                on_loss();   // reuses the rate-limited PLI sender
            }
        }
        sockaddr_in from{};
        socklen_t   fromlen = sizeof from;
        ssize_t n = ::recvfrom(fd_, buf.data(), buf.size(), 0,
                               (sockaddr*)&from, &fromlen);
        if (n > 0) {
            peer_ = from;           // remember the device addr for PLI feedback
            have_peer_ = true;
            on_packet(buf.data(), (int)n);   // may call on_loss() on a seq-gap
            continue;
        }
        if (n == 0)
            continue;   // a zero-length UDP datagram is valid, not a stream end
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
            continue;   // recv timeout / signal -> re-check stop_
        break;          // genuine fatal recv error
    }
    mark_ended();       // no more frames will arrive
}

// Called from on_packet (this same recv thread) on a video seq-gap.
void UdpStreamReader::on_loss() {
    if (!have_peer_ || fd_ < 0) return;
    uint64_t now = mono_ns();
    if (last_pli_ns_ && now - last_pli_ns_ < kPliMinGapNs) return;   // rate limit
    last_pli_ns_ = now;
    ::sendto(fd_, kPli, sizeof kPli, MSG_DONTWAIT,
             (const sockaddr*)&peer_, sizeof peer_);   // best-effort
    if (verbose_) fprintf(stderr, "[ef] loss -> PLI (request keyframe)\n");
}

}  // namespace internal
}  // namespace ef
