////////////////////////////////////////////////////////////////////////////////
//
// File:      proto.hpp
// Purpose:   SDK Endpoint wire framing (internal).
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

// MUST stay byte-compatible with the firmware's frame header: a 12-byte
// little-endian header + `len` payload bytes. Payload is a serialized protobuf
// ef.v1.Request/Response (primary control plane); the legacy request() path
// carries UTF-8 JSON in the same frame.
//
//   off sz field
//   0   1  magic    0xEF
//   1   1  version  1
//   2   1  type     1=REQUEST 2=RESPONSE 3=ERROR 4=EVENT
//   3   1  flags    reserved (0)
//   4   4  corr_id  request id; RESPONSE/ERROR echoes the REQUEST's id
//   8   4  len      payload length
//   12  .. payload  protobuf ef.v1.Request/Response (or JSON on the legacy path)
#ifndef EF_PROTO_HPP
#define EF_PROTO_HPP

#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>

namespace ef {
namespace proto {

constexpr uint8_t  MAGIC          = 0xEF;
constexpr uint8_t  VERSION        = 1;
constexpr uint32_t HDR_LEN        = 12;
constexpr uint32_t MAX_PAYLOAD    = 8192;  // matches the device's control-message cap
constexpr uint32_t MAX_FRAME      = HDR_LEN + MAX_PAYLOAD;

// Cap on a reassembly buffer. Two frames, not one: a late reply to a request that
// already timed out can sit at full size in front of the one being assembled, and
// that is the case scan_for_reply exists to recover.
constexpr size_t   MAX_RX         = 2 * (size_t)MAX_FRAME;

enum Type : uint8_t {
    REQUEST  = 1,
    RESPONSE = 2,
    ERROR    = 3,
    EVENT    = 4,
};

inline void put_le32(uint8_t* p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

inline uint32_t get_le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

// Is there a parseable frame header at the head of `buf`? Magic, version and a
// payload length within the cap -- exactly what the device validates, and no
// more. Type, flags and corr_id are NOT checked: a host stricter than the device
// rejects frames the device considers well-formed.
inline bool hdr_ok(const uint8_t* p, size_t n) {
    return n >= HDR_LEN && p[0] == MAGIC && p[1] == VERSION &&
           get_le32(p + 8) <= MAX_PAYLOAD;
}

enum class Scan {
    NEED_MORE,  // nothing decidable yet; feed more bytes
    MATCH,      // rx[0..] is a complete frame for `corr`
};

// Walk `rx` to the next complete frame that answers `corr`, consuming what it
// passes. A header that does not parse means the stream is mid-frame -- usually
// the tail of a reply whose request already timed out -- so resync to the next
// possible frame start rather than failing the exchange, which is what the device
// does. A complete frame that is an EVENT or answers a different request is
// skipped whole. On MATCH, rx begins with the wanted frame; otherwise rx holds
// only what might still become one.
inline Scan scan_for_reply(std::vector<uint8_t>& rx, uint32_t corr) {
    for (;;) {
        if (rx.size() < HDR_LEN) return Scan::NEED_MORE;
        if (!hdr_ok(rx.data(), rx.size())) {
            // A frame must begin with MAGIC, so the next candidate start is the
            // next MAGIC byte after this one; drop that whole run in one erase
            // rather than a memmove per byte. This collapses ordinary garbage to a
            // single erase; a MAGIC-DENSE run still costs one erase per byte, which
            // is left alone deliberately -- bounded by MAX_RX, off the happy path,
            // and on the host rather than the device.
            const void* m = std::memchr(rx.data() + 1, MAGIC, rx.size() - 1);
            rx.erase(rx.begin(),
                     m ? rx.begin() + ((const uint8_t*)m - rx.data()) : rx.end());
            continue;
        }
        uint32_t plen = get_le32(&rx[8]);
        if (rx.size() < HDR_LEN + plen) return Scan::NEED_MORE;
        // Only a RESPONSE or an ERROR answers a request. Testing "not an EVENT"
        // would also accept a REQUEST and every undefined type byte, and this
        // function runs precisely when the stream is desynced -- so a garbage run
        // that happens to parse as a header with a matching corr_id would be
        // handed up as a reply. The USB path already tests it this way.
        if ((rx[2] == RESPONSE || rx[2] == ERROR) && get_le32(&rx[4]) == corr)
            return Scan::MATCH;
        rx.erase(rx.begin(), rx.begin() + HDR_LEN + plen);
    }
}

// Append `n` bytes to a reassembly buffer, bounded by MAX_RX. Returns false and
// clears `rx` if they would not fit: past the cap the stream has stopped carrying
// frames, and keeping the bytes only feeds a scan that cannot succeed. Every
// transport using scan_for_reply must feed it through here -- the bound is half
// the reassembler.
inline bool append_bounded(std::vector<uint8_t>& rx, const uint8_t* p, size_t n) {
    if (rx.size() + n > MAX_RX) {
        rx.clear();
        return false;
    }
    rx.insert(rx.end(), p, p + n);
    return true;
}

}  // namespace proto
}  // namespace ef

#endif  // EF_PROTO_HPP
