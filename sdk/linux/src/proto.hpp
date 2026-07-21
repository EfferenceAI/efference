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

// MUST stay byte-compatible with the firmware's frame header. A frame is a
// 12-byte little-endian header followed by `len` bytes of payload. The payload
// is a serialized protobuf ef.v1.Request/Response on the primary control plane;
// the legacy request() path instead carries UTF-8 JSON in the same frame.
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

namespace ef {
namespace proto {

constexpr uint8_t  MAGIC          = 0xEF;
constexpr uint8_t  VERSION        = 1;
constexpr uint32_t HDR_LEN        = 12;
constexpr uint32_t MAX_PAYLOAD    = 8192;  // == EFR_CTL_MSG_MAX
constexpr uint32_t MAX_FRAME      = HDR_LEN + MAX_PAYLOAD;

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

}  // namespace proto
}  // namespace ef

#endif  // EF_PROTO_HPP
