// proto.hpp — SDK Endpoint wire protocol (v1), host side.
//
// MUST stay byte-compatible with the firmware:
// project/app/efference-sdk-endpoint/src/proto.h. A frame is a 12-byte
// little-endian header followed by `len` bytes of UTF-8 JSON:
//
//   off sz field
//   0   1  magic    0xEF
//   1   1  version  1
//   2   1  type     1=REQUEST 2=RESPONSE 3=ERROR 4=EVENT
//   3   1  flags    reserved (0)
//   4   4  corr_id  request id; RESPONSE/ERROR echoes the REQUEST's id
//   8   4  len      payload length
//   12  .. payload  JSON
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
