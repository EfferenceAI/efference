////////////////////////////////////////////////////////////////////////////////
//
// File:      mcap.hpp
// Purpose:   Minimal MCAP container writer/reader + foxglove message codecs
//            (internal).
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
//
// Wire/topic layout mirrors the on-device recorder, so host and device-local
// recordings share one file format:
//
//   /camera/color/0/image        foxglove.CompressedVideo (protobuf)
//   /camera/color/0/image_raw    foxglove.RawImage        (protobuf, nv12)
//   /camera/imu/0/accel/sample   foxglove.PoseInFrame     (position = m/s^2)
//   /camera/imu/0/gyro/sample    foxglove.PoseInFrame     (position = rad/s)
//
// Written unchunked/uncompressed with no summary (device uses noChunking +
// Compression::None), so the reader only walks top-level Schema/Channel/Message
// records (plus uncompressed Chunk records for safety).
//
////////////////////////////////////////////////////////////////////////////////

#ifndef EF_MCAP_HPP
#define EF_MCAP_HPP

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace ef {
namespace internal {
namespace mcap {

// ---- record opcodes (mcap spec) ---------------------------------------------
enum : uint8_t {
    OP_HEADER   = 0x01,
    OP_FOOTER   = 0x02,
    OP_SCHEMA   = 0x03,
    OP_CHANNEL  = 0x04,
    OP_MESSAGE  = 0x05,
    OP_CHUNK    = 0x06,
    OP_DATA_END = 0x0F,
};

// ---- topology shared by the host writer and the replay reader ---------------
// Ids/topics/frame ids match the device recorder verbatim.
constexpr uint16_t SCH_VIDEO = 1, SCH_RAW = 2, SCH_POSE = 3;
constexpr uint16_t CH_IMAGE = 1, CH_IMAGE_RAW = 2, CH_ACCEL = 3, CH_GYRO = 4;
constexpr const char* TOPIC_IMAGE     = "/camera/color/0/image";
constexpr const char* TOPIC_IMAGE_RAW = "/camera/color/0/image_raw";
constexpr const char* TOPIC_ACCEL     = "/camera/imu/0/accel/sample";
constexpr const char* TOPIC_GYRO      = "/camera/imu/0/gyro/sample";
constexpr const char* FRAME_COLOR = "color0";
constexpr const char* FRAME_IMU   = "imu0";

// ---- writer -------------------------------------------------------------------
// Sequential MCAP emitter: magic + Header, then schemas/channels/messages in
// call order, then finish() (DataEnd + Footer + magic). Not thread-safe; the
// owner (HostRecorder) serializes on its writer thread.
class Writer {
public:
    ~Writer() { finish(); }

    bool open(const std::string& path);
    bool add_schema(uint16_t id, const char* name,
                    const unsigned char* fds, size_t fds_len);
    bool add_channel(uint16_t id, uint16_t schema_id, const char* topic);
    bool write_message(uint16_t channel, uint64_t log_time_ns,
                       const uint8_t* data, size_t len);
    bool finish();   // idempotent; closes the file
    bool ok() const { return fp_ != nullptr; }

private:
    bool rec(uint8_t op, const std::vector<uint8_t>& body);
    std::FILE* fp_ = nullptr;
    uint32_t   seq_ = 0;    // per-file message sequence
};

// ---- reader -------------------------------------------------------------------
// Sequential top-level record walker: uncompressed Chunks unpacked in place,
// compressed ones fail the read (writers never emit them). next() returns false
// at Footer/EOF/parse error.
class Reader {
public:
    ~Reader() { close(); }

    struct Record {
        uint8_t              op = 0;
        std::vector<uint8_t> body;
    };

    bool open(const std::string& path);
    bool next(Record* out);
    void close();
    // rewind to the first record (after the leading magic)
    bool rewind();

private:
    std::FILE*           fp_ = nullptr;
    // Current uncompressed chunk, parsed lazily one sub-record per next() (never
    // duplicated whole into a separate record list).
    std::vector<uint8_t> chunk_;
    size_t               chunk_pos_ = 0;   // cursor into chunk_ (next sub-record)
    size_t               chunk_end_ = 0;   // end of the sub-record stream in chunk_
};

// ---- foxglove message codecs ----------------------------------------------------
// Hand-rolled protobuf (the three messages are tiny and fixed); byte-for-byte
// the field layout the device recorder emits.

// foxglove.CompressedVideo { timestamp=1, frame_id=2, data=3, format=4 }
void enc_compressed_video(std::vector<uint8_t>& out, uint64_t ts_ns,
                          const char* frame_id, const char* format,
                          const uint8_t* data, size_t len);

// foxglove.RawImage { timestamp=1, frame_id=2, width=3, height=4,
//                     encoding=5, step=6, data=7 }
void enc_raw_image(std::vector<uint8_t>& out, uint64_t ts_ns,
                   const char* frame_id, uint32_t width, uint32_t height,
                   const char* encoding, uint32_t step,
                   const uint8_t* data, size_t len);

// foxglove.PoseInFrame { timestamp=1, frame_id=2,
//                        pose=3 { position=1{x,y,z}, orientation=2{x,y,z,w} } }
void enc_pose_in_frame(std::vector<uint8_t>& out, uint64_t ts_ns,
                       const char* frame_id, double x, double y, double z);

// Decoded views of the same messages (replay side). Pointers alias the input.
struct VideoMsg {
    uint64_t       ts_ns = 0;
    const uint8_t* data  = nullptr;
    size_t         size  = 0;
    char           format[16] = {0};   // "h264"/"h265"/"jpeg"/"nv12"/...
    uint32_t       width = 0, height = 0;   // RawImage only
};
bool dec_compressed_video(const uint8_t* p, size_t n, VideoMsg* out);
bool dec_raw_image(const uint8_t* p, size_t n, VideoMsg* out);

struct PoseMsg {
    uint64_t ts_ns = 0;
    double   x = 0, y = 0, z = 0;   // pose.position
};
bool dec_pose_in_frame(const uint8_t* p, size_t n, PoseMsg* out);

// The embedded FileDescriptorSet blobs (mcap_schemas.inc), exposed for the
// writer's add_schema calls.
const unsigned char* fds_compressed_video(size_t* len);
const unsigned char* fds_raw_image(size_t* len);
const unsigned char* fds_pose_in_frame(size_t* len);

}  // namespace mcap
}  // namespace internal
}  // namespace ef

#endif  // EF_MCAP_HPP
