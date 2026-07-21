////////////////////////////////////////////////////////////////////////////////
//
// File:      mcap.cpp
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

#include "mcap.hpp"

#include <cstring>

namespace ef {
namespace internal {
namespace mcap {

#include "mcap_schemas.inc"

const unsigned char* fds_compressed_video(size_t* len) {
    *len = sizeof kFdsCompressedVideo;
    return kFdsCompressedVideo;
}
const unsigned char* fds_raw_image(size_t* len) {
    *len = sizeof kFdsRawImage;
    return kFdsRawImage;
}
const unsigned char* fds_pose_in_frame(size_t* len) {
    *len = sizeof kFdsPoseInFrame;
    return kFdsPoseInFrame;
}

namespace {

constexpr uint8_t kMagic[8] = {0x89, 'M', 'C', 'A', 'P', 0x30, '\r', '\n'};

// Any record whose body exceeds this is treated as corrupt.
constexpr uint64_t kMaxRecord = 256ull * 1024 * 1024;

// ---- little-endian primitive emitters (append to a byte vector) --------------
void put16(std::vector<uint8_t>& v, uint16_t x) { v.push_back((uint8_t)x); v.push_back((uint8_t)(x >> 8)); }
void put32(std::vector<uint8_t>& v, uint32_t x) { put16(v, (uint16_t)x); put16(v, (uint16_t)(x >> 16)); }
void put64(std::vector<uint8_t>& v, uint64_t x) { put32(v, (uint32_t)x); put32(v, (uint32_t)(x >> 32)); }
void put_str(std::vector<uint8_t>& v, const char* s) {
    uint32_t n = (uint32_t)std::strlen(s);
    put32(v, n);
    v.insert(v.end(), s, s + n);
}

uint32_t rd32(const uint8_t* p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 |
           (uint32_t)p[3] << 24;
}
uint64_t rd64(const uint8_t* p) {
    uint64_t lo = rd32(p), hi = rd32(p + 4);
    return lo | (hi << 32);
}

// ---- protobuf mini writer -----------------------------------------------------
constexpr uint32_t WT_VARINT = 0, WT_FIXED64 = 1, WT_LEN = 2;

void pb_varint(std::vector<uint8_t>& v, uint64_t x) {
    do {
        uint8_t b = x & 0x7f;
        x >>= 7;
        if (x) b |= 0x80;
        v.push_back(b);
    } while (x);
}
void pb_key(std::vector<uint8_t>& v, uint32_t field, uint32_t wt) {
    pb_varint(v, ((uint64_t)field << 3) | wt);
}
void pb_f_varint(std::vector<uint8_t>& v, uint32_t field, uint64_t x) {
    pb_key(v, field, WT_VARINT);
    pb_varint(v, x);
}
void pb_f_str(std::vector<uint8_t>& v, uint32_t field, const char* s) {
    pb_key(v, field, WT_LEN);
    size_t n = std::strlen(s);
    pb_varint(v, n);
    v.insert(v.end(), s, s + n);
}
void pb_f_bytes(std::vector<uint8_t>& v, uint32_t field, const uint8_t* d, size_t n) {
    pb_key(v, field, WT_LEN);
    pb_varint(v, n);
    if (n) v.insert(v.end(), d, d + n);
}
void pb_f_double(std::vector<uint8_t>& v, uint32_t field, double x) {
    pb_key(v, field, WT_FIXED64);
    uint64_t bits;
    std::memcpy(&bits, &x, 8);
    put64(v, bits);
}
void pb_f_msg(std::vector<uint8_t>& v, uint32_t field, const std::vector<uint8_t>& sub) {
    pb_key(v, field, WT_LEN);
    pb_varint(v, sub.size());
    v.insert(v.end(), sub.begin(), sub.end());
}
// google.protobuf.Timestamp { seconds=1, nanos=2 } (zero fields omitted)
void pb_f_timestamp(std::vector<uint8_t>& v, uint32_t field, uint64_t ts_ns) {
    std::vector<uint8_t> t;
    uint64_t sec = ts_ns / 1000000000ull;
    uint32_t ns  = (uint32_t)(ts_ns % 1000000000ull);
    if (sec) pb_f_varint(t, 1, sec);
    if (ns)  pb_f_varint(t, 2, ns);
    pb_f_msg(v, field, t);
}

// ---- protobuf mini reader ------------------------------------------------------
struct PbView {
    const uint8_t* p = nullptr;
    size_t         n = 0;
};

bool pb_read_varint(PbView& v, uint64_t* out) {
    uint64_t x = 0;
    for (int shift = 0; shift < 64; shift += 7) {
        if (!v.n) return false;
        uint8_t b = *v.p++;
        v.n--;
        x |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) { *out = x; return true; }
    }
    return false;
}

// One field: number+wiretype, and either the varint value, the fixed64 bits,
// or a view of the length-delimited payload. Unknown wire types fail.
bool pb_next(PbView& v, uint32_t* field, uint32_t* wt, uint64_t* scalar, PbView* len) {
    uint64_t key = 0;
    if (!pb_read_varint(v, &key)) return false;
    *field = (uint32_t)(key >> 3);
    *wt    = (uint32_t)(key & 7);
    switch (*wt) {
        case WT_VARINT:
            return pb_read_varint(v, scalar);
        case WT_FIXED64:
            if (v.n < 8) return false;
            *scalar = rd64(v.p);
            v.p += 8; v.n -= 8;
            return true;
        case WT_LEN: {
            uint64_t n = 0;
            if (!pb_read_varint(v, &n) || n > v.n) return false;
            len->p = v.p;
            len->n = (size_t)n;
            v.p += n; v.n -= (size_t)n;
            return true;
        }
        case 5:   // fixed32
            if (v.n < 4) return false;
            *scalar = rd32(v.p);
            v.p += 4; v.n -= 4;
            return true;
        default:
            return false;
    }
}

uint64_t dec_timestamp(PbView t) {
    uint64_t sec = 0, ns = 0;
    uint32_t f, wt;
    uint64_t s;
    PbView sub;
    while (t.n && pb_next(t, &f, &wt, &s, &sub)) {
        if (f == 1 && wt == WT_VARINT) sec = s;
        if (f == 2 && wt == WT_VARINT) ns = s;
    }
    return sec * 1000000000ull + ns;
}

}  // namespace

// ---- Writer ---------------------------------------------------------------------

bool Writer::rec(uint8_t op, const std::vector<uint8_t>& body) {
    if (!fp_) return false;
    uint8_t hdr[9];
    hdr[0] = op;
    uint64_t n = body.size();
    for (int i = 0; i < 8; i++) hdr[1 + i] = (uint8_t)(n >> (8 * i));
    if (std::fwrite(hdr, 1, 9, fp_) != 9) return false;
    if (n && std::fwrite(body.data(), 1, body.size(), fp_) != body.size())
        return false;
    return true;
}

bool Writer::open(const std::string& path) {
    if (fp_) return false;
    fp_ = std::fopen(path.c_str(), "wb");
    if (!fp_) return false;
    if (std::fwrite(kMagic, 1, 8, fp_) != 8) { finish(); return false; }
    std::vector<uint8_t> b;
    put_str(b, "");                    // profile
    put_str(b, "efference-sdk-host");  // library
    if (!rec(OP_HEADER, b)) { finish(); return false; }
    return true;
}

bool Writer::add_schema(uint16_t id, const char* name,
                        const unsigned char* fds, size_t fds_len) {
    std::vector<uint8_t> b;
    put16(b, id);
    put_str(b, name);
    put_str(b, "protobuf");
    put32(b, (uint32_t)fds_len);
    b.insert(b.end(), fds, fds + fds_len);
    return rec(OP_SCHEMA, b);
}

bool Writer::add_channel(uint16_t id, uint16_t schema_id, const char* topic) {
    std::vector<uint8_t> b;
    put16(b, id);
    put16(b, schema_id);
    put_str(b, topic);
    put_str(b, "protobuf");
    put32(b, 0);   // empty metadata map
    return rec(OP_CHANNEL, b);
}

bool Writer::write_message(uint16_t channel, uint64_t log_time_ns,
                           const uint8_t* data, size_t len) {
    std::vector<uint8_t> b;
    b.reserve(22 + len);
    put16(b, channel);
    put32(b, seq_++);
    put64(b, log_time_ns);   // log_time
    put64(b, log_time_ns);   // publish_time
    if (len) b.insert(b.end(), data, data + len);
    return rec(OP_MESSAGE, b);
}

bool Writer::finish() {
    if (!fp_) return true;
    std::vector<uint8_t> de;
    put32(de, 0);              // data_section_crc: 0 = not computed
    bool ok = rec(OP_DATA_END, de);
    std::vector<uint8_t> ft;
    put64(ft, 0);              // summary_start: 0 = no summary section
    put64(ft, 0);              // summary_offset_start
    put32(ft, 0);              // summary_crc
    ok = rec(OP_FOOTER, ft) && ok;
    ok = (std::fwrite(kMagic, 1, 8, fp_) == 8) && ok;
    ok = (std::fclose(fp_) == 0) && ok;
    fp_ = nullptr;
    return ok;
}

// ---- Reader ---------------------------------------------------------------------

bool Reader::open(const std::string& path) {
    close();
    fp_ = std::fopen(path.c_str(), "rb");
    if (!fp_) return false;
    uint8_t magic[8];
    if (std::fread(magic, 1, 8, fp_) != 8 || std::memcmp(magic, kMagic, 8) != 0) {
        close();
        return false;
    }
    return true;
}

bool Reader::rewind() {
    if (!fp_) return false;
    chunk_.clear(); chunk_pos_ = chunk_end_ = 0;
    return std::fseek(fp_, 8, SEEK_SET) == 0;
}

void Reader::close() {
    if (fp_) { std::fclose(fp_); fp_ = nullptr; }
    chunk_.clear(); chunk_pos_ = chunk_end_ = 0;
}

bool Reader::next(Record* out) {
    for (;;) {
        // Serve the next sub-record from the current chunk, parsed in place so
        // the chunk is never duplicated whole into a separate record list.
        if (chunk_pos_ < chunk_end_) {
            const uint8_t* q = chunk_.data() + chunk_pos_;
            size_t         qn = chunk_end_ - chunk_pos_;
            if (qn < 9) { chunk_pos_ = chunk_end_; continue; }   // trailing junk
            uint64_t bn = 0;
            for (int i = 0; i < 8; i++) bn |= (uint64_t)q[1 + i] << (8 * i);
            if (bn > qn - 9) { chunk_pos_ = chunk_end_; return false; }
            out->op = q[0];
            out->body.assign(q + 9, q + 9 + bn);
            chunk_pos_ += 9 + (size_t)bn;
            return true;
        }
        if (!fp_) return false;
        uint8_t hdr[9];
        if (std::fread(hdr, 1, 9, fp_) != 9) return false;   // EOF / trailing magic
        uint64_t n = 0;
        for (int i = 0; i < 8; i++) n |= (uint64_t)hdr[1 + i] << (8 * i);
        if (hdr[0] == OP_FOOTER || n > kMaxRecord) return false;
        Record r;
        r.op = hdr[0];
        r.body.resize((size_t)n);
        if (n && std::fread(r.body.data(), 1, r.body.size(), fp_) != r.body.size())
            return false;

        if (r.op == OP_CHUNK) {
            // Chunk: start(8) end(8) uncompressed_size(8) crc(4)
            //        compression(str) records_len(8) records...
            const uint8_t* p = r.body.data();
            size_t         left = r.body.size();
            if (left < 28) return false;
            p += 24; left -= 24;   // start/end/uncompressed_size
            p += 4;  left -= 4;    // uncompressed_crc
            if (left < 4) return false;
            uint32_t clen = rd32(p);
            p += 4; left -= 4;
            if (clen > left) return false;
            const bool uncompressed = (clen == 0);
            p += clen; left -= clen;
            if (left < 8) return false;
            uint64_t rlen = rd64(p);
            p += 8; left -= 8;
            if (rlen > left || !uncompressed) return false;   // zstd/lz4: unsupported
            // Keep the chunk body; next() serves one sub-record at a time from
            // [rec_off, rec_off+rlen). Grab the offset before move invalidates p.
            const size_t rec_off = (size_t)(p - r.body.data());
            chunk_     = std::move(r.body);
            chunk_pos_ = rec_off;
            chunk_end_ = rec_off + (size_t)rlen;
            continue;   // serve sub-records from chunk_
        }
        *out = std::move(r);
        return true;
    }
}

// ---- foxglove message codecs ------------------------------------------------------

void enc_compressed_video(std::vector<uint8_t>& out, uint64_t ts_ns,
                          const char* frame_id, const char* format,
                          const uint8_t* data, size_t len) {
    out.clear();
    out.reserve(len + 64);
    pb_f_timestamp(out, 1, ts_ns);
    pb_f_str(out, 2, frame_id);
    pb_f_str(out, 4, format);      // format before data, like the device writer
    pb_f_bytes(out, 3, data, len);
}

void enc_raw_image(std::vector<uint8_t>& out, uint64_t ts_ns,
                   const char* frame_id, uint32_t width, uint32_t height,
                   const char* encoding, uint32_t step,
                   const uint8_t* data, size_t len) {
    out.clear();
    out.reserve(len + 64);
    pb_f_timestamp(out, 1, ts_ns);
    pb_f_str(out, 2, frame_id);
    pb_f_varint(out, 3, width);
    pb_f_varint(out, 4, height);
    pb_f_str(out, 5, encoding);
    pb_f_varint(out, 6, step);
    pb_f_bytes(out, 7, data, len);
}

void enc_pose_in_frame(std::vector<uint8_t>& out, uint64_t ts_ns,
                       const char* frame_id, double x, double y, double z) {
    out.clear();
    std::vector<uint8_t> pos, ori, pose;
    if (x != 0.0) pb_f_double(pos, 1, x);
    if (y != 0.0) pb_f_double(pos, 2, y);
    if (z != 0.0) pb_f_double(pos, 3, z);
    pb_f_double(ori, 4, 1.0);      // identity quaternion (w only, like the device)
    pb_f_msg(pose, 1, pos);
    pb_f_msg(pose, 2, ori);
    pb_f_timestamp(out, 1, ts_ns);
    pb_f_str(out, 2, frame_id);
    pb_f_msg(out, 3, pose);
}

bool dec_compressed_video(const uint8_t* p, size_t n, VideoMsg* out) {
    *out = VideoMsg{};
    PbView v{p, n};
    uint32_t f, wt;
    uint64_t s;
    PbView sub;
    while (v.n) {
        if (!pb_next(v, &f, &wt, &s, &sub)) return false;
        if (f == 1 && wt == WT_LEN) out->ts_ns = dec_timestamp(sub);
        if (f == 3 && wt == WT_LEN) { out->data = sub.p; out->size = sub.n; }
        if (f == 4 && wt == WT_LEN) {
            size_t c = sub.n < sizeof out->format - 1 ? sub.n : sizeof out->format - 1;
            std::memcpy(out->format, sub.p, c);
        }
    }
    return out->data != nullptr;
}

bool dec_raw_image(const uint8_t* p, size_t n, VideoMsg* out) {
    *out = VideoMsg{};
    PbView v{p, n};
    uint32_t f, wt;
    uint64_t s;
    PbView sub;
    while (v.n) {
        if (!pb_next(v, &f, &wt, &s, &sub)) return false;
        if (f == 1 && wt == WT_LEN)    out->ts_ns  = dec_timestamp(sub);
        if (f == 3 && wt == WT_VARINT) out->width  = (uint32_t)s;
        if (f == 4 && wt == WT_VARINT) out->height = (uint32_t)s;
        if (f == 5 && wt == WT_LEN) {
            size_t c = sub.n < sizeof out->format - 1 ? sub.n : sizeof out->format - 1;
            std::memcpy(out->format, sub.p, c);
        }
        if (f == 7 && wt == WT_LEN) { out->data = sub.p; out->size = sub.n; }
    }
    return out->data != nullptr;
}

bool dec_pose_in_frame(const uint8_t* p, size_t n, PoseMsg* out) {
    *out = PoseMsg{};
    PbView v{p, n};
    uint32_t f, wt;
    uint64_t s;
    PbView sub;
    bool have_pose = false;
    while (v.n) {
        if (!pb_next(v, &f, &wt, &s, &sub)) return false;
        if (f == 1 && wt == WT_LEN) out->ts_ns = dec_timestamp(sub);
        if (f == 3 && wt == WT_LEN) {
            PbView pv = sub;
            uint32_t pf, pwt;
            uint64_t ps;
            PbView psub;
            while (pv.n) {
                if (!pb_next(pv, &pf, &pwt, &ps, &psub)) return false;
                if (pf == 1 && pwt == WT_LEN) {   // position
                    have_pose = true;
                    PbView xv = psub;
                    uint32_t xf, xwt;
                    uint64_t xs;
                    PbView xsub;
                    while (xv.n) {
                        if (!pb_next(xv, &xf, &xwt, &xs, &xsub)) return false;
                        if (xwt == WT_FIXED64) {
                            double d;
                            std::memcpy(&d, &xs, 8);
                            if (xf == 1) out->x = d;
                            if (xf == 2) out->y = d;
                            if (xf == 3) out->z = d;
                        }
                    }
                }
            }
        }
    }
    return have_pose;
}

}  // namespace mcap
}  // namespace internal
}  // namespace ef
