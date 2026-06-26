// ef-view — live viewer for the device's MCAP stream: decoded H.265 video on the
// left, a 3D device-orientation gizmo on the right (X=red, Y=green, Z=blue axes
// rotated by a quaternion integrated from the IMU).
//
//   ef-view [device_index]
//
// It opens the device, streams ep3 in-process via Device::stream_mcap(sink,...),
// parses the MCAP records as they arrive, software-decodes the H.265 video with
// libavcodec (no GPU), and renders with SDL2. Per §8 the USB reader runs on its
// own thread at line rate; the decode/view path may drop (it resyncs to the next
// keyframe when behind) but never stalls the reader. A terminal end (producer
// restart / drop) halts the view and waits for the user — no auto-resume (§7).
//
// Orientation = complementary filter (gyro integration α=0.98 + accel tilt
// correction), ported from sync_visualize.py. accel/gyro arrive as foxglove
// PoseInFrame messages (pose.position = x/y/z); we have no magnetometer, so yaw
// is gyro-only and drifts.
//
// Keys:  SPACE = stop / restart   ·   Q / Esc = quit
#include <ef/device.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <SDL2/SDL.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

using namespace ef;

static const double PI = std::acos(-1.0);

// ---- little-endian readers --------------------------------------------------
static uint16_t le16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t le32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static uint64_t le64(const uint8_t* p) {
    uint64_t v = 0; for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i); return v;
}

// ---- minimal protobuf field scans ------------------------------------------
static bool pb_varint(const uint8_t* b, size_t n, size_t* i, uint64_t* out) {
    uint64_t v = 0; int sh = 0;
    while (*i < n) {
        uint8_t c = b[(*i)++]; v |= (uint64_t)(c & 0x7f) << sh; sh += 7;
        if (!(c & 0x80)) { *out = v; return true; }
        if (sh > 63) return false;
    }
    return false;
}
static bool pb_find_len(const uint8_t* b, size_t n, int want, const uint8_t** data, size_t* len) {
    size_t i = 0;
    while (i < n) {
        uint64_t tag; if (!pb_varint(b, n, &i, &tag)) return false;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (wt == 2) {
            uint64_t l; if (!pb_varint(b, n, &i, &l)) return false;
            if (i + l > n) return false;
            if (fn == want) { *data = b + i; *len = (size_t)l; return true; }
            i += l;
        } else if (wt == 0) { uint64_t v; if (!pb_varint(b, n, &i, &v)) return false; }
        else if (wt == 1) { i += 8; } else if (wt == 5) { i += 4; } else return false;
    }
    return false;
}
static bool pb_find_varint(const uint8_t* b, size_t n, int want, uint64_t* out) {
    size_t i = 0;
    while (i < n) {
        uint64_t tag; if (!pb_varint(b, n, &i, &tag)) return false;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (wt == 0) { uint64_t v; if (!pb_varint(b, n, &i, &v)) return false; if (fn == want) { *out = v; return true; } }
        else if (wt == 2) { uint64_t l; if (!pb_varint(b, n, &i, &l)) return false; i += (size_t)l; }
        else if (wt == 1) { i += 8; } else if (wt == 5) { i += 4; } else return false;
    }
    return false;
}
static double pb_double(const uint8_t* b, size_t n, int want) {
    size_t i = 0;
    while (i < n) {
        uint64_t tag; if (!pb_varint(b, n, &i, &tag)) break;
        int fn = (int)(tag >> 3), wt = (int)(tag & 7);
        if (wt == 1) { if (i + 8 > n) break; if (fn == want) { double d; std::memcpy(&d, b + i, 8); return d; } i += 8; }
        else if (wt == 2) { uint64_t l; if (!pb_varint(b, n, &i, &l)) break; i += (size_t)l; }
        else if (wt == 0) { uint64_t v; if (!pb_varint(b, n, &i, &v)) break; }
        else if (wt == 5) { i += 4; } else break;
    }
    return 0.0;
}

// ---- quaternion math (ported from sync_visualize.py) ------------------------
struct Quat { double w = 1, x = 0, y = 0, z = 0; };
static Quat q_mul(const Quat& a, const Quat& b) {
    return { a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
             a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
             a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
             a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w };
}
static Quat q_norm(Quat q) {
    double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (n < 1e-12) return { 1, 0, 0, 0 };
    return { q.w/n, q.x/n, q.y/n, q.z/n };
}
static void q_rotate(const Quat& q, double vx, double vy, double vz, double out[3]) {
    double w = q.w, x = q.x, y = q.y, z = q.z;
    out[0] = (1-2*(y*y+z*z))*vx + 2*(x*y - w*z)*vy + 2*(x*z + w*y)*vz;
    out[1] = 2*(x*y + w*z)*vx + (1-2*(x*x+z*z))*vy + 2*(y*z - w*x)*vz;
    out[2] = 2*(x*z - w*y)*vx + 2*(y*z + w*x)*vy + (1-2*(x*x+y*y))*vz;
}
static Quat q_from_accel(double ax, double ay, double az) {
    double n = std::sqrt(ax*ax + ay*ay + az*az);
    if (n < 1e-6) return { 1, 0, 0, 0 };
    ax /= n; ay /= n; az /= n;
    double roll = std::atan2(ay, az);
    double pitch = std::atan2(-ax, std::sqrt(ay*ay + az*az));
    double cr = std::cos(roll/2), sr = std::sin(roll/2);
    double cp = std::cos(pitch/2), sp = std::sin(pitch/2);
    return q_norm({ cr*cp, sr*cp, cr*sp, -sr*sp });
}
static Quat integrate_gyro(const Quat& q, double wx, double wy, double wz, double dt) {
    Quat omega{ 0, wx*dt, wy*dt, wz*dt };
    Quat dq = q_mul(q, omega);
    return q_norm({ q.w + 0.5*dq.w, q.x + 0.5*dq.x, q.y + 0.5*dq.y, q.z + 0.5*dq.z });
}
static Quat q_slerp(Quat a, Quat b, double t) {
    double dot = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
    if (dot < 0) { b = { -b.w, -b.x, -b.y, -b.z }; dot = -dot; }
    if (dot > 0.9995)
        return q_norm({ a.w + t*(b.w-a.w), a.x + t*(b.x-a.x), a.y + t*(b.y-a.y), a.z + t*(b.z-a.z) });
    double th0 = std::acos(std::min(1.0, std::max(-1.0, dot))), th = th0 * t, s0 = std::sin(th0);
    double c0 = std::sin(th0 - th) / s0, c1 = std::sin(th) / s0;
    return { c0*a.w + c1*b.w, c0*a.x + c1*b.x, c0*a.y + c1*b.y, c0*a.z + c1*b.z };
}

// ---- shared state -----------------------------------------------------------
namespace {
constexpr double kAlpha  = 0.98;   // gyro trust; (1-alpha) = accel correction
constexpr size_t kVidMax = 16;     // video AU backlog before resync to a keyframe

// Display tuning (set in main from env). The synthetic IMU is a tiny ±0.05 rad/s
// sine that integrates to <2deg of orientation — invisible. EF_VIEW_GAIN>1
// amplifies the gyro and relaxes the accel pull so the wave is visible during
// the synthetic phase. Default 1.0 = faithful (for real IMU later).
double g_gain  = 1.0;
double g_alpha = kAlpha;

std::mutex g_mu;
std::deque<std::vector<uint8_t>> g_vid;   // pending video access units (codec per g_stream_codec)

// Codec of the live stream, auto-detected from the MCAP CompressedVideo.format
// field (h264/h265). The device's codec is host-selectable (ef-config), so the
// viewer must follow it rather than assume HEVC. AV_CODEC_ID_NONE until the
// first video message reveals it; the decoder is (re)built when this changes.
std::atomic<int> g_stream_codec{AV_CODEC_ID_NONE};

// Raw NV12 path (codec=raw): the device sends foxglove.RawImage on a separate
// channel (no codec, self-describing dims). When set, frames in g_vid are whole
// NV12 buffers (not coded AUs) and bypass the decoder — swscale NV12->RGB direct.
std::atomic<bool> g_stream_raw{false};
std::atomic<int>  g_raw_w{0}, g_raw_h{0};

std::mutex g_orient_mu;
Quat       g_q;                            // current orientation estimate

std::atomic<bool> g_quit{false};
std::atomic<bool> g_pause{false};
std::atomic<long> g_au_seen{0}, g_frames{0}, g_acc_n{0}, g_gyr_n{0}, g_vmatch{0};
double g_lax = 0, g_lay = 0, g_laz = 0, g_lgx = 0, g_lgy = 0, g_lgz = 0;  // last raw (under g_orient_mu)

std::mutex  g_status_mu; std::string g_status = "starting";
void set_status(const std::string& s) { std::lock_guard<std::mutex> l(g_status_mu); g_status = s; }
std::string get_status() { std::lock_guard<std::mutex> l(g_status_mu); return g_status; }

// Scan annex-B NAL units for a keyframe. Codec-specific NAL header + types:
//   H.265: 2-byte header, type=(b>>1)&0x3f; IDR/CRA/BLA = 16..21, VPS = 32.
//   H.264: 1-byte header, type=b&0x1f;       IDR = 5, SPS = 7.
bool au_is_keyframe(const uint8_t* d, size_t n, int codec_id) {
    for (size_t i = 0; i + 4 < n; i++) {
        if (d[i] == 0 && d[i+1] == 0 && ((d[i+2] == 1) || (d[i+2] == 0 && d[i+3] == 1))) {
            size_t h = (d[i+2] == 1) ? i + 3 : i + 4;
            if (h >= n) continue;
            if (codec_id == AV_CODEC_ID_H264) {
                int t = d[h] & 0x1f; if (t == 5 || t == 7) return true;
            } else {  // HEVC (default)
                int t = (d[h] >> 1) & 0x3f; if ((t >= 16 && t <= 21) || t == 32) return true;
            }
        }
    }
    return false;
}
}  // namespace

// ---- incremental MCAP record parser ----------------------------------------
class McapParse {
public:
    // Start (and restart) in the discarding state: a fresh stream joins mid-GOP, so
    // the first AUs are P-frames whose VPS/SPS/PPS we haven't seen -> the HEVC decoder
    // spews "PPS id out of range" until the first IDR. Gate decoding on the first
    // keyframe instead of feeding the orphan P-frames.
    void reset() { buf_.clear(); magic_ = false; chan_.clear(); have_g_ = false; discarding_ = true;
                   g_stream_raw.store(false); }  /* re-detect raw vs coded on each (re)connect */
    void feed(const uint8_t* d, size_t n) {
        buf_.insert(buf_.end(), d, d + n);
        size_t pos = 0;
        if (!magic_) { if (buf_.size() < 8) return; pos = 8; magic_ = true; }
        while (buf_.size() - pos >= 9) {
            uint8_t op = buf_[pos]; uint64_t rlen = le64(&buf_[pos + 1]);
            if (buf_.size() - pos - 9 < rlen) break;
            handle(op, &buf_[pos + 9], (size_t)rlen);
            pos += 9 + (size_t)rlen;
        }
        if (pos) buf_.erase(buf_.begin(), buf_.begin() + (long)pos);
    }
private:
    void handle(uint8_t op, const uint8_t* b, size_t len) {
        if (op == 0x04) {                                       // Channel
            if (len < 8) return;
            uint16_t id = le16(b); uint32_t tlen = le32(b + 4);
            if (8 + tlen > len) return;
            chan_[id] = std::string((const char*)b + 8, tlen);
            fprintf(stderr, "[ef-view chan] id=%u topic=\"%s\"\n", id, chan_[id].c_str());
        } else if (op == 0x05) {                                // Message
            if (len < 22) return;
            uint16_t cid = le16(b);
            uint64_t log_ns = le64(b + 6);                      // channel_id(2)+sequence(4)
            auto it = chan_.find(cid); if (it == chan_.end()) return;
            route(it->second, b + 22, len - 22, log_ns);
        }
    }
    void route(const std::string& topic, const uint8_t* data, size_t dlen, uint64_t log_ns) {
        if (ends_with(topic, "image_raw")) {                    // raw NV12 (foxglove.RawImage)
            uint64_t w = 0, h = 0;                              // width=3, height=4
            pb_find_varint(data, dlen, 3, &w);
            pb_find_varint(data, dlen, 4, &h);
            const uint8_t* px; size_t pl;                       // RawImage.data = field 7
            if (w > 0 && h > 0 && pb_find_len(data, dlen, 7, &px, &pl) && pl > 0) {
                g_au_seen.fetch_add(1);
                g_stream_raw.store(true);
                g_raw_w.store((int)w);
                g_raw_h.store((int)h);
                std::vector<uint8_t> v(px, px + pl);
                std::lock_guard<std::mutex> l(g_mu);
                // every raw frame is independent (no GOP) — no keyframe gating;
                // just bound the backlog by dropping the oldest if behind.
                if (g_vid.size() >= kVidMax) g_vid.pop_front();
                g_vid.push_back(std::move(v));
            }
            return;
        }
        if (ends_with(topic, "/image")) {                       // video (lenient match)
            g_vmatch.fetch_add(1);
            const uint8_t* fmt; size_t fl;                      // CompressedVideo.format = field 4
            if (pb_find_len(data, dlen, 4, &fmt, &fl) && fl > 0) {
                std::string f((const char*)fmt, fl);
                int id = (f == "h264" || f == "avc")  ? AV_CODEC_ID_H264 :
                         (f == "h265" || f == "hevc") ? AV_CODEC_ID_HEVC : AV_CODEC_ID_NONE;
                if (id != AV_CODEC_ID_NONE && id != g_stream_codec.load())
                    g_stream_codec.store(id);
            }
            const uint8_t* au; size_t al;                       // CompressedVideo.data = field 3
            if (pb_find_len(data, dlen, 3, &au, &al) && al > 0) {
                g_au_seen.fetch_add(1);
                std::vector<uint8_t> v(au, au + al);
                int cid = g_stream_codec.load();
                bool kf = au_is_keyframe(v.data(), v.size(), cid);
                std::lock_guard<std::mutex> l(g_mu);
                // Behind? Drop CONTIGUOUSLY until the next keyframe (a clean IDR
                // cut) — never mid-GOP, which orphans references and yields
                // "Could not find ref with POC". Resume only at a keyframe.
                if (discarding_) {
                    if (!kf) return;
                    discarding_ = false;
                    g_vid.clear();
                } else if (g_vid.size() >= kVidMax) {
                    discarding_ = true;
                    if (!kf) return;
                    g_vid.clear();
                }
                g_vid.push_back(std::move(v));
            }
        } else if (ends_with(topic, "/accel/sample") || ends_with(topic, "/gyro/sample")) {
            const uint8_t* pose; size_t pl;                     // PoseInFrame.pose = field 3
            if (!pb_find_len(data, dlen, 3, &pose, &pl)) return;
            const uint8_t* pos; size_t nl;                      // Pose.position = field 1
            if (!pb_find_len(pose, pl, 1, &pos, &nl)) return;
            double vx = pb_double(pos, nl, 1), vy = pb_double(pos, nl, 2), vz = pb_double(pos, nl, 3);
            if (ends_with(topic, "/accel/sample")) {
                ax_ = vx; ay_ = vy; az_ = vz; have_a_ = true;
                { std::lock_guard<std::mutex> l(g_orient_mu); g_lax = vx; g_lay = vy; g_laz = vz; }
                g_acc_n.fetch_add(1);
            } else {
                { std::lock_guard<std::mutex> l(g_orient_mu); g_lgx = vx; g_lgy = vy; g_lgz = vz; }
                g_gyr_n.fetch_add(1);
                update_gyro(vx, vy, vz, log_ns);   // (locks g_orient_mu itself — not held here)
            }
        }
    }
    // complementary filter step on each gyro sample (uses latest accel for tilt)
    void update_gyro(double gx, double gy, double gz, uint64_t log_ns) {
        double t = (double)log_ns * 1e-9;
        if (have_g_) {
            double dt = t - tlast_;
            if (dt > 0 && dt < 0.5) {
                Quat qcur; { std::lock_guard<std::mutex> l(g_orient_mu); qcur = g_q; }
                Quat qg = integrate_gyro(qcur, gx * g_gain, gy * g_gain, gz * g_gain, dt);
                Quat out = qg;
                if (have_a_) out = q_norm(q_slerp(qg, q_from_accel(ax_, ay_, az_), 1.0 - g_alpha));
                std::lock_guard<std::mutex> l(g_orient_mu); g_q = out;
            }
        }
        tlast_ = t; have_g_ = true;
    }
    static bool ends_with(const std::string& s, const char* suf) {
        size_t n = std::strlen(suf);
        return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
    }
    std::vector<uint8_t> buf_; bool magic_ = false;
    std::unordered_map<uint16_t, std::string> chan_;
    double ax_ = 0, ay_ = 0, az_ = 0; bool have_a_ = false;
    double tlast_ = 0; bool have_g_ = false;
    bool discarding_ = true;    // gate decode until first keyframe (startup join + behind-resync)
};

// ---- tiny 5x7 bitmap font (only the glyphs the gizmo needs) -----------------
static const uint8_t* glyph(char c) {
    static const uint8_t SP[7]={0,0,0,0,0,0,0},
      D0[7]={0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, D1[7]={0x04,0x0C,0x04,0x04,0x04,0x04,0x0E},
      D2[7]={0x0E,0x11,0x01,0x02,0x04,0x08,0x1F}, D3[7]={0x1F,0x02,0x04,0x02,0x01,0x11,0x0E},
      D4[7]={0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, D5[7]={0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E},
      D6[7]={0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, D7[7]={0x1F,0x01,0x02,0x04,0x08,0x08,0x08},
      D8[7]={0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, D9[7]={0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C},
      DOT[7]={0,0,0,0,0,0x06,0x06}, PLUS[7]={0,0x04,0x04,0x1F,0x04,0x04,0}, MINUS[7]={0,0,0,0x1F,0,0,0},
      EQ[7]={0,0,0x1F,0,0x1F,0,0},
      GX[7]={0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, GY[7]={0x11,0x11,0x0A,0x04,0x04,0x04,0x04},
      GZ[7]={0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, GW[7]={0x11,0x11,0x11,0x15,0x15,0x1B,0x11};
    switch (c) {
        case ' ': return SP; case '0': return D0; case '1': return D1; case '2': return D2;
        case '3': return D3; case '4': return D4; case '5': return D5; case '6': return D6;
        case '7': return D7; case '8': return D8; case '9': return D9; case '.': return DOT;
        case '+': return PLUS; case '-': return MINUS; case '=': return EQ;
        case 'x': case 'X': return GX; case 'y': case 'Y': return GY;
        case 'z': case 'Z': return GZ; case 'w': case 'W': return GW;
        default: return nullptr;
    }
}
static void draw_text(SDL_Renderer* r, int x, int y, int s, SDL_Color c, const char* str) {
    SDL_SetRenderDrawColor(r, c.r, c.g, c.b, 255);
    for (const char* p = str; *p; p++) {
        const uint8_t* g = glyph(*p);
        if (g) for (int row = 0; row < 7; row++)
            for (int col = 0; col < 5; col++)
                if (g[row] & (1 << (4 - col))) {
                    SDL_Rect px{ x + col*s, y + row*s, s, s }; SDL_RenderFillRect(r, &px);
                }
        x += 6 * s;
    }
}

// ---- 3D orientation gizmo ---------------------------------------------------
static void view_xform(double vx, double vy, double vz, double out[3]) {
    static const double el = 28*PI/180, az = -40*PI/180;
    static const double ca = std::cos(az), sa = std::sin(az), ce = std::cos(el), se = std::sin(el);
    double x1 = ca*vx + sa*vz, y1 = vy, z1 = -sa*vx + ca*vz;   // Ry(az)
    out[0] = x1; out[1] = ce*y1 - se*z1; out[2] = se*y1 + ce*z1;  // Rx(el)
}
static SDL_Point project(double vx, double vy, double vz, int cx, int cy, double scale) {
    double o[3]; view_xform(vx, vy, vz, o);
    return SDL_Point{ cx + (int)(o[0]*scale), cy - (int)(o[1]*scale) };
}
static double depth(const Quat& q, double vx, double vy, double vz) {
    double w[3]; q_rotate(q, vx, vy, vz, w); double o[3]; view_xform(w[0], w[1], w[2], o); return o[2];
}

static void draw_orientation(SDL_Renderer* r, const SDL_Rect& a, const Quat& q) {
    SDL_SetRenderDrawColor(r, 35, 35, 45, 255); SDL_RenderFillRect(r, &a);
    int cx = a.x + a.w/2, cy = a.y + a.h/2;
    double scale = std::min(a.w, a.h) * 0.34;
    const SDL_Color GREY{100,100,115,255}, WHITE{220,220,220,255}, CNEG{80,80,80,255};
    const SDL_Color CX{220,60,60,255}, CY{60,200,60,255}, CZ{60,140,255,255};

    // reference equator + cross (in view space, not rotated)
    SDL_SetRenderDrawColor(r, GREY.r, GREY.g, GREY.b, 255);
    SDL_Point prev{0,0}; bool first = true;
    for (double th = 0; th <= 2*PI + 0.01; th += 2*PI/60) {
        SDL_Point p = project(std::cos(th)*0.72, 0, std::sin(th)*0.72, cx, cy, scale);
        if (!first) SDL_RenderDrawLine(r, prev.x, prev.y, p.x, p.y);
        prev = p; first = false;
    }
    SDL_Point o0 = project(0,0,0, cx, cy, scale);
    for (double* v : (double[][3]){{0.75,0,0},{-0.75,0,0},{0,0,0.75},{0,0,-0.75}}) {
        SDL_Point e = project(v[0], v[1], v[2], cx, cy, scale);
        SDL_RenderDrawLine(r, o0.x, o0.y, e.x, e.y);
    }

    // device axes, depth-sorted (draw far first)
    struct Axis { double px,py,pz, nx,ny,nz; SDL_Color c; char lbl; };
    Axis ax[3] = {
        {1,0,0,-1,0,0, CX,'X'}, {0,1,0,0,-1,0, CY,'Y'}, {0,0,1,0,0,-1, CZ,'Z'} };
    int ord[3] = {0,1,2};
    std::sort(ord, ord+3, [&](int i, int j){ return depth(q,ax[i].px,ax[i].py,ax[i].pz) < depth(q,ax[j].px,ax[j].py,ax[j].pz); });
    for (int k = 0; k < 3; k++) {
        Axis& A = ax[ord[k]];
        double pp[3], pn[3]; q_rotate(q, A.px,A.py,A.pz, pp); q_rotate(q, A.nx,A.ny,A.nz, pn);
        SDL_Point tip  = project(pp[0]*0.82, pp[1]*0.82, pp[2]*0.82, cx, cy, scale);
        SDL_Point tail = project(pn[0]*0.45, pn[1]*0.45, pn[2]*0.45, cx, cy, scale);
        SDL_SetRenderDrawColor(r, CNEG.r, CNEG.g, CNEG.b, 255);
        SDL_RenderDrawLine(r, tail.x, tail.y, o0.x, o0.y);
        SDL_SetRenderDrawColor(r, A.c.r, A.c.g, A.c.b, 255);
        SDL_RenderDrawLine(r, o0.x, o0.y, tip.x, tip.y);
        SDL_RenderDrawLine(r, o0.x, o0.y-1, tip.x, tip.y-1);          // 2px-ish
        SDL_Rect dot{ tip.x-3, tip.y-3, 6, 6 }; SDL_RenderFillRect(r, &dot);
        char ls[2] = { A.lbl, 0 };
        draw_text(r, tip.x + (tip.x>=o0.x?6:-14), tip.y - 6, 2, A.c, ls);
    }
    SDL_SetRenderDrawColor(r, WHITE.r, WHITE.g, WHITE.b, 255);
    SDL_Rect c{ o0.x-3, o0.y-3, 6, 6 }; SDL_RenderFillRect(r, &c);

    // quaternion readout (bottom-left)
    char buf[40];
    const struct { const char* l; double v; SDL_Color c; } rows[4] = {
        {"w", q.w, WHITE}, {"x", q.x, CX}, {"y", q.y, CY}, {"z", q.z, CZ} };
    for (int i = 0; i < 4; i++) {
        std::snprintf(buf, sizeof buf, "%s=%+.3f", rows[i].l, rows[i].v);
        draw_text(r, a.x + 8, a.y + a.h - 70 + i*16, 2, rows[i].c, buf);
    }
}

int main(int argc, char** argv) {
    int dev_index = (argc > 1) ? atoi(argv[1]) : 0;
    Device device;
    InitParameters init; init.verbose = 1; init.device_id = dev_index;
    ERROR_CODE st = device.open(init);
    if (st != ERROR_CODE::SUCCESS) { fprintf(stderr, "Failed to open device: %s\n", to_string(st)); return 1; }

    if (SDL_Init(SDL_INIT_VIDEO) != 0) { fprintf(stderr, "SDL_Init: %s\n", SDL_GetError()); device.close(); return 1; }
    const int W = 1280, H = 720, VIDW = 896;
    SDL_Window*   win = SDL_CreateWindow("ef-view", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, W, H, SDL_WINDOW_SHOWN);
    SDL_Renderer* ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
    if (!ren) ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_SOFTWARE);
    SDL_Texture* tex = nullptr; int tex_w = 0, tex_h = 0;

    // Frame-threading parallelizes across frames -> throughput, but adds
    // ~(threads-1) frames of latency. The CPU has headroom, so default LOW (3)
    // for low latency; EF_VIEW_THREADS tunes it. EF_VIEW_SLICE=1 switches to
    // slice threading (low latency, but only parallelizes if the stream is
    // multi-slice — single-slice H.265 then decodes ~single-threaded).
    int nthreads = 3;
    if (const char* t = getenv("EF_VIEW_THREADS")) { nthreads = atoi(t); if (nthreads < 1) nthreads = 1; }
    int slice = getenv("EF_VIEW_SLICE") ? atoi(getenv("EF_VIEW_SLICE")) : 0;

    // The codec is host-selectable, so the decoder is (re)built to match the
    // stream's detected codec (g_stream_codec). Default HEVC until the first
    // video message reveals otherwise; rebuilt in the loop when it changes.
    auto make_dec = [&](int codec_id) -> AVCodecContext* {
        const AVCodec* c = avcodec_find_decoder((AVCodecID)codec_id);
        if (!c) { fprintf(stderr, "ef-view: no decoder for codec id %d\n", codec_id); return nullptr; }
        AVCodecContext* d = avcodec_alloc_context3(c);
        d->thread_count = nthreads;
        d->thread_type  = slice ? FF_THREAD_SLICE : FF_THREAD_FRAME;
        avcodec_open2(d, c, nullptr);
        fprintf(stderr, "ef-view: decoder=%s threads=%d type=%s\n",
                c->name, nthreads, slice ? "SLICE" : "FRAME");
        return d;
    };
    // EF_VIEW_CODEC=h264|h265 forces the decoder (skips/overrides auto-detect);
    // otherwise default HEVC and let the stream's format field drive it.
    int cur_codec = AV_CODEC_ID_HEVC;
    if (const char* fc = getenv("EF_VIEW_CODEC")) {
        std::string f = fc;
        int id = (f == "h264" || f == "avc")  ? AV_CODEC_ID_H264 :
                 (f == "h265" || f == "hevc") ? AV_CODEC_ID_HEVC : AV_CODEC_ID_NONE;
        if (id != AV_CODEC_ID_NONE) { cur_codec = id; g_stream_codec.store(id); }
        else fprintf(stderr, "ef-view: ignoring EF_VIEW_CODEC=%s (use h264|h265)\n", fc);
    }
    AVCodecContext* dec = make_dec(cur_codec);
    AVPacket* pkt = av_packet_alloc(); AVFrame* frm = av_frame_alloc();
    SwsContext* sws = nullptr; int sws_w = 0, sws_h = 0, sws_fmt = -1;
    uint8_t* rgb[4] = {}; int rstr[4] = {};   // RGB24 dst (swscale handles range/matrix)

    // Scale one source frame (decoded YUV, or raw NV12) -> RGB24 -> the SDL
    // texture. Shared by the coded (avcodec) and raw (NV12 passthrough) paths.
    auto show_frame = [&](const uint8_t* const src[4], const int srcstride[4],
                          int pixfmt, int w, int h) {
        if (w <= 0 || h <= 0) return;
        if (!sws || sws_w != w || sws_h != h || sws_fmt != pixfmt) {
            if (sws) sws_freeContext(sws);
            if (rgb[0]) av_freep(&rgb[0]);
            sws = sws_getContext(w, h, (AVPixelFormat)pixfmt, w, h, AV_PIX_FMT_RGB24,
                                 SWS_BILINEAR, nullptr, nullptr, nullptr);
            av_image_alloc(rgb, rstr, w, h, AV_PIX_FMT_RGB24, 1);
            sws_w = w; sws_h = h; sws_fmt = pixfmt;
        }
        sws_scale(sws, src, srcstride, 0, h, rgb, rstr);
        if (!tex || tex_w != w || tex_h != h) {
            if (tex) SDL_DestroyTexture(tex);
            tex = SDL_CreateTexture(ren, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, w, h);
            tex_w = w; tex_h = h;
        }
        SDL_UpdateTexture(tex, nullptr, rgb[0], rstr[0]);
        g_frames.fetch_add(1);
    };

    if (const char* gs = getenv("EF_VIEW_GAIN")) {
        g_gain = atof(gs);
        if (g_gain > 1.0) g_alpha = 0.9995;   // gyro-dominated so the (amplified) sine shows
    }
    fprintf(stderr, "ef-view: SPACE=stop/restart  Q/Esc=quit   axes X=red Y=green Z=blue"
                    "  [gyro gain=%.1f alpha=%.4f]\n", g_gain, g_alpha);

    std::thread reader([&] {
        McapParse parser;
        while (!g_quit.load()) {
            if (g_pause.load()) { SDL_Delay(40); continue; }
            parser.reset();
            { std::lock_guard<std::mutex> l(g_mu); g_vid.clear(); }
            set_status("streaming");
            StreamResult r = device.stream_mcap(
                [&](const uint8_t* d, size_t n) { parser.feed(d, n); },
                [] { return g_quit.load() || g_pause.load(); });
            if (r.end != STREAM_END::STOPPED) {
                g_pause.store(true);
                set_status(std::string("HALTED: ") + to_string(r.end) + " — SPACE to restart");
                fprintf(stderr, "[stream] %s (%s)\n", to_string(r.end), r.detail.c_str());
            } else if (g_pause.load()) set_status("stopped — SPACE to restart");
        }
    });

    bool running = true; std::string last_title; Uint32 last_log = SDL_GetTicks();
    long prev_au = 0, prev_dec = 0;
    while (running) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) running = false;
            else if (e.type == SDL_KEYDOWN) {
                SDL_Keycode k = e.key.keysym.sym;
                if (k == SDLK_q || k == SDLK_ESCAPE) running = false;
                else if (k == SDLK_SPACE) { bool p = !g_pause.load(); g_pause.store(p); if (!p && dec) avcodec_flush_buffers(dec); }
            }
        }

        // Follow a host-driven codec switch: rebuild the decoder when the stream
        // reveals a codec different from what `dec` is decoding.
        int want_codec = g_stream_codec.load();
        if (want_codec != AV_CODEC_ID_NONE && want_codec != cur_codec) {
            AVCodecContext* nd = make_dec(want_codec);
            if (nd) { avcodec_free_context(&dec); dec = nd; cur_codec = want_codec; }
        }

        const bool raw = g_stream_raw.load();
        for (int budget = 0; budget < 8; budget++) {
            std::vector<uint8_t> buf;
            { std::lock_guard<std::mutex> l(g_mu); if (g_vid.empty()) break; buf.swap(g_vid.front()); g_vid.pop_front(); }
            if (raw) {
                // whole NV12 frame: Y plane then interleaved UV (no decode).
                int w = g_raw_w.load(), h = g_raw_h.load();
                size_t need = (size_t)w * h * 3 / 2;
                if (w > 0 && h > 0 && buf.size() >= need) {
                    const uint8_t* src[4] = { buf.data(), buf.data() + (size_t)w * h, nullptr, nullptr };
                    int srcstr[4] = { w, w, 0, 0 };
                    show_frame(src, srcstr, AV_PIX_FMT_NV12, w, h);
                }
                continue;
            }
            if (!dec) break;
            pkt->data = buf.data(); pkt->size = (int)buf.size();
            if (avcodec_send_packet(dec, pkt) == 0) {
                while (avcodec_receive_frame(dec, frm) == 0) {
                    if (frm->width <= 0 || frm->height <= 0) continue;
                    show_frame((const uint8_t* const*)frm->data, frm->linesize,
                               frm->format, frm->width, frm->height);
                }
            }
        }

        SDL_SetRenderDrawColor(ren, 18, 18, 18, 255); SDL_RenderClear(ren);
        if (tex && tex_h > 0) {
            float ar = (float)tex_w / (float)tex_h;
            int dw = VIDW, dh = (int)(VIDW / ar);
            if (dh > H) { dh = H; dw = (int)(H * ar); }
            SDL_Rect dst{ (VIDW - dw)/2, (H - dh)/2, dw, dh };
            SDL_RenderCopy(ren, tex, nullptr, &dst);
        }
        Quat q; { std::lock_guard<std::mutex> l(g_orient_mu); q = g_q; }
        SDL_Rect panel{ VIDW, 0, W - VIDW, H };
        draw_orientation(ren, panel, q);
        SDL_SetRenderDrawColor(ren, 80, 80, 80, 255); SDL_RenderDrawLine(ren, VIDW, 0, VIDW, H);
        SDL_RenderPresent(ren);

        Uint32 now_ticks = SDL_GetTicks();
        if (now_ticks - last_log >= 1000) {
            double dt = (now_ticks - last_log) / 1000.0;
            long au = g_au_seen.load(), dc = g_frames.load();
            size_t qd; { std::lock_guard<std::mutex> l(g_mu); qd = g_vid.size(); }
            // au_fps = video frames ARRIVING (should be ~30); dec_fps = frames
            // DECODED; qdepth = host backlog (grows => decode behind => latency).
            fprintf(stderr,
                "[ef-view] in=%.0ffps dec=%.0ffps qdepth=%zu | au=%ld dec=%ld acc=%ld gyr=%ld\n",
                (au - prev_au) / dt, (dc - prev_dec) / dt, qd, au, dc,
                g_acc_n.load(), g_gyr_n.load());
            prev_au = au; prev_dec = dc; last_log = now_ticks;
        }
        std::string title = "ef-view — " + get_status();
        if (title != last_title) { SDL_SetWindowTitle(win, title.c_str()); last_title = title; }
        SDL_Delay(4);
    }

    g_quit.store(true); reader.join();
    if (sws) sws_freeContext(sws); if (rgb[0]) av_freep(&rgb[0]);
    av_frame_free(&frm); av_packet_free(&pkt); avcodec_free_context(&dec);
    if (tex) SDL_DestroyTexture(tex);
    SDL_DestroyRenderer(ren); SDL_DestroyWindow(win); SDL_Quit(); device.close();
    return 0;
}
