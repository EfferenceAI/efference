////////////////////////////////////////////////////////////////////////////////
//
// File:      mcap_replay_reader.cpp
// Purpose:   MCAP file-replay data source (internal): plays a recording back
//            through the same assembler the live transports feed.
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

#include "mcap_replay_reader.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>

#include "mcap.hpp"

namespace ef {
namespace internal {

namespace {

// ef_stream wire constants (must match stream_assembler.cpp).
constexpr uint8_t kMagic = 0xEF, kTypeVideo = 1, kTypeImu = 2;
constexpr uint8_t kFragStart = 0x01, kFragEnd = 0x02;
constexpr int     kVidHdr = 44, kImuHdr = 20, kImuSample = 40;

void wr16(uint8_t* p, uint16_t v) { p[0] = (uint8_t)v; p[1] = (uint8_t)(v >> 8); }
void wr32(uint8_t* p, uint32_t v) { wr16(p, (uint16_t)v); wr16(p + 2, (uint16_t)(v >> 16)); }
void wr64(uint8_t* p, uint64_t v) { wr32(p, (uint32_t)v); wr32(p + 4, (uint32_t)(v >> 32)); }
void wrf(uint8_t* p, float f)     { std::memcpy(p, &f, 4); }

// What role a channel plays in the replay, resolved from its topic.
enum class Role { OTHER, VIDEO, VIDEO_RAW, ACCEL, GYRO };

Role role_of(const std::string& topic) {
    if (topic == mcap::TOPIC_IMAGE)     return Role::VIDEO;
    if (topic == mcap::TOPIC_IMAGE_RAW) return Role::VIDEO_RAW;
    if (topic == mcap::TOPIC_ACCEL)     return Role::ACCEL;
    if (topic == mcap::TOPIC_GYRO)      return Role::GYRO;
    return Role::OTHER;
}

// Message record body: channel_id(2) sequence(4) log_time(8) publish_time(8) data.
struct MsgView {
    uint16_t       channel = 0;
    uint64_t       log_time = 0;
    const uint8_t* data = nullptr;
    size_t         size = 0;
};

bool msg_view(const std::vector<uint8_t>& body, MsgView* out) {
    if (body.size() < 22) return false;
    out->channel  = (uint16_t)(body[0] | body[1] << 8);
    uint64_t lt = 0;
    for (int i = 0; i < 8; i++) lt |= (uint64_t)body[6 + i] << (8 * i);
    out->log_time = lt;
    out->data     = body.data() + 22;
    out->size     = body.size() - 22;
    return true;
}

// Channel record body prefix: id(2) schema_id(2) topic(str).
bool channel_view(const std::vector<uint8_t>& body, uint16_t* id, std::string* topic) {
    if (body.size() < 8) return false;
    *id = (uint16_t)(body[0] | body[1] << 8);
    uint32_t tlen = (uint32_t)body[4] | (uint32_t)body[5] << 8 |
                    (uint32_t)body[6] << 16 | (uint32_t)body[7] << 24;
    if (8 + (size_t)tlen > body.size()) return false;
    topic->assign((const char*)body.data() + 8, tlen);
    return true;
}

}  // namespace

Status McapReplayReader::start(const std::string& path, int verbose) {
    if (running_.load()) return Status::ALREADY_OPEN;
    path_    = path;
    verbose_ = verbose;

    // Pre-scan: latch channel roles and the first video message's geometry so
    // Device::open() can validate/derive the session codec before playback.
    mcap::Reader r;
    if (!r.open(path)) {
        std::FILE* probe = std::fopen(path.c_str(), "rb");
        if (!probe) return Status::FILE_NOT_FOUND;
        std::fclose(probe);
        return Status::NO_STREAM;   // exists but is not an MCAP file
    }
    std::map<uint16_t, Role> roles;
    bool any_known = false;
    mcap::Reader::Record rec;
    while (r.next(&rec)) {
        if (rec.op == mcap::OP_CHANNEL) {
            uint16_t    id = 0;
            std::string topic;
            if (channel_view(rec.body, &id, &topic)) {
                roles[id] = role_of(topic);
                if (roles[id] != Role::OTHER) any_known = true;
            }
        } else if (rec.op == mcap::OP_MESSAGE && fmt_.empty()) {
            MsgView m;
            if (!msg_view(rec.body, &m)) continue;
            auto it = roles.find(m.channel);
            if (it == roles.end()) continue;
            mcap::VideoMsg vm;
            if (it->second == Role::VIDEO &&
                mcap::dec_compressed_video(m.data, m.size, &vm)) {
                fmt_ = vm.format;
                break;
            }
            if (it->second == Role::VIDEO_RAW &&
                mcap::dec_raw_image(m.data, m.size, &vm)) {
                fmt_ = vm.format[0] ? vm.format : "nv12";
                w_   = (int)vm.width;
                h_   = (int)vm.height;
                break;
            }
        }
    }
    r.close();
    if (!any_known) return Status::NO_STREAM;   // no channel this SDK understands

    stop_.store(false);
    running_.store(true);
    thread_ = std::thread([this] { run(); });
    return Status::SUCCESS;
}

void McapReplayReader::stop() {
    stop_.store(true);
    if (thread_.joinable()) thread_.join();
    running_.store(false);
    mark_ended();
}

void McapReplayReader::run() {
    mcap::Reader r;
    if (!r.open(path_)) {
        running_.store(false);
        mark_ended();
        return;
    }

    std::map<uint16_t, Role> roles;
    // Pacing: the first paced message pins file time t0 to wall time w0; every
    // later message is released at w0 + (log_time - t0).
    uint64_t t0 = 0;
    bool     have_t0 = false;
    auto     w0 = std::chrono::steady_clock::now();
    auto pace = [&](uint64_t lt) {
        if (!have_t0) { t0 = lt; w0 = std::chrono::steady_clock::now(); have_t0 = true; return; }
        if (lt <= t0) return;
        auto target = w0 + std::chrono::nanoseconds(lt - t0);
        while (!stop_.load()) {
            auto now = std::chrono::steady_clock::now();
            if (now >= target) break;
            auto left = target - now;
            std::this_thread::sleep_for(
                left < std::chrono::milliseconds(50) ? left
                                                     : std::chrono::milliseconds(50));
        }
    };

    // accel/gyro PoseInFrame streams re-joined into full ImuSamples: the device
    // recorder writes the accel then the gyro of each sample back to back.
    mcap::PoseMsg accel{};
    bool          have_accel = false;
    std::vector<uint8_t> pkt;

    mcap::Reader::Record rec;
    while (!stop_.load() && r.next(&rec)) {
        if (rec.op == mcap::OP_CHANNEL) {
            uint16_t    id = 0;
            std::string topic;
            if (channel_view(rec.body, &id, &topic)) roles[id] = role_of(topic);
            continue;
        }
        if (rec.op != mcap::OP_MESSAGE) continue;
        MsgView m;
        if (!msg_view(rec.body, &m)) continue;
        auto it = roles.find(m.channel);
        if (it == roles.end() || it->second == Role::OTHER) continue;

        if (it->second == Role::VIDEO || it->second == Role::VIDEO_RAW) {
            mcap::VideoMsg vm;
            bool ok = (it->second == Role::VIDEO)
                          ? mcap::dec_compressed_video(m.data, m.size, &vm)
                          : mcap::dec_raw_image(m.data, m.size, &vm);
            if (!ok || !vm.size) continue;
            pace(m.log_time);
            if (stop_.load()) break;
            // One whole frame per synthesized packet (START|END, offset 0).
            pkt.resize(kVidHdr + vm.size);
            uint8_t* p = pkt.data();
            std::memset(p, 0, kVidHdr);
            p[0] = kMagic; p[1] = 1; p[2] = kTypeVideo; p[3] = kFragStart | kFragEnd;
            wr32(p + 4, ++vseq_);                     // video packet seq
            wr32(p + 8, vframe_++);                   // frame_id, its own counter
            wr32(p + 12, 0);                          // offset
            wr32(p + 16, (uint32_t)vm.size);          // plen
            wr32(p + 20, (uint32_t)vm.size);          // fsize
            wr16(p + 24, (uint16_t)(vm.width ? vm.width : (uint32_t)w_));
            wr16(p + 26, (uint16_t)(vm.height ? vm.height : (uint32_t)h_));
            wr64(p + 32, vm.ts_ns ? vm.ts_ns : m.log_time);
            std::memcpy(p + kVidHdr, vm.data, vm.size);
            on_packet(pkt.data(), (int)pkt.size());
        } else if (it->second == Role::ACCEL) {
            if (mcap::dec_pose_in_frame(m.data, m.size, &accel)) have_accel = true;
        } else if (it->second == Role::GYRO) {
            mcap::PoseMsg gyro;
            if (!mcap::dec_pose_in_frame(m.data, m.size, &gyro)) continue;
            pace(m.log_time);
            if (stop_.load()) break;
            pkt.resize(kImuHdr + kImuSample);
            uint8_t* p = pkt.data();
            std::memset(p, 0, pkt.size());
            p[0] = kMagic; p[1] = 1; p[2] = kTypeImu; p[3] = 0;
            // IMU has its OWN packet seq. Sharing the video one made the frame ids
            // jump by the IMU rate (840 Hz against 30 fps), which a consumer reads
            // as near-total frame loss on a file that is intact.
            wr32(p + 4, ++imu_pkt_seq_);
            wr64(p + 8, iseq_++);                     // base_seq
            wr16(p + 16, 1);                          // n samples
            wr16(p + 18, (uint16_t)kImuSample);       // stride
            uint8_t* s = p + kImuHdr;
            wr64(s, gyro.ts_ns ? gyro.ts_ns : m.log_time);
            if (have_accel) {
                wrf(s + 8,  (float)accel.x);
                wrf(s + 12, (float)accel.y);
                wrf(s + 16, (float)accel.z);
            }
            wrf(s + 20, (float)gyro.x);
            wrf(s + 24, (float)gyro.y);
            wrf(s + 28, (float)gyro.z);
            on_packet(pkt.data(), (int)pkt.size());
        }
    }
    r.close();
    if (verbose_) std::fprintf(stderr, "[ef] mcap replay finished: %s\n", path_.c_str());
    running_.store(false);
    mark_ended();   // grab() surfaces END_OF_STREAM after the last frame
}

}  // namespace internal
}  // namespace ef
