////////////////////////////////////////////////////////////////////////////////
//
// File:      mcap_replay_reader.hpp
// Purpose:   MCAP file-replay data source (internal), analogous to a
//            prerecorded-clip reader.
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

#ifndef EF_MCAP_REPLAY_READER_HPP
#define EF_MCAP_REPLAY_READER_HPP

#include <map>
#include <string>

#include "stream_assembler.hpp"

namespace ef {
namespace internal {

// Replays a recording (host or downloaded device-local session, same format;
// see mcap.hpp) through the shared StreamAssembler so grab()/retrieve_*()
// behave as on a live session. A playback thread walks Message records, paces
// them by log_time, and feeds synthesized ef_stream packets into on_packet();
// accel/gyro PoseInFrame pairs are re-joined into ImuSamples. EOF surfaces as
// END_OF_STREAM from grab() (mark_ended), like transport loss on a live run.
class McapReplayReader : public StreamAssembler {
public:
    McapReplayReader() = default;
    ~McapReplayReader() override { stop(); }

    // Pre-scans the file for the video format/geometry, then starts playback.
    // FILE_NOT_FOUND: can't open; NO_STREAM: not an MCAP / no known channels.
    Status start(const std::string& path, int verbose);
    void   stop() override;

    // Detected during the pre-scan ("" when the recording carries no video).
    const std::string& format() const { return fmt_; }
    int width() const  { return w_; }
    int height() const { return h_; }

private:
    void run();

    std::string path_;
    int         verbose_ = 0;
    std::string fmt_;
    int         w_ = 0, h_ = 0;
    uint32_t    vseq_ = 0;      // synthesized ef_stream packet sequence
    uint64_t    iseq_ = 0;      // synthesized IMU sample sequence
};

}  // namespace internal
}  // namespace ef

#endif  // EF_MCAP_REPLAY_READER_HPP
