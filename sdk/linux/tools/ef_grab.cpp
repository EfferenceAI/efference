////////////////////////////////////////////////////////////////////////////////
//
// File:      ef_grab.cpp
// Purpose:   Live capture demo, the open / grab / retrieve loop.
// Author:    Calvin Nguyen, Gianluca Bencomo
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

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

static volatile std::sig_atomic_t g_stop = 0;
static void on_sigint(int) { g_stop = 1; }

// ./ef-grab [seconds] [--codec raw|h264|h265|h264hq|h265hq]
//           [--record out.mcap] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto] [--verbose]
int main(int argc, char** argv) {
    double         seconds = 0;
    std::string    record_path;
    InitParameters init;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--record") && i + 1 < argc) record_path = argv[++i];
        else if (!std::strcmp(argv[i], "--ble") && i + 1 < argc) {
            init.input_type  = INPUT_TYPE::STREAM;
            init.ble_address = argv[++i];
        } else if (!std::strcmp(argv[i], "--udp") && i + 1 < argc) {
            // BLE control + WiFi/UDP data: the host IP the device streams to.
            std::string hp = argv[++i];
            std::string::size_type colon = hp.rfind(':');
            if (colon != std::string::npos) {
                init.udp_port = (uint16_t)std::atoi(hp.c_str() + colon + 1);
                hp.resize(colon);
            }
            init.udp_host = hp;
        } else if (!std::strcmp(argv[i], "--codec") && i + 1 < argc) {
            const std::string c = argv[++i];
            if      (c == "raw")    init.compression = COMPRESSION_MODE::RAW;
            else if (c == "h264")   init.compression = COMPRESSION_MODE::H264;
            else if (c == "h264hq") init.compression = COMPRESSION_MODE::H264_HQ;
            else if (c == "h265")   init.compression = COMPRESSION_MODE::H265;
            else if (c == "h265hq") init.compression = COMPRESSION_MODE::H265_HQ;
            else { std::fprintf(stderr, "unknown codec '%s'\n", c.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--verbose")) {
            init.verbose = 1;
        } else if (!std::strcmp(argv[i], "--flip") && i + 1 < argc) {
            const std::string f = argv[++i];
            if      (f == "on")   init.flip_mode = FLIP_MODE::ON;
            else if (f == "off")  init.flip_mode = FLIP_MODE::OFF;
            else if (f == "auto") init.flip_mode = FLIP_MODE::AUTO;
            else { std::fprintf(stderr, "unknown flip '%s' (on|off|auto)\n", f.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--password") && i + 1 < argc) {
            init.ble_password = argv[++i];
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            seconds = std::atof(argv[i]);
        } else {
            std::fprintf(stderr, "unknown argument '%s'\n", argv[i]);
            return 2;
        }
    }
    std::signal(SIGINT, on_sigint);

    Device dev;
    ERROR_CODE ec = dev.open(init);   // capture starts here
    if (ec != ERROR_CODE::SUCCESS) {
        std::fprintf(stderr, "open failed: %s\n", to_string(ec));
        return 1;
    }
    std::printf("open: %s, state %s\n",
                to_string(dev.get_device_information().input_type),
                to_string(dev.get_state()));

    if (!record_path.empty()) {
        RecordingParameters rp;
        rp.target = RECORDING_TARGET::HOST_FILE;
        rp.path   = record_path;
        if (dev.enable_recording(rp) != ERROR_CODE::SUCCESS)
            std::fprintf(stderr, "recording to %s failed to start\n", record_path.c_str());
    }

    Mat         image;
    SensorsData sensors;
    uint64_t    frames = 0, imu = 0, t0 = dev.get_timestamp().nanoseconds();

    while (!g_stop) {
        if (seconds > 0 &&
            (double)(dev.get_timestamp().nanoseconds() - t0) / 1e9 >= seconds)
            break;

        ec = dev.grab();
        if (ec == ERROR_CODE::GRAB_TIMEOUT)     continue;   // no frame yet, keep looping
        if (ec == ERROR_CODE::CORRUPTED_FRAME)  continue;   // lossy frame dropped
        if (ec != ERROR_CODE::SUCCESS) {
            std::fprintf(stderr, "grab: %s\n", to_string(ec));
            break;
        }

        if (dev.retrieve_image(image) == ERROR_CODE::SUCCESS) frames++;
        if (dev.retrieve_imu(sensors) == ERROR_CODE::SUCCESS) imu += sensors.samples.size();

        if (frames % 30 == 1)
            std::printf("frame %u  %dx%d  ts=%llu  imu=%llu (%s)\n",
                        image.getFrameId(), image.getWidth(), image.getHeight(),
                        (unsigned long long)image.getTimestamp().nanoseconds(),
                        (unsigned long long)imu, to_string(sensors.motion_state));
    }

    double dt = (double)(dev.get_timestamp().nanoseconds() - t0) / 1e9;
    std::printf("\n%llu frames, %llu imu samples in %.1f s (%.1f fps)\n",
                (unsigned long long)frames, (unsigned long long)imu, dt,
                dt > 0 ? frames / dt : 0.0);
    dev.close();
    return 0;
}
