////////////////////////////////////////////////////////////////////////////////
//
// File:      efference_viewer.cpp
// Purpose:   Live viewer, decoded video in an OpenCV window + live IMU values.
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

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ef/Device.hpp>
#include <ef/opencv.hpp>   // ef::toCvMat, zero-copy ef::Mat -> cv::Mat

using namespace ef;

static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

// ./efference-viewer [--codec raw|h264|h265|h264hq|h265hq] [--h264|--h265] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto]
// (--h264/--h265 are shorthands for --codec h264/h265.)
int main(int argc, char** argv) {
    InitParameters init;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--h265")) init.compression = COMPRESSION_MODE::H265;
        else if (!std::strcmp(argv[i], "--h264")) init.compression = COMPRESSION_MODE::H264;
        else if (!std::strcmp(argv[i], "--codec") && i + 1 < argc) {
            const std::string c = argv[++i];
            if      (c == "raw")    init.compression = COMPRESSION_MODE::RAW;
            else if (c == "h264")   init.compression = COMPRESSION_MODE::H264;
            else if (c == "h264hq") init.compression = COMPRESSION_MODE::H264_HQ;
            else if (c == "h265")   init.compression = COMPRESSION_MODE::H265;
            else if (c == "h265hq") init.compression = COMPRESSION_MODE::H265_HQ;
            else { std::fprintf(stderr, "unknown codec '%s'\n", c.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--ble") && i + 1 < argc) {
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
        } else if (!std::strcmp(argv[i], "--flip") && i + 1 < argc) {
            const std::string f = argv[++i];
            if      (f == "on")   init.flip_mode = FLIP_MODE::ON;
            else if (f == "off")  init.flip_mode = FLIP_MODE::OFF;
            else if (f == "auto") init.flip_mode = FLIP_MODE::AUTO;
            else { std::fprintf(stderr, "unknown flip '%s' (on|off|auto)\n", f.c_str()); return 2; }
        } else if (!std::strcmp(argv[i], "--password") && i + 1 < argc) {
            init.ble_password = argv[++i];
        } else {
            std::fprintf(stderr, "unknown or malformed argument '%s'\n", argv[i]);
            return 2;
        }
    }

    std::signal(SIGINT, on_signal);    // Ctrl-C
    std::signal(SIGTERM, on_signal);

    Device dev;
    ERROR_CODE ec = dev.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::fprintf(stderr, "open failed: %s\n", to_string(ec));
        return 1;
    }

    const char* kWindow = "efference-viewer";
    cv::namedWindow(kWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    {   // start the window near half the sensor size; it stays user-resizable.
        const CameraConfiguration& cam = dev.get_device_information().camera_configuration;
        int w = cam.resolution.width  > 0 ? cam.resolution.width  : 1920;
        int h = cam.resolution.height > 0 ? cam.resolution.height : 1080;
        cv::resizeWindow(kWindow, w / 2, h / 2);
    }

    Mat         image;
    SensorsData sensors;
    uint64_t    imu_total = 0;
    // Latest IMU sample (m/s^2 accel, rad/s gyro), held across grabs so the
    // readout persists on frames that carried no new samples.
    float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
    uint64_t last_log_ns = dev.get_timestamp().nanoseconds();
    bool window_shown = false;   // latched once the window actually maps

    while (!g_stop) {
        ec = dev.grab();
        if (ec == ERROR_CODE::GRAB_TIMEOUT || ec == ERROR_CODE::CORRUPTED_FRAME) {
            if (cv::waitKey(1) >= 0) { /* fall through to key handling below */ }
            continue;
        }
        if (ec != ERROR_CODE::SUCCESS) {
            std::fprintf(stderr, "grab: %s\n", to_string(ec));
            break;
        }
        if (dev.retrieve_imu(sensors) == ERROR_CODE::SUCCESS && !sensors.samples.empty()) {
            const ImuSample& s = sensors.samples.back();   // newest of this batch
            ax = s.acceleration[0];     ay = s.acceleration[1];     az = s.acceleration[2];
            gx = s.angular_velocity[0]; gy = s.angular_velocity[1]; gz = s.angular_velocity[2];
            imu_total += sensors.samples.size();
        }

        // VIEW::BGR is decoded + colour-converted by the SDK to OpenCV's native
        // channel order, so ef::toCvMat() is a zero-copy, display-ready view.
        if (dev.retrieve_image(image, VIEW::BGR) != ERROR_CODE::SUCCESS) continue;
        cv::Mat frame = ef::toCvMat(image);
        if (frame.empty()) continue;

        // Status bar ABOVE the video (never covers the image): a dark strip the
        // frame's width with the frame + IMU readout centered. vconcat copies to
        // a fresh canvas, so the SDK's frame buffer is untouched.
        char hud[256];
        std::snprintf(hud, sizeof hud,
                      "frame %u    accel[%+.2f %+.2f %+.2f] m/s^2    gyro[%+.2f %+.2f %+.2f] rad/s",
                      image.getFrameId(), ax, ay, az, gx, gy, gz);
        const int    bar_h      = 44;
        const double font_scale = 0.6;
        const int    thickness  = 1;
        int baseline = 0;
        cv::Size ts = cv::getTextSize(hud, cv::FONT_HERSHEY_SIMPLEX, font_scale,
                                      thickness, &baseline);
        cv::Mat bar(bar_h, frame.cols, frame.type(), cv::Scalar(28, 28, 28));
        int tx = (frame.cols - ts.width) / 2; if (tx < 0) tx = 0;   // horizontally centered
        cv::putText(bar, hud, cv::Point(tx, (bar_h + ts.height) / 2),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0),
                    thickness, cv::LINE_AA);

        cv::Mat canvas;
        cv::vconcat(bar, frame, canvas);
        cv::imshow(kWindow, canvas);

        // Also echo to the console once per second (overlays are easy to miss).
        uint64_t now = dev.get_timestamp().nanoseconds();
        if (now - last_log_ns >= 1000000000ULL) {
            last_log_ns = now;
            std::fprintf(stderr,
                "frame %u  accel[%+.2f %+.2f %+.2f] m/s^2  gyro[%+.2f %+.2f %+.2f] rad/s  "
                "imu=%llu\n",
                image.getFrameId(), ax, ay, az, gx, gy, gz,
                (unsigned long long)imu_total);
        }

        // Pump the GUI event loop and poll for quit (ESC or 'q'). waitKey is what
        // makes the window actually paint, so it must run every frame.
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
        // Detect a title-bar close, but only AFTER the window maps once: the
        // compositor may report not-yet-visible on the first frames (startup
        // race, esp. Wayland), so latch VISIBLE>=1 before trusting a later <1.
        if (cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE) >= 1) window_shown = true;
        else if (window_shown) break;
    }

    cv::destroyAllWindows();
    dev.close();
    return 0;
}
