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
#include <memory>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ef/Device.hpp>
#include <ef/OpenCV.hpp>   // ef::toCvMat, zero-copy ef::Mat -> cv::Mat

using namespace ef;

static volatile std::sig_atomic_t g_stop = 0;
static void on_signal(int) { g_stop = 1; }

// ./efference-viewer [--codec raw|h264|h265|h264hq|h265hq] [--h264|--h265] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto] [--headless]
// (--h264/--h265 are shorthands for --codec h264/h265.)
// Rectification is done on-device (see `ef calibration --camera --rectify`); the
// viewer just displays the delivered frames.
// --headless (alias --no-window): open the session and hold it, no OpenCV window.
// Use it to command the device to push video+IMU to a --udp target (a remote
// host) without a local display; --flip is a host-side display transform and has
// no effect on the forwarded stream, so the receiver applies its own flip.
int main(int argc, char** argv) {
    InitParameters init;
    bool headless = false;
    bool flip_set = false;
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
            // The host the device streams video+IMU to over WiFi/UDP. Works with
            // either control transport: USB (default) or BLE (--ble). Point it at
            // a reachable IP (a remote host, or your own LAN IP), not 127.0.0.1.
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
            flip_set = true;
        } else if (!std::strcmp(argv[i], "--password") && i + 1 < argc) {
            init.ble_password = argv[++i];
        } else if (!std::strcmp(argv[i], "--headless") || !std::strcmp(argv[i], "--no-window")) {
            headless = true;
        } else {
            std::fprintf(stderr, "unknown or malformed argument '%s'\n", argv[i]);
            return 2;
        }
    }

    // ---- warn on flag combinations that can't do what they look like ----------
    const bool forwarding = !init.udp_host.empty();
    // BLE has no local data plane: without --udp there is nothing to grab, so
    // open()/grab() would fail. Say why up front instead of a bare error code.
    if (init.input_type == INPUT_TYPE::STREAM && !forwarding) {
        std::fprintf(stderr,
            "--ble needs --udp <host>: BLE carries only control, so the device has\n"
            "nowhere to send video (USB isoc is the only local data plane). Add\n"
            "--udp <reachable-ip>, or drop --ble to capture over USB.\n");
        return 2;
    }
    // Headless with no target neither shows nor forwards: every frame is discarded.
    if (headless && !forwarding)
        std::fprintf(stderr,
            "warning: --headless without --udp neither shows nor forwards video; "
            "frames are captured over USB and discarded.\n");
    // --flip only rotates the local window, so it does nothing headless or forwarded.
    if (flip_set && init.flip_mode != FLIP_MODE::OFF && (headless || forwarding))
        std::fprintf(stderr,
            "warning: --flip only rotates the local viewer window; it does not "
            "affect the %s. The receiver applies its own flip.\n",
            headless ? "session (headless: no window)" : "forwarded stream");

    std::signal(SIGINT, on_signal);    // Ctrl-C
    std::signal(SIGTERM, on_signal);

    Device dev;
    ERROR_CODE ec = dev.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::fprintf(stderr, "open failed: %s\n", to_string(ec));
        return 1;
    }

    const char* kWindow = "efference-viewer";
    if (!headless) {
        cv::namedWindow(kWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        // start the window near half the sensor size; it stays user-resizable.
        const CameraConfiguration& cam = dev.get_device_information().camera_configuration;
        int w = cam.resolution.width  > 0 ? cam.resolution.width  : 1920;
        int h = cam.resolution.height > 0 ? cam.resolution.height : 1200;
        cv::resizeWindow(kWindow, w / 2, h / 2);
    }

    Mat         image;
    SensorsData sensors;
    uint64_t    imu_total = 0, frames_ok = 0, timeouts = 0;
    // Latest IMU sample, held across grabs so the readout persists on frames
    // with no new samples. accel m/s^2, gyro rad/s.
    float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
    const uint64_t start_ns = dev.get_timestamp().nanoseconds();
    uint64_t last_log_ns = start_ns;
    bool window_shown = false;   // latched once the window actually maps

    if (headless && forwarding)
        std::fprintf(stderr, "forwarding to %s:%u over WiFi/UDP; local frames "
                     "stay 0 when the target is remote. Ctrl-C to stop.\n",
                     init.udp_host.c_str(), init.udp_port);

    // Headless has no window, so emit a once-per-second liveness line instead.
    auto heartbeat = [&](uint64_t now) {
        if (!headless || now - last_log_ns < 1000000000ULL) return;
        last_log_ns = now;
        std::fprintf(stderr, "streaming %s -> %s:%u  frames=%llu timeouts=%llu "
                     "imu=%llu  up %llus\n", to_string(init.compression),
                     init.udp_host.c_str(), init.udp_port,
                     (unsigned long long)frames_ok, (unsigned long long)timeouts,
                     (unsigned long long)imu_total,
                     (unsigned long long)((now - start_ns) / 1000000000ULL));
    };

    while (!g_stop) {
        ec = dev.grab();
        if (ec == ERROR_CODE::GRAB_TIMEOUT || ec == ERROR_CODE::CORRUPTED_FRAME) {
            ++timeouts;
            if (headless) heartbeat(dev.get_timestamp().nanoseconds());
            else cv::waitKey(1);   // pump the GUI event loop
            continue;
        }
        if (ec != ERROR_CODE::SUCCESS) {
            std::fprintf(stderr, "grab: %s\n", to_string(ec));
            break;
        }
        ++frames_ok;
        if (dev.retrieve_imu(sensors) == ERROR_CODE::SUCCESS && !sensors.samples.empty()) {
            const ImuSample& s = sensors.samples.back();   // newest of this batch
            ax = s.acceleration[0];     ay = s.acceleration[1];     az = s.acceleration[2];
            gx = s.angular_velocity[0]; gy = s.angular_velocity[1]; gz = s.angular_velocity[2];
            imu_total += sensors.samples.size();
        }

        // No window in headless; frames only arrive on udp-to-self. Just tick.
        if (headless) { heartbeat(dev.get_timestamp().nanoseconds()); continue; }

        // VIEW::BGR is decoded to OpenCV channel order, so toCvMat() is zero-copy.
        if (dev.retrieve_image(image, VIEW::BGR) != ERROR_CODE::SUCCESS) continue;
        cv::Mat frame = ef::toCvMat(image);
        if (frame.empty()) continue;

        // Dark status strip above the video (never covers it) with frame + IMU.
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
        int tx = (frame.cols - ts.width) / 2; if (tx < 0) tx = 0;   // centered
        cv::putText(bar, hud, cv::Point(tx, (bar_h + ts.height) / 2),
                    cv::FONT_HERSHEY_SIMPLEX, font_scale, cv::Scalar(0, 255, 0),
                    thickness, cv::LINE_AA);

        cv::Mat canvas;
        cv::vconcat(bar, frame, canvas);
        cv::imshow(kWindow, canvas);

        // Echo once per second; the overlay is easy to miss.
        uint64_t now = dev.get_timestamp().nanoseconds();
        if (now - last_log_ns >= 1000000000ULL) {
            last_log_ns = now;
            std::fprintf(stderr,
                "frame %u  accel[%+.2f %+.2f %+.2f] m/s^2  gyro[%+.2f %+.2f %+.2f] rad/s  "
                "imu=%llu\n",
                image.getFrameId(), ax, ay, az, gx, gy, gz,
                (unsigned long long)imu_total);
        }

        // waitKey paints the window and polls for quit (ESC/q); run it each frame.
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
        // Trust a not-visible report only after the window has mapped once
        // (compositors, esp. Wayland, report <1 during startup).
        if (cv::getWindowProperty(kWindow, cv::WND_PROP_VISIBLE) >= 1) window_shown = true;
        else if (window_shown) break;
    }

    if (!headless) cv::destroyAllWindows();
    dev.close();
    return 0;
}
