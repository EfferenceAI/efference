////////////////////////////////////////////////////////////////////////////////
//
// File:      02_calibrate_camera.cpp
// Purpose:   Capture checkerboard views and fit Double Sphere intrinsics.
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

// Captures 40 detected views at 0.5-second intervals, then fits the identifiable
// Double Sphere submodel [fx, fy, cx, cy, xi=0, alpha].
//
//   ./build/opencv_02_calibrate_camera
//   ./build/opencv_02_calibrate_camera --pattern 9x6
//   ./build/opencv_02_calibrate_camera --square-size 25.0
//   ./build/opencv_02_calibrate_camera --ble <MAC>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <ef/Checkerboard.h>
#include <ef/Device.hpp>
#include <ef/OpenCV.hpp>

using namespace ef;

namespace {

constexpr int kViews = 40;
constexpr auto kCaptureGap = std::chrono::milliseconds(500);
constexpr char kWindow[] = "efference - calibrate camera";

void draw_board(cv::Mat& image, const std::vector<float>& xy, int cols, int rows) {
    auto point = [&](int r, int c) {
        return cv::Point(cvRound(xy[2 * (r * cols + c)]),
                         cvRound(xy[2 * (r * cols + c) + 1]));
    };
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            if (c + 1 < cols)
                cv::line(image, point(r, c), point(r, c + 1),
                         {255, 0, 0}, 2, cv::LINE_AA);
            if (r + 1 < rows)
                cv::line(image, point(r, c), point(r + 1, c),
                         {255, 0, 0}, 2, cv::LINE_AA);
            cv::circle(image, point(r, c), 4, {0, 0, 255}, -1);
        }
}

}  // namespace

int main(int argc, char** argv) {
    InitParameters init;
    init.flip_mode = FLIP_MODE::ON;
    int cols = 11, rows = 8;
    double square_size = 30.0; // 30mm squares
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--ble") && i + 1 < argc) {
            init.input_type = INPUT_TYPE::STREAM;
            init.ble_address = argv[++i];
        } else if (!std::strcmp(argv[i], "--pattern") && i + 1 < argc) {
            std::sscanf(argv[++i], "%dx%d", &cols, &rows);
        } else if (!std::strcmp(argv[i], "--square-size") && i + 1 < argc) {
            square_size = std::strtod(argv[++i], nullptr);
        }
    }
    if (cols < 3 || rows < 3 || square_size <= 0.0) {
        std::fprintf(stderr, "pattern dimensions and square size must be positive\n");
        return 1;
    }

    Device device;
    ERROR_CODE error = device.open(init);
    if (error != ERROR_CODE::SUCCESS) {
        std::fprintf(stderr, "open failed: %s\n", to_string(error));
        return 1;
    }
    if (ckb_version() < 1) {
        std::fprintf(stderr, "libcamera_cal version 1+ required\n");
        return 1;
    }

    const int npts = cols * rows;
    CkbParams params;
    ckb_default_params(&params);

    cv::namedWindow(kWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(kWindow, 960, 600);

    std::vector<double> observations;
    observations.reserve(2 * npts * kViews);
    auto last_capture = std::chrono::steady_clock::now() - kCaptureGap;
    int width = 0, height = 0;

    Mat frame;
    cv::Mat gray;
    std::vector<float> corners(2 * npts);
    while (observations.size() < static_cast<size_t>(2 * npts * kViews)) {
        ERROR_CODE grab = device.grab();
        if (grab == ERROR_CODE::GRAB_TIMEOUT) continue;
        if (grab != ERROR_CODE::SUCCESS) {
            std::fprintf(stderr, "grab failed: %s\n", to_string(grab));
            break;
        }
        if (device.retrieve_image(frame, VIEW::BGR) != ERROR_CODE::SUCCESS)
            continue;

        cv::Mat bgr = ef::toCvMat(frame);
        width = bgr.cols;
        height = bgr.rows;
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        bool found = ckb_detect_board(
            gray.data, gray.cols, gray.rows, static_cast<int>(gray.step),
            cols, rows, &params, corners.data());

        const auto now = std::chrono::steady_clock::now();
        if (found && now - last_capture >= kCaptureGap) {
            observations.insert(observations.end(), corners.begin(), corners.end());
            last_capture = now;
        }

        if (found) draw_board(bgr, corners, cols, rows);
        char status[32];
        std::snprintf(status, sizeof status, "captured %zu/%d",
                      observations.size() / (2 * npts), kViews);
        cv::putText(bgr, status, {10, 40}, cv::FONT_HERSHEY_SIMPLEX,
                    1.1, {255, 255, 255}, 2);
        cv::imshow(kWindow, bgr);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    device.close();
    cv::destroyAllWindows();

    const int nviews = static_cast<int>(observations.size()) / (2 * npts);
    if (nviews < kViews) {
        std::fprintf(stderr, "aborted with %d/%d views\n", nviews, kViews);
        return 1;
    }

    std::vector<double> object_points(3 * npts, 0.0);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            object_points[3 * (r * cols + c)] = c * square_size;
            object_points[3 * (r * cols + c) + 1] = r * square_size;
        }

    double intr[6], stddev[6], rms;
    if (!ckb_calibrate_double_sphere(
            object_points.data(), observations.data(), nviews, npts,
            width, height, CKB_CALIB_FIX_XI_ZERO,
            intr, stddev, &rms, nullptr)) {
        std::fprintf(stderr, "calibration failed\n");
        return 1;
    }

    static const char* names[] = {"fx", "fy", "cx", "cy", "xi", "alpha"};
    std::printf("\nDouble Sphere (xi fixed 0), %d views, %dx%d\n", nviews, width, height);
    std::printf("RMS: %.4f px\n", rms);
    for (int i = 0; i < 6; ++i)
        std::printf("  %-5s %12.6f  +/- %.4g\n", names[i], intr[i], stddev[i]);
    return 0;
}
