// Example (opencv): grab frames over USB and show them in an OpenCV window.
//
// Ask for VIEW::BGR so the frame is already in OpenCV's channel order and
// ef::toCvMat() is a zero-copy view, with no colour conversion or per-frame
// allocation.
//
//   ./opencv_display        # USB
//
// Press ESC or 'q' to quit.

#include <iostream>

#include <opencv2/highgui.hpp>

#include <ef/Device.hpp>
#include <ef/OpenCV.hpp>

using namespace ef;

int main() {
    InitParameters init;              // USB, native resolution

    Device device;
    ERROR_CODE ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        if (ec == ERROR_CODE::DEVICE_NOT_AVAILABLE)
            std::cerr << "(the device is single-client: close any other SDK "
                         "program using it and retry)\n";
        return 1;
    }

    // Resizable window at half the sensor size; imshow's default (WINDOW_AUTOSIZE)
    // would open at the full 1920x1200 frame size and swallow the whole screen.
    const char* kWindow = "efference opencv";
    cv::namedWindow(kWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    cv::resizeWindow(kWindow, 960, 600);

    Mat frame;
    while (true) {
        ERROR_CODE g = device.grab();
        if (g == ERROR_CODE::GRAB_TIMEOUT) continue;      // non-fatal: keep looping
        if (g != ERROR_CODE::SUCCESS) {
            std::cerr << "grab failed: " << to_string(g) << "\n";
            break;
        }
        if (device.retrieve_image(frame, VIEW::BGR) != ERROR_CODE::SUCCESS) continue;

        cv::imshow(kWindow, ef::toCvMat(frame));   // zero-copy
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    device.close();
    return 0;
}
