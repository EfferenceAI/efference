// Example (opencv): grab frames over USB and show them in an OpenCV window.
//
// Ask for VIEW::BGR so the frame is already in OpenCV's channel order and
// ef::toCvMat() is a zero-copy view, with no colour conversion or per-frame
// allocation.
//
//   ./opencv_01_opencv_display        # USB
//
// Press ESC or 'q' to quit.

#include <iostream>

#include <opencv2/highgui.hpp>

#include <ef/Device.hpp>
#include <ef/opencv.hpp>

using namespace ef;

int main() {
    InitParameters init;              // USB, native resolution

    Device dev;
    if (dev.open(init) != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed\n";
        return 1;
    }

    Mat frame;
    while (true) {
        ERROR_CODE g = dev.grab();
        if (g == ERROR_CODE::GRAB_TIMEOUT) continue;      // non-fatal: keep looping
        if (g != ERROR_CODE::SUCCESS) {
            std::cerr << "grab failed: " << to_string(g) << "\n";
            break;
        }
        if (dev.retrieve_image(frame, VIEW::BGR) != ERROR_CODE::SUCCESS) continue;

        cv::imshow("efference opencv", ef::toCvMat(frame));   // zero-copy
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') break;
    }

    dev.close();
    return 0;
}
