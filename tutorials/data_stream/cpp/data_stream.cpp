// Example (wired): stream video + IMU over USB.
//
// open() auto-starts capture over USB, so grab() delivers frames immediately.
// Each grab() latches a frame; retrieve_image() / retrieve_imu()
// hand back the pixels and the IMU samples captured since the previous grab().
//
//   ./data_stream [num_frames]

#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    const int target = (argc >= 2) ? std::stoi(argv[1]) : 150;   // ~5 s at 30 fps

    InitParameters init;              // USB, native resolution, H265, IMU on
    init.enable_imu = true;

    Device device;
    ERROR_CODE ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    Mat         frame;
    SensorsData imu;
    uint64_t    imu_total = 0;
    int         frames    = 0;

    while (frames < target) {
        ERROR_CODE g = device.grab();
        if (g == ERROR_CODE::GRAB_TIMEOUT) continue;   // non-fatal: keep looping
        if (g != ERROR_CODE::SUCCESS) {
            std::cerr << "grab failed: " << to_string(g) << "\n";
            break;
        }

        if (device.retrieve_image(frame, VIEW::NV12) != ERROR_CODE::SUCCESS)
            continue;                                  // no decoded frame this grab
        device.retrieve_imu(imu, TIME_REFERENCE::IMAGE);   // may be empty; that's fine
        imu_total += imu.samples.size();
        ++frames;

        if (frames % 30 == 0)
            std::cout << "frame " << frame.getFrameId()
                      << "  " << frame.getWidth() << "x" << frame.getHeight()
                      << "  imu +" << imu.samples.size() << " (" << imu_total << " total)"
                      << "  motion=" << to_string(imu.motion_state) << "\n";
    }

    std::cout << "captured " << frames << " frames, " << imu_total << " IMU samples\n";
    device.close();
    return 0;
}
