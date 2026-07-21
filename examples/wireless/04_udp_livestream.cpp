// Example (wireless): live video + IMU over WiFi, controlled over Bluetooth.
//
// The fully-wireless data path: Bluetooth LE carries the control plane while
// the M1 pushes H.265 video and IMU to this host over WiFi/UDP. Set udp_host to
// THIS machine's IP as the device can reach it on the WiFi network (the device
// cannot infer it over BLE). The M1 must already be on WiFi (see example 03),
// and this host must be on the same network.
//
//   ./wireless_04_udp_livestream <ble_mac> <this_host_ip> [udp_port] [num_frames]

#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <ble_mac> <this_host_ip> [udp_port] [num_frames]\n";
        return 2;
    }

    InitParameters init;
    init.input_type  = INPUT_TYPE::STREAM;        // BLE control + WiFi/UDP data
    init.ble_address = argv[1];
    init.udp_host    = argv[2];                    // where the device sends video
    if (argc >= 4) init.udp_port = static_cast<uint16_t>(std::stoi(argv[3]));
    init.compression = COMPRESSION_MODE::H265;
    init.enable_imu  = true;

    const int target = (argc >= 5) ? std::stoi(argv[4]) : 150;

    std::cout << "BLE control " << init.ble_address
              << ", video -> " << init.udp_host << ":" << init.udp_port << " ...\n";

    Device device;
    ERROR_CODE status = device.open(init);
    if (status != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(status) << "\n";
        return 1;
    }

    Mat         frame;
    SensorsData imu;
    int         frames    = 0;
    uint64_t    imu_total = 0;

    while (frames < target) {
        ERROR_CODE g = device.grab();
        if (g == ERROR_CODE::GRAB_TIMEOUT) continue;
        if (g != ERROR_CODE::SUCCESS) {
            std::cerr << "grab: " << to_string(g) << "\n";
            break;
        }
        device.retrieve_image(frame, VIEW::NV12);
        device.retrieve_imu(imu, TIME_REFERENCE::IMAGE);
        imu_total += imu.samples.size();
        ++frames;
        if (frames % 30 == 0)
            std::cout << "frame " << frame.getFrameId() << "  "
                      << frame.getWidth() << "x" << frame.getHeight()
                      << "  imu " << imu_total << "\n";
    }

    std::cout << "streamed " << frames << " frames over WiFi ("
              << imu_total << " IMU samples)\n";
    device.close();
    return 0;
}
