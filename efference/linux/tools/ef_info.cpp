// ef-info — dump the full DeviceInformation a connected device reports.
// A diagnostic companion to example 1 (which prints only the serial number).
//
//   ef-info [device_index]
#include <ef/device.hpp>

#include <cstdlib>
#include <iostream>

using namespace ef;

int main(int argc, char** argv) {
    Device device;

    InitParameters init;
    init.verbose   = 1;
    init.device_id = (argc > 1) ? std::atoi(argv[1]) : 0;

    ERROR_CODE st = device.open(init);
    if (st != ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open device: " << to_string(st) << "\n";
        return 1;
    }

    try {
        DeviceInformation i = device.get_device_information();
        const auto& c = i.camera;
        const auto& e = i.encoding;
        const auto& m = i.imu;
        const auto& k = m.calibration;

        std::cout << "\n=== DeviceInformation ===\n";
        std::cout << "  serial_number    : " << i.serial_number << "\n";
        std::cout << "  model            : " << to_string(i.model) << " (\"" << i.model_name << "\")\n";
        std::cout << "  firmware_version : " << i.firmware_version
                  << " (int " << i.firmware_version_int << ")\n";
        std::cout << "  input_type       : " << to_string(i.input_type) << "\n";

        std::cout << "\n  [camera]\n";
        std::cout << "    fps/resolution : " << c.fps << " / " << to_string(c.resolution)
                  << " (" << c.width << "x" << c.height << ")\n";
        std::cout << "    intrinsics     : fx=" << c.fx << " fy=" << c.fy
                  << " cx=" << c.cx << " cy=" << c.cy << " xi=" << c.xi << " alpha=" << c.alpha << "\n";
        std::cout << "    extrinsics     : t=(" << c.tx << "," << c.ty << "," << c.tz << ")"
                  << " q=(" << c.rw << "," << c.rx << "," << c.ry << "," << c.rz << ")\n";
        std::cout << "    exposure       : " << to_string(c.auto_exposure_mode)
                  << " manual=" << c.manual_exposure_time << "us iso_limit=" << c.iso_limit << "\n";
        std::cout << "    white_balance  : " << to_string(c.white_balance_mode)
                  << " manual=" << c.manual_white_balance_temperature << "K\n";
        std::cout << "    gamma          : " << to_string(c.gamma_mode) << "\n";
        std::cout << "    noise_reduction: " << to_string(c.noise_reduction_mode)
                  << " strength=" << c.noise_reduction_strength << "\n";
        std::cout << "    sharpening     : " << c.sharpening_strength << "\n";
        std::cout << "    distortion     : " << to_string(c.distortion_mode) << "\n";

        std::cout << "\n  [encoding]\n";
        std::cout << "    codec/bitrate/rc: " << to_string(e.codec) << " / " << e.bitrate
                  << " / " << to_string(e.rate_control) << "\n";

        std::cout << "\n  [imu]\n";
        std::cout << "    enabled        : " << (m.enabled ? "true" : "false") << "\n";
        std::cout << "    accel/gyro     : +-" << m.accelerometer_range << "g / +-"
                  << m.gyroscope_range << "dps @ " << m.sampling_rate << "Hz\n";
        std::cout << "    noise_density  : accel=" << k.accel_noise_density
                  << " gyro=" << k.gyro_noise_density << "\n";
        std::cout << "    random_walk    : accel=" << k.accel_random_walk
                  << " gyro=" << k.gyro_random_walk << "\n";

        std::cout << "\n  [packer]\n";
        std::cout << "    format/segments: " << i.packer.format << " / " << i.packer.segments << "ms\n";
        std::cout << "    topics         : " << i.packer.topics.size() << "\n";

        std::cout << "\n  [interfaces]\n";
        std::cout << "    usb.log_level  : " << i.usb.log_level << "\n";
        std::cout << "    emmc.backup    : " << (i.emmc.backup ? "true" : "false") << "\n";
        std::cout << "    wifi           : ssid=\"" << i.wifi.ssid << "\" ip=" << i.wifi.ip_address
                  << " log=" << i.wifi.log_level << "\n";
        std::cout << "    bt             : mac=" << i.bt.mac_address
                  << " paired=" << (i.bt.paired_state ? "true" : "false")
                  << " log=" << i.bt.log_level << "\n";

        const auto& cap = i.caps;
        std::cout << "\n  [caps]\n";
        std::cout << "    codecs        : ";
        for (auto& s : cap.codecs)        std::cout << s << " ";
        std::cout << "\n    pixel_formats : ";
        for (auto& s : cap.pixel_formats) std::cout << s << " ";
        std::cout << "\n    containers    : ";
        for (auto& s : cap.containers)    std::cout << s << " ";
        std::cout << "\n    framerates    : ";
        for (auto  f : cap.framerates_fps) std::cout << f << " ";
        std::cout << "\n    resolutions   : ";
        for (auto& r : cap.resolutions)
            std::cout << r.name << "(" << (r.binning.empty() ? "-" : r.binning) << ") ";
        std::cout << "\n";

        std::cout << "\n  [live]\n";
        std::cout << "    orchestrator_reachable : " << (i.orchestrator_reachable ? "true" : "false") << "\n";
        std::cout << "    capture_reachable      : " << (i.capture_reachable ? "true" : "false") << "\n";

        std::cout << "\n  raw_json: " << i.raw_json << "\n";
    } catch (const std::exception& ex) {
        std::cerr << "get_device_information failed: " << ex.what() << "\n";
        device.close();
        return 1;
    }

    device.close();
    return 0;
}
