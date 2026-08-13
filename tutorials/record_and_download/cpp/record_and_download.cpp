// Example (wired): record on the device, then pull the file over USB.
//
// A DEVICE_LOCAL recording is written to the M1's own eMMC and keeps running
// even if the host disconnects. Here we start one, let it run briefly, stop it,
// and download the resulting .mcap over the USB control link (no WiFi needed).
//
//   ./record_and_download [seconds]

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    const int secs = (argc >= 2) ? std::stoi(argv[1]) : 3;

    Device device;
    ERROR_CODE ec = device.open();
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    // Give the recording an explicit name so we can download exactly this one.
    // The device rejects a duplicate name (RECORDING_ALREADY_EXISTS), so a
    // second run needs the earlier one removed first (`ef-cli record delete
    // example_recording`). "" would let the device pick rec_<utc>_<seq>, but
    // then we'd have to guess which entry in list_recordings() is ours.
    const std::string name = "example_recording";

    RecordingParameters rp;
    rp.target = RECORDING_TARGET::DEVICE_LOCAL;
    rp.name   = name;

    ec = device.enable_recording(rp);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "enable_recording failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }
    std::cout << "recording \"" << name << "\" for " << secs << " s...\n";
    std::this_thread::sleep_for(std::chrono::seconds(secs));
    device.disable_recording();

    const std::string dest = name + ".mcap";

    std::cout << "downloading \"" << name << "\" -> " << dest << " ...\n";
    ec = device.download_recording(name, dest);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "download failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }

    std::cout << "saved " << dest << "\n";
    device.close();
    return 0;
}
