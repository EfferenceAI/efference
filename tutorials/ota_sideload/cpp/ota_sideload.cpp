// Example (wired): update the firmware by sideloading a local .eff over USB.
//
// Passing a local file path to update() streams the signed .eff across the USB
// control link, with no network involved. The device verifies the signature,
// writes it to the inactive A/B slot, and reboots into it. The A/B scheme means
// an interrupted or bad update never bricks the device: it falls back to the
// slot that was already running.
//
//   ./wired_06_ota_sideload <path-to-update.eff>

#include <fstream>
#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <update.eff>\n";
        return 2;
    }
    const std::string eff = argv[1];
    if (!std::ifstream(eff).good()) {
        std::cerr << "no such file: " << eff << "\n";
        return 2;
    }

    Device device;
    if (device.open() != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed\n";
        return 1;
    }

    std::cout << "current firmware: "
              << device.get_device_information().firmware_version << "\n"
              << "sideloading " << eff << " over USB...\n";

    // update() blocks until the device has received, verified, and applied the
    // image. The callback fires on each phase (DOWNLOADING here means receiving
    // over the wire, then VERIFYING, READY_TO_APPLY, APPLYING).
    ERROR_CODE ec = device.update(eff, [](const UpdateStatus& s) {
        std::cout << "  " << to_string(s.state);
        if (s.progress >= 0)      std::cout << "  " << s.progress << "%";
        if (!s.message.empty())   std::cout << "  " << s.message;
        std::cout << "\n";
    });
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "update failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }

    std::cout << "update complete, now running firmware "
              << device.get_device_information().firmware_version << "\n";
    device.close();
    return 0;
}
