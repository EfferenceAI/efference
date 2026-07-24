// Example (wireless): update the firmware over the air.
//
// Passing an http(s):// URL to update() has the device download the signed .eff
// itself over its WiFi (the M1 must already be on WiFi, see example 03), verify
// it, write it to the inactive A/B slot, and reboot. Control can stay on USB, or
// pass a BLE MAC to drive the whole update untethered over Bluetooth.
//
//   ./wireless_06_ota_update <url> [ble_mac]

#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <url> [ble_mac]\n";
        return 2;
    }
    const std::string url = argv[1];

    InitParameters init;
    if (argc >= 3) {                          // drive the update over Bluetooth
        init.input_type  = INPUT_TYPE::STREAM;
        init.ble_address = argv[2];
    }

    Device device;
    if (device.open(init) != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed\n";
        return 1;
    }

    std::cout << "current firmware: "
              << device.get_device_information().firmware_version
              << ", fetching " << url << " over WiFi...\n";

    // The device downloads over WiFi, so DOWNLOADING progress is a real transfer
    // percentage. update() blocks through VERIFYING, READY_TO_APPLY, APPLYING,
    // and returns once the device has rebooted into the new slot. If the device
    // is not on WiFi the download can't run, so update() returns FAILED_TO_UPDATE.
    ERROR_CODE ec = device.update(url, [](const UpdateStatus& s) {
        std::cout << "  " << to_string(s.state);
        if (s.progress >= 0)    std::cout << "  " << s.progress << "%";
        if (!s.message.empty()) std::cout << "  " << s.message;
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
