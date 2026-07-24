// Example (wireless): discover attached and nearby M1s.
//
// get_device_list() enumerates USB devices immediately; scan_ble=true also
// scans for M1s advertising over Bluetooth LE. No open() required, this is how
// you find a device's BLE MAC to hand to the other wireless examples.
//
//   ./wireless_02_discover_devices [ble_scan_ms]

#include <iostream>
#include <string>
#include <vector>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    const uint32_t scan_ms = (argc >= 2) ? std::stoul(argv[1]) : 3000;

    std::cout << "Scanning USB + BLE (" << scan_ms << " ms)...\n";
    std::vector<DeviceProperties> devices = Device::get_device_list(true, scan_ms);

    if (devices.empty()) {
        std::cout << "No devices found.\n";
        return 0;
    }

    for (const auto& d : devices) {
        std::cout << "- " << to_string(d.input_type);
        if (d.input_type == INPUT_TYPE::USB)
            std::cout << "  id=" << d.device_id << "  serial=" << d.serial;
        else
            std::cout << "  ble=" << d.ble_address << "  name=\"" << d.ble_name << "\"";
        std::cout << "\n";
    }
    std::cout << devices.size() << " device(s).\n";
    return 0;
}
