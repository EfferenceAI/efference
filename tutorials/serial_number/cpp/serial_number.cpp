// Example (wired): print the M1's serial number.
//
// The smallest possible SDK program: open over USB, read one field.
//
//   ./wired_01_serial_number

#include <iostream>

#include <ef/Device.hpp>

using namespace ef;

int main() {
    Device device;

    ERROR_CODE ec = device.open();   // USB is the default; capture auto-starts
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    DeviceInformation info = device.get_device_information();
    std::cout << "Serial:   " << info.serial << "\n"
              << "Model:    " << to_string(info.model) << "\n"
              << "Firmware: " << info.firmware_version << "\n";

    device.close();
    return 0;
}
