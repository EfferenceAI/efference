// Example (wireless): open over USB or Bluetooth.
//
// With no argument, opens over USB. Given a BLE MAC, opens the control plane
// over Bluetooth LE instead (video/IMU over WiFi is example 04).
//
//   ./wireless_01_choosing_a_connection [BLE_MAC]

#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    InitParameters init;

    if (argc >= 2) {
        init.input_type  = INPUT_TYPE::STREAM;   // BLE control plane
        init.ble_address = argv[1];
        std::cout << "Opening over Bluetooth (" << init.ble_address << ")...\n";
    } else {
        std::cout << "Opening over USB (default)...\n";
    }

    Device device;
    ERROR_CODE ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    const WirelessConfiguration& wifi = device.get_device_information().wireless;
    std::cout << "Connected over "
              << (init.input_type == INPUT_TYPE::STREAM ? "Bluetooth" : "USB")
              << " (state " << to_string(device.get_state()) << "). The M1's WiFi is "
              << (wifi.wifi_connected
                      ? ("on \"" + wifi.wifi_ssid + "\" (" + wifi.wifi_ip_address + ")")
                      : std::string("not connected"))
              << ".\n";

    device.close();
    return 0;
}
