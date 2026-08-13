// Example (wireless): put the M1 on a WiFi network.
//
// Provisioning works over any control link. Over USB (default) just run it;
// to provision over Bluetooth pass the BLE MAC as the 4th argument. The country
// code (e.g. "US") unlocks 5 GHz channels. This is the prerequisite for the
// WiFi/UDP livestream in example 04.
//
//   ./wifi_provisioning <ssid> <password> [country] [ble_mac]

#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0]
                  << " <ssid> <password> [country] [ble_mac]\n";
        return 2;
    }
    const std::string ssid    = argv[1];
    const std::string psk     = argv[2];
    const std::string country = (argc >= 4) ? argv[3] : "";

    InitParameters init;
    if (argc >= 5) {                          // provision over BLE instead of USB
        init.input_type  = INPUT_TYPE::STREAM;
        init.ble_address = argv[4];
    }

    Device device;
    ERROR_CODE ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    ec = device.wifi_add(ssid, psk, country);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "wifi_add failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }
    ec = device.wifi_select(ssid);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "wifi_select failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }

    // The SDK refreshes the cached wireless snapshot after a provisioning verb.
    // Association is asynchronous, so it may still be connecting right here.
    const WirelessConfiguration& w = device.get_device_information().wireless;
    if (w.wifi_connected)
        std::cout << "Connected to \"" << w.wifi_ssid << "\" (ip "
                  << w.wifi_ip_address << ").\n";
    else
        std::cout << "Provisioned \"" << ssid
                  << "\". Association in progress - re-check with wifi_status.\n";

    device.close();
    return 0;
}
