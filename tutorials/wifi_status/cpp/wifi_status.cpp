// Example (wired): is the M1 on WiFi?
//
// Opens over USB and reads the wireless snapshot cached at open().
//
//   ./wifi_status

#include <iostream>

#include <ef/Device.hpp>

using namespace ef;

int main() {
    Device device;

    ERROR_CODE ec = device.open();   // USB is the default
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    const WirelessConfiguration& wifi = device.get_device_information().wireless;
    if (wifi.wifi_state == "auth_failed")
        std::cout << "Authentication failed for \"" << wifi.wifi_ssid
                  << "\" - check the password and re-provision.\n";
    else if (wifi.wifi_state == "unknown")
        std::cout << "The snapshot could not be refreshed, so the link is unknown.\n";
    else if (!wifi.wifi_connected)
        std::cout << "The M1 is not connected to any WiFi network.\n";
    else if (wifi.wifi_health != WIFI_HEALTH::UNSPECIFIED &&
             wifi.wifi_health <  WIFI_HEALTH::UNVERIFIED)
        // Associated but not usable: no address, no resolver, or nothing reachable.
        std::cout << "The M1 is associated with \"" << wifi.wifi_ssid
                  << "\" but the link cannot carry traffic yet.\n";
    else
        std::cout << "The M1 is on \"" << wifi.wifi_ssid << "\" (ip "
                  << wifi.wifi_ip_address << ", rssi " << wifi.wifi_rssi
                  << ", internet " << (wifi.internet_reachable ? "yes" : "no") << ").\n";

    device.close();
    return 0;
}
