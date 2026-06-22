#include <iostream>
#include <cstdlib>
#include <exception>

#include <ef/device.hpp>

using namespace ef;

int main() {
    Device device;

    InitParameters init_params;
    init_params.verbose = 0;

    ERROR_CODE status = device.open(init_params);
    if (status != ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open device. Error: " << static_cast<int>(status) << "\n";
        return 1;
    }

    try {
        DeviceInformation device_info = device.get_device_information();
        std::cout << "Hello. This is my serial number: " << device_info.serial_number << ".\n";
    } catch (const std::exception& e) {
        std::cerr << "Failed to retrieve device information. Error: " << e.what() << "\n";
        device.close();
        return 1;
    }

    device.close();
    return 0;
}