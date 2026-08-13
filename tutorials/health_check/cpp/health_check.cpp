// Example (wired): run the device health check.
//
// Opens over USB, runs a health sweep, and prints every probe. Pass `--deep`
// to include the stress tier.
//
//   ./health_check [--deep]

#include <iostream>
#include <string>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    const bool deep = (argc >= 2 && std::string(argv[1]) == "--deep");

    Device device;
    ERROR_CODE ec = device.open();
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    HealthStatus health;
    ec = device.health_check(health, deep);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "health check failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }

    std::cout << "overall=" << (health.passed ? "PASS" : "FAIL")
              << " camera=" << to_string(health.camera)
              << " imu="    << to_string(health.imu)
              << " checks=" << health.checks.size() << "\n";
    for (const auto& c : health.checks)
        std::cout << "  [" << (c.passed ? "PASS" : "FAIL") << "] " << c.name
                  << (c.detail.empty() ? "" : (" - " + c.detail)) << "\n";

    device.close();
    return health.passed ? 0 : 1;
}
