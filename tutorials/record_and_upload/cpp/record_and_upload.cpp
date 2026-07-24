// Example (wireless): record on the device, then upload it over WiFi.
//
// Bluetooth LE carries the control plane while the M1 uploads the finished
// .mcap to your URL over its own WiFi, so the M1 must already be on WiFi (see
// example 03). The URL is any PUT target: a pre-signed AWS S3 URL, or a plain
// http://<host>:<port>/<path> receiver on your network. Nothing streams to this
// host; the device does the upload itself and we just poll its progress.
//
//   ./wireless_05_record_and_upload <ble_mac> <url> [seconds]

#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include <ef/Device.hpp>

using namespace ef;

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <ble_mac> <url> [seconds]\n";
        return 2;
    }
    const std::string url  = argv[2];
    const int         secs = (argc >= 4) ? std::stoi(argv[3]) : 3;

    InitParameters init;
    init.input_type  = INPUT_TYPE::STREAM;   // Bluetooth LE control plane
    init.ble_address = argv[1];

    Device device;
    ERROR_CODE ec = device.open(init);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "open failed: " << to_string(ec) << "\n";
        return 1;
    }

    // Record to the device's eMMC (DEVICE_LOCAL), then stop. A duplicate name is
    // rejected, so re-runs need the earlier one deleted (`ef-cli record delete`).
    const std::string name = "upload_example";
    RecordingParameters rp;
    rp.target = RECORDING_TARGET::DEVICE_LOCAL;
    rp.name   = name;
    if (device.enable_recording(rp) != ERROR_CODE::SUCCESS) {
        std::cerr << "enable_recording failed (does \"" << name
                  << "\" already exist?)\n";
        device.close();
        return 1;
    }
    std::cout << "recording \"" << name << "\" for " << secs << " s...\n";
    std::this_thread::sleep_for(std::chrono::seconds(secs));
    device.disable_recording();

    // Hand the device the destination; it uploads over WiFi in the background.
    // WIFI_NOT_CONNECTED here means the M1 has not joined a network yet.
    std::cout << "uploading \"" << name << "\" to " << url << " ...\n";
    ec = device.upload_recording(name, url);
    if (ec != ERROR_CODE::SUCCESS) {
        std::cerr << "upload_recording failed: " << to_string(ec) << "\n";
        device.close();
        return 1;
    }

    // Poll this session's upload state until it leaves RUNNING (done or failed).
    const double MB = 1024.0 * 1024.0;
    RecordingStatus rs;
    bool settled = false;   // saw a status where the upload was no longer RUNNING
    for (int i = 0; i < 120 && !settled; ++i) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (device.get_recording_status(rs, name) != ERROR_CODE::SUCCESS) continue;
        if (rs.upload == UPLOAD_STATE::RUNNING) {
            std::cout << "  " << rs.upload_bytes_sent / MB << " / "
                      << rs.upload_bytes_total / MB << " MB\n";
            continue;
        }
        settled = true;
    }

    if (!settled) {
        std::cerr << "upload did not finish within the timeout\n";
        device.close();
        return 1;
    }
    if (rs.last_error != ERROR_CODE::SUCCESS) {
        std::cerr << "upload failed: " << to_string(rs.last_error) << "\n";
        device.close();
        return 1;
    }
    std::cout << "upload complete (" << rs.upload_bytes_sent / MB << " MB)\n";
    device.close();
    return 0;
}
