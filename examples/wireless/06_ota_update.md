---
title: "OTA update over WiFi (wireless)"
description: "Have the device download and apply a firmware update over its own WiFi."
---

Updates the M1 over the air: you give `update()` a URL and the device downloads
the signed `.eff` itself over WiFi, verifies it, and reboots into it. Control can
stay on USB, or run over Bluetooth for a fully untethered update. The M1 must
already be on WiFi (see the provisioning example).

## Walkthrough

Open the control plane. Leave it on USB, or set a BLE address to drive the update
over Bluetooth:

```cpp
InitParameters init;
if (argc >= 3) {                          // drive the update over Bluetooth
    init.input_type  = INPUT_TYPE::STREAM;
    init.ble_address = argv[2];
}
Device device;
if (device.open(init) != ERROR_CODE::SUCCESS) { std::cerr << "open failed\n"; return 1; }
```

Pass an `http(s)` URL to `update()`. Unlike a local path (which sideloads over
the wire), a URL tells the device to fetch the image over its own WiFi, so the
`DOWNLOADING` progress is a real transfer percentage. The call blocks through
verify and apply, and returns once the device has rebooted:

```cpp
ERROR_CODE ec = device.update(url, [](const UpdateStatus& s) {
    std::cout << "  " << to_string(s.state);
    if (s.progress >= 0)    std::cout << "  " << s.progress << "%";
    if (!s.message.empty()) std::cout << "  " << s.message;
    std::cout << "\n";
});
```

If the device is not on WiFi the download cannot run, so `update()` returns
`FAILED_TO_UPDATE`. On success, `get_device_information().firmware_version` reads
the new build.

## Notes

- Pass `""` as the URL to use the device's configured default update server, and
  call `check_update()` first to see whether a newer build is available there.
- A/B slots keep it safe: a dropped WiFi connection mid-download just leaves the
  running slot untouched, so you can retry.

## Run it

```sh
# control over USB, device downloads over WiFi
./build/wireless_06_ota_update "https://your-server/m1/update.eff"

# fully untethered: control over Bluetooth, download over WiFi
./build/wireless_06_ota_update "https://your-server/m1/update.eff" AA:BB:CC:DD:EE:FF
```

## Expected output

The current firmware version, a line per update phase (with download progress),
and a final "now running firmware <version>" once the device comes back up.
