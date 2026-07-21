---
title: "OTA sideload (wired)"
description: "Update the firmware from a local .eff over the USB wire, no network."
---

Updates the M1 by streaming a signed firmware file across the USB control link.
Nothing touches the network: you hand `update()` a local path and the device
takes it from there. Use this on the bench, or anywhere the device has no WiFi.

## Walkthrough

Open over USB and read the current firmware version so you can confirm the bump
afterward:

```cpp
Device device;
if (device.open() != ERROR_CODE::SUCCESS) { std::cerr << "open failed\n"; return 1; }

std::cout << "current firmware: "
          << device.get_device_information().firmware_version << "\n";
```

Pass a local file path to `update()`. Because it is a path (not an `http(s)` URL),
the SDK sideloads the bytes over the USB control link. The device verifies the
signature, writes the image to the inactive A/B slot, and reboots into it. The
call blocks and reports each phase through the callback:

```cpp
ERROR_CODE ec = device.update(eff, [](const UpdateStatus& s) {
    std::cout << "  " << to_string(s.state);          // DOWNLOADING/VERIFYING/READY_TO_APPLY/APPLYING
    if (s.progress >= 0)    std::cout << "  " << s.progress << "%";
    if (!s.message.empty()) std::cout << "  " << s.message;
    std::cout << "\n";
});
```

When `update()` returns `SUCCESS` the device has rebooted into the new slot, so
`get_device_information().firmware_version` now reads the new build.

## Notes

- A/B slots make this safe: an interrupted or bad image falls back to the slot
  that was already running, so a failed update never bricks the device.
- `check_update()` reports whether the device's configured server has a newer
  build; it is independent of a manual sideload like this one.

## Run it

```sh
./build/wired_06_ota_sideload path/to/update.eff
```

## Expected output

The current firmware version, a line per update phase, and a final
"now running firmware <version>" once the device comes back up.
