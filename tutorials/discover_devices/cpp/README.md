---
title: "Discover devices"
description: "Enumerate attached USB and nearby Bluetooth M1s without opening one."
---

Lists every M1 the host can see: USB devices immediately, and BLE devices via a
short scan. This is how you find a device's Bluetooth MAC to hand to the other
wireless examples.

## Walkthrough

`get_device_list()` is static, so it enumerates devices without opening one. Pass
`scan_ble = true` and a scan duration in milliseconds to include nearby Bluetooth
devices; USB devices are always returned:

```cpp
std::vector<DeviceProperties> devices = Device::get_device_list(true, scan_ms);
```

Each entry is a `DeviceProperties`. Read `input_type` to tell USB from BLE: USB
entries carry a `device_id` and serial, BLE entries carry a `ble_address` and the
advertised name (this is where you get the MAC for the other examples):

```cpp
for (const auto& d : devices) {
    std::cout << "- " << to_string(d.input_type);
    if (d.input_type == INPUT_TYPE::USB)
        std::cout << "  id=" << d.device_id << "  serial=" << d.serial;
    else
        std::cout << "  ble=" << d.ble_address << "  name=\"" << d.ble_name << "\"";
    std::cout << "\n";
}
```

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/discover_devices          # 3 s BLE scan
./build/discover_devices 6000     # 6 s BLE scan
```

## Expected output

One line per device, then a total.

```text
Scanning USB + BLE (3000 ms)...
- USB  id=0  serial=7f3a1c9d20b4e6f8
- STREAM  ble=60:48:9C:BA:32:F1  name="Efference M1"
2 device(s).
```
