---
title: "Choosing a connection"
description: "Open the M1 over USB or over Bluetooth LE with one code path."
---

Shows that the transport is a configuration choice, not a different API. With no
argument it opens over USB; given a BLE MAC it opens the control plane over
Bluetooth. Everything after `open()` is identical either way.

## Walkthrough

The transport is selected on `InitParameters`. Leave it at the default for USB,
or set `INPUT_TYPE::STREAM` plus a `ble_address` for Bluetooth control:

```cpp
InitParameters init;

if (argc >= 2) {
    init.input_type  = INPUT_TYPE::STREAM;   // BLE control plane
    init.ble_address = argv[1];
}

Device device;
ERROR_CODE ec = device.open(init);
```

From here the API is the same regardless of link. This example reads the cached
device state and WiFi association, which work identically over USB or BLE:

```cpp
const WirelessConfiguration& wifi = device.get_device_information().wireless;
std::cout << "Connected over "
          << (init.input_type == INPUT_TYPE::STREAM ? "Bluetooth" : "USB")
          << " (state " << to_string(device.get_state()) << ").\n";
```

Over BLE with no WiFi/UDP data plane configured, the device stays `IDLE` (there
is no video to grab); that is expected for a control-only connection.

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/choosing_a_connection                    # USB
./build/choosing_a_connection AA:BB:CC:DD:EE:FF   # Bluetooth
```

Find a device's BLE MAC with the discovery example.

## Expected output

```text
Opening over USB (default)...
Connected over USB (state IDLE). The M1's WiFi is on "MyNetwork" (192.168.1.50).
```
