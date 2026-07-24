---
title: "WiFi provisioning"
description: "Join the M1 to a WiFi network over USB or Bluetooth."
---

Saves a WiFi network to the M1 and selects it. This is the prerequisite for the
WiFi/UDP livestream and for the device uploading recordings on its own.

## Walkthrough

Provisioning works over any control link. Open over USB by default, or set a BLE
address to provision a fresh device with no cable:

```cpp
InitParameters init;
if (argc >= 5) {                          // provision over BLE instead of USB
    init.input_type  = INPUT_TYPE::STREAM;
    init.ble_address = argv[4];
}
Device device;
ERROR_CODE ec = device.open(init);
```

`wifi_add()` stores the network (the country code, for example `US`, unlocks
5 GHz channels), then `wifi_select()` tells the device to join it:

```cpp
ec = device.wifi_add(ssid, psk, country);
...
ec = device.wifi_select(ssid);
```

Association is asynchronous, so the cached snapshot may still show it connecting
right after the call. Confirm later with the WiFi status example:

```cpp
const WirelessConfiguration& w = device.get_device_information().wireless;
if (w.wifi_connected)
    std::cout << "Connected to \"" << w.wifi_ssid << "\" (ip "
              << w.wifi_ip_address << ").\n";
```

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/wifi_provisioning "MySSID" "MyPassword" US
./build/wifi_provisioning "MySSID" "MyPassword" US AA:BB:CC:DD:EE:FF   # over BLE
```

## Expected output

Either a "Connected to ..." line or a "Provisioned ..., association in progress"
line.
