---
title: "WiFi status"
description: "Read the M1's WiFi association from the cached wireless snapshot."
---

Opens over USB and reports whether the M1 is on WiFi, reading the wireless
snapshot with no live round trip. Handy right after provisioning to see if the
device associated.

## Walkthrough

After `open()`, the wireless state lives on the cached `DeviceInformation`.
`get_device_information().wireless` is a `WirelessConfiguration` captured at
`open()`, so reading it is instant:

```cpp
const WirelessConfiguration& wifi = device.get_device_information().wireless;
```

Everything you need is on that struct: whether it is connected, and if so the
SSID, IP, RSSI, and whether the internet is reachable:

```cpp
if (!wifi.wifi_connected)
    std::cout << "The M1 is not connected to any WiFi network.\n";
else
    std::cout << "The M1 is on \"" << wifi.wifi_ssid << "\" (ip "
              << wifi.wifi_ip_address << ", rssi " << wifi.wifi_rssi
              << ", internet " << (wifi.internet_reachable ? "yes" : "no") << ").\n";
```

Association is asynchronous, so just after provisioning the snapshot may still
read disconnected. Call `refresh_device_information()` for a fresh one, then read
`get_device_information()` **again** — it returns by value, so the `wifi` reference
above is a copy of the old snapshot and does not follow the refresh. Re-opening
works too but is the long way round.

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/wifi_status
```

## Expected output

One line: either not connected, or the association details.

```text
The M1 is on "MyNetwork" (ip 192.168.1.50, rssi -47, internet yes).
```
