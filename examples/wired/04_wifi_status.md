---
title: "WiFi status (wired)"
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
read disconnected; re-open (or re-run) for a fresh one.

## Run it

```sh
./build/wired_04_wifi_status
```

## Expected output

Either `The M1 is not connected to any WiFi network.` or a line with the SSID,
IP, RSSI, and internet reachability.
