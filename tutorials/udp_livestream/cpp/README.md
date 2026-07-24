---
title: "UDP livestream"
description: "Live H.265 video and IMU over WiFi, controlled over Bluetooth."
---

The fully wireless data path. Bluetooth LE carries the control plane while the
M1 pushes H.265 video and IMU to this host over WiFi/UDP. The M1 must already be
on WiFi (see provisioning), and this host must be on the same network.

## Walkthrough

This is the split-plane setup: `INPUT_TYPE::STREAM` with a `ble_address` for
control and a `udp_host` for data. `udp_host` is THIS machine's IP as the device
can reach it, since the device cannot infer it over Bluetooth:

```cpp
InitParameters init;
init.input_type  = INPUT_TYPE::STREAM;        // BLE control + WiFi/UDP data
init.ble_address = argv[1];
init.udp_host    = argv[2];                    // where the device sends video
init.compression = COMPRESSION_MODE::H265;
init.enable_imu  = true;
```

The codec must be an encoded one (`H264` or `H265`) on this path. Raw NV12 is
about 830 Mbit/s at 1200p30, far beyond what the WiFi link carries, so opening
with `COMPRESSION_MODE::RAW` and a `udp_host` is rejected with
`INSUFFICIENT_WIFI_BANDWIDTH`. Raw streaming is a wired-USB feature, where the
SuperSpeed link has the bandwidth for it.

Once open, the capture loop is identical to the wired data-stream example. Only
the transport differs; `grab()`, `retrieve_image()`, and `retrieve_imu()` are the
same calls:

```cpp
while (frames < target) {
    ERROR_CODE g = device.grab();
    if (g == ERROR_CODE::GRAB_TIMEOUT) continue;
    if (g != ERROR_CODE::SUCCESS) { std::cerr << "grab: " << to_string(g) << "\n"; break; }
    device.retrieve_image(frame, VIEW::NV12);
    device.retrieve_imu(imu, TIME_REFERENCE::IMAGE);
    ++frames;
}
```

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/udp_livestream <ble_mac> <this_host_ip> [udp_port] [num_frames]
./build/udp_livestream AA:BB:CC:DD:EE:FF 192.168.1.50
```

## Expected output

A progress line every 30 frames, then a summary of frames streamed and IMU
samples received over WiFi.
