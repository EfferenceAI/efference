---
title: "Grab loop"
description: "A bare open, grab, and retrieve loop for throughput checks and debugging."
---

The minimal capture loop, kept deliberately unadorned. It opens the device, runs
`grab()` then `retrieve_image()` / `retrieve_imu()`, and prints per-frame stats
and a final frames-per-second summary. Use it to confirm the data plane works, to
measure throughput, or to capture a clip while debugging. For an annotated version
of the same loop, see the `data_stream` tutorial.

## Options

| Flag | Effect |
|---|---|
| `[seconds]` | Run for a fixed duration, then stop (default: until Ctrl-C). |
| `--codec raw\|h264\|h265\|h264hq\|h265hq` | Capture codec (default: device setting). |
| `--record out.mcap` | Tee the stream to an MCAP file. |
| `--ble MAC` | Control over Bluetooth instead of USB. |
| `--password PW` | BLE password (defaults to the factory password). |
| `--udp HOST[:PORT]` | Stream video/IMU over WiFi/UDP to this host (add `--ble` for BLE control). |
| `--flip on\|off\|auto` | Image flip. |
| `--verbose` | Extra transport logging. |

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/grab                                 # capture until Ctrl-C
./build/grab 5                               # capture for 5 seconds
./build/grab 5 --codec h265 --record clip.mcap
```

## Expected output

A line per frame, then a summary.

```text
open: USB, state IDLE
frame 1  1920x1200  ts=1784750273186480170  imu=84 (MOVING)
frame 31  1920x1200  ts=1784750274186451170  imu=924 (MOVING)

83 frames, 2352 imu samples in 3.0 s (27.5 fps)
```
