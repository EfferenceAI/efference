---
title: "Calibrate camera"
description: "Live checkerboard capture and a Double Sphere intrinsics fit, with an optional push to the device."
---

Calibrates the M1's camera intrinsics from a checkerboard. It shows the live feed
with detection overlay, captures 40 views automatically as you move the board
through the field of view, fits a Double Sphere model (with `xi` fixed at 0), and
prints the intrinsics, their 1-sigma uncertainty, and the RMS reprojection error.
It then offers to push the result to the device.

Needs OpenCV and the prebuilt `libcamera_cal.so` for your architecture
(`sdk/linux/lib/<arch>/`). It is skipped on architectures without that library.

## Options

| Flag | Effect |
|---|---|
| `--pattern WxH` | Inner-corner count of the board (default: `11x8`). |
| `--square-size MM` | Physical square size for metric output (default: unit squares). |
| `--ble MAC` | Control over Bluetooth instead of USB. |

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/calibrate_camera                                 # USB, default 11x8 board
./build/calibrate_camera --pattern 9x6 --square-size 25.0
```

For a good fit, move the board across the whole frame (center, all edges and
corners, near and far) and tilt it between captures. Forty identical frontal
views fit poorly. Press `Esc` or `q` to abort.

## Expected output

A live window during capture, then a printed intrinsics line
(`fx fy cx cy xi alpha`) with uncertainties and RMS, and a `[y/N]` prompt to push
the calibration to the device.
