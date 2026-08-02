---
title: "Calibrate IMU"
description: "Estimate gyro bias and the accelerometer ellipsoid, write them to the device, then record calibrated samples."
---

Calibrates the M1's IMU in two steps: it measures the gyro zero-bias with the
device held still, then fits the accelerometer ellipsoid as you rotate the device
through all orientations. It prints the resulting bias and scale terms.

**This writes the fit to the device.** The values persist and apply from the next
session onward, replacing whatever calibration was stored before. To go back to
the factory values, run `ef-cli calibration --imu --reset`.

Needs the prebuilt `libimu_cal.so` for your architecture
(`sdk/linux/lib/<arch>/`). It is skipped on architectures without that library.

## Options

| Flag | Effect |
|---|---|
| `--ble MAC` | Control over Bluetooth instead of USB. |
| `--password PW` | BLE password (defaults to the factory password). |

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/calibrate_imu                                 # USB
./build/calibrate_imu --ble AA:BB:CC:DD:EE:FF         # Bluetooth control
```

Follow the on-screen prompts: keep the device still for the gyro step, then slowly
rotate it through every orientation for the accelerometer step.

## Expected output

The estimated gyro bias and accelerometer ellipsoid terms. To capture calibrated
samples afterward, record with `ef-cli calibration --imu --mode calibrated`.
