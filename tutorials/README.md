# Efference SDK tutorials (Linux, C++)

Small, self-contained programs that each demonstrate one part of the SDK. Every
tutorial lives under `tutorials/<topic>/<language>/` and holds its sources (named
after the topic, e.g. `serial_number.cpp`), a `README.md`, and a `CMakeLists.txt`.
They build individually, so `build/` never fills up with binaries you did not ask
for.

## Build the SDK first

```sh
./build.sh          # from the repo root
```

This produces `libef` and the `ef-cli` control tool. Tutorials link against it.

## Build and run a tutorial

Each tutorial has its own `build.sh` that locates the SDK build tree for you, so
no `env.sh` or CMake flags are needed. From the tutorial's folder:

```sh
cd tutorials/serial_number/cpp
./build.sh                      # compiles every .cpp here into build/<name>
./build/serial_number           # the binary is named after the source file
```

To build every tutorial at once, from the repo root:

```sh
./build.sh --tutorials
```

Each tutorial is a standalone CMake project that finds the SDK with
`find_package(ef)`. Its `CMakeLists.txt` builds every `.cpp` in the folder into
its own binary named after the file, so a topic can hold more than one program.
To add a tutorial, create `tutorials/<topic>/<language>/` with one or more
descriptively named sources, a `README.md`, a `CMakeLists.txt`, and a `build.sh`
(copy an existing one).

Each README follows the same shape: a title and description, what the tutorial
does, `## Build and run`, and `## Expected output`. Tutorials driven by
command-line flags list them in an `## Options` table; those that take only a
positional argument or two note them inline in the run commands.

## Tutorials

### Wired (USB)

| Topic | Shows |
|---|---|
| [`serial_number`](serial_number/cpp/README.md) | Open over USB, read `DeviceInformation`. |
| [`health_check`](health_check/cpp/README.md) | `health_check()` sweep (`--deep` for the stress tier). |
| [`data_stream`](data_stream/cpp/README.md) | The `grab()` to `retrieve_image()` / `retrieve_imu()` loop. |
| [`wifi_status`](wifi_status/cpp/README.md) | Read the M1's WiFi association from the cached snapshot. |
| [`record_and_download`](record_and_download/cpp/README.md) | Record to the device's eMMC, then pull the `.mcap` over USB. |
| [`ota_sideload`](ota_sideload/cpp/README.md) | Update firmware from a local `.eff` over USB, no network. |

### Wireless (Bluetooth control + WiFi data)

| Topic | Shows |
|---|---|
| [`choosing_a_connection`](choosing_a_connection/cpp/README.md) | Open over USB or Bluetooth (`INPUT_TYPE::STREAM`). |
| [`discover_devices`](discover_devices/cpp/README.md) | `get_device_list(scan_ble=true)`; find USB and BLE M1s. |
| [`wifi_provisioning`](wifi_provisioning/cpp/README.md) | `wifi_add()` + `wifi_select()` over USB or BLE. |
| [`udp_livestream`](udp_livestream/cpp/README.md) | BLE control + WiFi/UDP video and IMU, the fully wireless data path. |
| [`record_and_upload`](record_and_upload/cpp/README.md) | BLE control + WiFi upload; the device uploads its `.mcap` to an S3/HTTP URL. |
| [`ota_update`](ota_update/cpp/README.md) | OTA over the device's own WiFi (control over USB or BLE). |

### OpenCV (display + interop)

Built only where OpenCV and FFmpeg are both present. These use `ef::toCvMat()`
(`include/ef/OpenCV.hpp`) for a zero-copy view of a grabbed frame as a `cv::Mat`.

| Topic | Shows |
|---|---|
| [`opencv_display`](opencv_display/cpp/README.md) | `grab()` to `retrieve_image(VIEW::BGR)` to `ef::toCvMat()` to `cv::imshow()`. |
| [`calibrate_camera`](calibrate_camera/cpp/README.md) | Live checkerboard capture and a Double Sphere intrinsics fit. Needs `libcamera_cal`. |

### Calibration and debugging

| Topic | Shows |
|---|---|
| [`calibrate_imu`](calibrate_imu/cpp/README.md) | Gyro bias + accelerometer ellipsoid estimation. Needs `libimu_cal`. |
| [`grab`](grab/cpp/README.md) | A bare open/grab/retrieve loop for throughput checks and debugging. |
