# Efference SDK examples (Linux, C++)

Small, self-contained programs that each demonstrate one part of the SDK. They
build automatically with the SDK: every `.cpp` under `wired/` and `wireless/`
becomes a binary in `build/` named `<group>_<file>` (e.g. `wired_01_serial_number`,
`wireless_04_udp_livestream`).

```sh
cd sdk/linux
./build.sh
./build/wired_01_serial_number
```

Adding a new example is just dropping a `.cpp` in `wired/` or `wireless/` and
re-running the build. CMake picks it up.

Each example has a companion `.md` that explains its concepts; the program names
in the tables below link to them. Those files carry Mintlify frontmatter
(`title` / `description`), so they can be pulled into the hosted docs site as-is.

## Wired (USB)

| Program | Shows |
|---|---|
| [`wired_01_serial_number`](wired/01_serial_number.md)           | Open over USB, read `DeviceInformation`. |
| [`wired_02_health_check`](wired/02_health_check.md)             | `health_check()` sweep (`--deep` for the stress tier). |
| [`wired_03_data_stream`](wired/03_data_stream.md)               | The `grab()` → `retrieve_image()` / `retrieve_imu()` loop. |
| [`wired_04_wifi_status`](wired/04_wifi_status.md)               | Read the M1's WiFi association from the cached wireless snapshot. |
| [`wired_05_record_and_download`](wired/05_record_and_download.md)| Record to the device's eMMC, then pull the `.mcap` over USB. |
| [`wired_06_ota_sideload`](wired/06_ota_sideload.md)             | Update the firmware from a local `.eff` over the USB wire (no network). |

## Wireless (Bluetooth control + WiFi data)

| Program | Shows |
|---|---|
| [`wireless_01_choosing_a_connection`](wireless/01_choosing_a_connection.md) | Open over USB or Bluetooth (`INPUT_TYPE::STREAM`). |
| [`wireless_02_discover_devices`](wireless/02_discover_devices.md)      | `get_device_list(scan_ble=true)`, find USB + BLE M1s (and their BLE MACs). |
| [`wireless_03_wifi_provisioning`](wireless/03_wifi_provisioning.md)     | `wifi_add()` + `wifi_select()` over USB **or** BLE. |
| [`wireless_04_udp_livestream`](wireless/04_udp_livestream.md)        | **BLE control + WiFi/UDP video+IMU**, the fully-wireless data path. |
| [`wireless_05_record_and_upload`](wireless/05_record_and_upload.md)     | **BLE control + WiFi upload**, record on the device, then it uploads the `.mcap` to an S3/HTTP URL. |
| [`wireless_06_ota_update`](wireless/06_ota_update.md)     | **OTA over WiFi**, the device downloads and applies a firmware update over its own WiFi (control over USB or BLE). |

### Wireless quick-start

```sh
# 1. find the device's BLE MAC
./build/wireless_02_discover_devices

# 2. put it on your WiFi (over USB here; append the BLE MAC to do it over BLE)
./build/wireless_03_wifi_provisioning "MySSID" "MyPassword" US

# 3. live video + IMU over WiFi, controlled over Bluetooth
#    (second arg is THIS host's IP on that WiFi network)
./build/wireless_04_udp_livestream <ble_mac> 192.168.1.50

# 3b. or record on the device and have it upload the .mcap over WiFi
#     (URL is a pre-signed S3 link or an http://host:port/path receiver)
./build/wireless_05_record_and_upload <ble_mac> "http://192.168.1.50:8098/upload.mcap"
```

> The control plane (USB or BLE) is always required; WiFi/UDP carries only the
> high-bandwidth video+IMU data. `udp_host` must be reachable by the device. The
> device cannot infer this host's IP over Bluetooth, so you pass it in.

## OpenCV (display + interop)

Built only when OpenCV **and** FFmpeg are both present (`apt install libopencv-dev
libavcodec-dev libavutil-dev libswscale-dev`); skipped otherwise, since the USB
display decodes H.265 with `retrieve_image(VIEW::BGR)`. These use `ef::toCvMat()`
(`include/ef/opencv.hpp`), a zero-copy view of a grabbed `ef::Mat` as a `cv::Mat`:
ask `retrieve_image()` for `VIEW::BGR` and the frame is already in OpenCV's
channel order, with no colour conversion.

| Program | Shows |
|---|---|
| [`opencv_01_opencv_display`](opencv/01_opencv_display.md) | `grab()` → `retrieve_image(VIEW::BGR)` → `ef::toCvMat()` → `cv::imshow()`. |

```sh
./build/opencv_01_opencv_display        # USB
```

Once a frame is a `cv::Mat` it drops straight into any OpenCV pipeline. Press
`Esc` or `q` to quit.
