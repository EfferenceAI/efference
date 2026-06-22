# Function Guide

## `get_device_information()`

This method returns a `ef.DeviceInformation` object. This is exactly our config file + additional information, and it includes:
* **`serial_number`** *(int)*: The unique integer ID of the connected device.
* **`model`** *(ef.MODEL)*: The specific hardware model (e.g., `ef.MODEL.M1`).
* **`firmware_version`** *(str)*: The current firmware version running on the SoC.
* **`input_type`** *(ef.INPUT_TYPE)*: How the data is currently being streamed (`USB_C`, `WIFI`, `BT`).

For the camera, we have:
* **`.camera.fps`** *(int)*: Current framerate (e.g., 30, 60).
* **`.camera.resolution`** *(ef.RESOLUTION)*: The current resolution setting (e.g., `HD1200`).
* **`camera.width` / `camera.height`** *(int)*: Raw pixel dimensions.
* *Double Sphere Model attributes:* `fx`, `fy` (focal lengths), `cx`, `cy` (principal points), `xi` (first sphere parameter), and `alpha` (second sphere parameter).
* ISP params?

### Encoding (`.encoding`)
* **`camera.codec`** *(ef.CODEC)*: Active compression codec (e.g., `H265`, `H264`, `RAW`, `mjpeg`).
* **`camera.bitrate`** *(float)*: Target bitrate for the hardware encoder.
* **`camera.rate_control`** *(str)*: `CBR` (Constant Bitrate), `VBR` (Variable Bitrate) or `FQP` (Fixed Quantization Parameter).

### IMU Configuration (`.imu`)
* **`.accelerometer_range`** *(int)*: e.g., ±4g or ±8g.
* **`.gyroscope_range`** *(int)*: e.g., ±250 dps.
* **`.sampling_rate`** *(int)*: e.g., 500 or 1000.

### Noise & Calibration (`.imu.calibration`)
* **`.accel_noise_density` / `.gyro_noise_density`** *(float)*
* **`.accel_random_walk` / `.gyro_random_walk`** *(float)*
* **`.device_to_imu_transform`** *(Matrix4x4)*: The precise 3D extrinsic matrix linking the optical center to the physical IMU chip.

what else?

### Data Packing and Formatting (`.packer`)
* **`.format`** *(str)*: Format of the data being recorded (mcap, ...).
* **`.segments`** *(int)*: ms per segment recorded, 0 for continuous stream.

### USB Communications (`.usb`)
* **`.log_level`** *(int)*: Logging level of telemetry sent over this interface (0-7).

emmc, ddr4, soc?

### eMMC Storage (`.emmc`)
* **`.backup`** *(bool)*: Record data to device eMMC storage as backup for livestream.

### Wi-Fi (`.wifi`)
* **`.ssid`** *(str)*: Connected network name.
* **`.ip_address`** *(str)*: Local IP address for wireless streaming/SSH.
* **`.log_level`** *(int)*: Logging level of telemetry sent over this interface (0-7).

### Bluetooth (`.bt`)
* **`.mac_address`** *(str)*: Device BT MAC address.
* **`.paired_state`** *(bool)*: Useful for triggering remote data collection.
* **`.log_level`** *(int)*: Logging level of telemetry sent over this interface (0-7).

what else?

## Health Check


