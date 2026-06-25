# Efference SDK — Linux (C++)

The Linux host implementation of the `ef` SDK. Talks to the device's SDK
Endpoint over USB (vendor bulk interface, class `FF`/`EF`/`03`) using the
12-byte-header wire protocol. libusb-1.0 is the only external dependency.

This is a self-contained CMake project. Everything is built from this directory;
output lands in `efference/linux/build/`.

## Build

From this directory, just run the build script — it installs the dependencies on
first run (apt-based distros), then configures and builds:

```sh
cd efference/linux
./build.sh          # configure + build
./build.sh clean    # wipe build/ first, then build
```

Artifacts land in `build/`:
- `build/libef.a` — the SDK library
- `build/ef-info` — full device-info dump (diagnostic)
- `build/example1_serial_number` — the repo's example 1, verbatim

## Run

Plug in the device, then:

```sh
./build/example1_serial_number      # prints the serial number
./build/ef-info                     # dumps the whole DeviceInformation
```

### USB permissions

libusb needs write access to the device node. Install a udev rule once:

```sh
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="39c5", ATTR{idProduct}=="0001", MODE="0660", TAG+="uaccess"' \
  | sudo tee /etc/udev/rules.d/51-efference.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

(Re-plug the device after installing the rule.) Without it, calls fail with
`ACCESS_DENIED` — run the binary with `sudo` as a fallback.

## API

```cpp
#include <ef/device.hpp>

ef::Device device;
ef::InitParameters init;            // .verbose, .device_id
if (device.open(init) != ef::ERROR_CODE::SUCCESS) { /* ... */ }
ef::DeviceInformation info = device.get_device_information();   // throws on failure
device.close();
```

See `include/ef/device.hpp` for the full `DeviceInformation` struct.
