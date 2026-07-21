<div align="left">

# Efference SDK

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

</div>

The Efference SDK is the host-side interface to the Efference M1. The device
firmware is closed, but everything you need to talk to the M1 from your own
machine lives here: a control plane for device info, health, configuration,
WiFi, recording, and firmware updates, and a grab-and-retrieve data plane for
live video and IMU over USB or WiFi.

One library backs all of it (`ef::Device`), with a command-line tool (`ef`), a
live viewer, and worked examples built on top.

## Quickstart

```sh
cd sdk/linux
./build.sh                      # add --deps on Debian/Ubuntu to install dependencies
./build/ef info                 # with the M1 plugged in over USB
./build/wired_01_serial_number  # or run one of the examples
```

`./build.sh` builds the library, the `ef` CLI, the viewer, and every example
into `sdk/linux/build/`.

## Where to go next

- **`sdk/linux/README.md`** is the SDK reference: build options, the full `ef`
  command set, and the connection model (USB, Bluetooth control, WiFi/UDP data,
  MCAP replay).
- **`examples/`** holds small self-contained programs, one per feature (wired,
  wireless, OpenCV). Start with `examples/README.md`.
- **`proto/ef.proto`** is the wire protocol, for integrating without the C++
  library (for example a mobile app talking to the M1 over Bluetooth).

## License

BSD-3-Clause. See [LICENSE](LICENSE).
