# Efference SDK: Linux (C++)

The host-side C++ library and tools for the Efference **M1** sensor. A single
`ef::Device` handle exposes the control plane (identity, configuration, health,
WiFi, recording, updates) and a grab-and-retrieve data plane for live video + IMU
(`open` / `grab` / `retrieve`), over USB, Bluetooth LE, or WiFi/UDP.

This README is the one-stop reference for **everything the SDK can do**: the
library API, the command-line tools, the transports, and the common workflows.

---

## Requirements

Linux · C++17 · CMake ≥ 3.16 · pkg-config · `libusb-1.0` · `libcurl`.

## Build

```sh
./build.sh            # configure + build into build/
./build.sh --deps     # first run on Debian/Ubuntu: also apt-get the deps and
                      # install the udev rule (needs sudo), then build
```

Or plain CMake: `cmake -S . -B build && cmake --build build`. Binaries land in
`build/`: `ef-cli` and `efference-viewer`. Tutorials build separately; see
[`tutorials/README.md`](../../tutorials/README.md).

## USB device permissions (required)

The M1 connects over USB, and your user must have permission to open it.
**This is a required one-time setup step.** Without it the SDK cannot access the
device: the CLI prints `INSUFFICIENT_PERMISSIONS` even though the device is
plugged in, and you would have to run everything as root.

Install the USB permission rule once (needs sudo), then unplug and replug the
device:

```sh
sdk/linux/build.sh --udev
```

`./build.sh --deps` installs the same rule as part of first-run setup. Either way
it is done once per machine.

## Running the CLI

The CLI is `sdk/linux/build/ef-cli`. Three ways to run it, all from the repo
root, in increasing order of commitment:

```sh
sdk/linux/build/ef-cli info     # run it by its full path; nothing to set up
source env.sh                   # add ef-cli to PATH for this shell, then: ef-cli info
sdk/linux/build.sh --install    # install ef-cli to /usr/local/bin, always on PATH (sudo once)
```

`source env.sh` also sets `CMAKE_PREFIX_PATH` so the tutorials resolve
`find_package(ef)`, and it works from any directory if you give its full path
(`source /path/to/efference/env.sh`). To source it in every new shell, add that
line to your `~/.bashrc`; to skip sourcing entirely, use `--install` above.

---

## Command-line tools

| Tool | Purpose |
|---|---|
| `ef-cli` | Control CLI, one subcommand per SDK verb (info, health, record, wifi, update, …). |
| `efference-viewer` | Live decoded video window + live IMU values (accel/gyro). |

> **Tool naming:** the health check is `ef-cli health`; the viewer is `efference-viewer`.

### `ef-cli`: control CLI

```text
ef-cli [--ble <MAC>] [--device <id>] [--password <pw>] [--udp <host[:port]>] [--verbose] <command> [args]

global flags
  --ble <MAC>            connect over Bluetooth LE instead of USB
  --device <id>          pick one of several USB devices
  --password <pw>        BLE control password (factory default 123456)
  --udp <host[:port]>    with --ble: device streams video+IMU to this host over WiFi/UDP (default port 5005)
  --verbose              print the control-plane traffic

commands
  list [--scan-ble]              discover devices (USB; add BLE with --scan-ble)
  info                           device information snapshot (serial, fw, camera, IMU, WiFi, MACs)
  config                         list the ENABLED capture modes + codecs (+ current config)
  config set <W> <H> <fps> <codec>  set capture config (idle only; codec: raw|h264|h264hq|h265|h265hq)
  calibration [--get]            show camera + IMU calibration
  calibration --camera --set <fx> <fy> <cx> <cy> <xi> <alpha> <W> <H> [--rectify on|off] [--fov-scale <s>]
                                 set camera intrinsics (idle only; published as recording
                                 metadata; --rectify on|off toggles on-device rectification,
                                 default off; when on, recordings ship rectified (rectilinear) frames,
                                 else raw fisheye (double_sphere); --fov-scale sets the rectified
                                 output FOV, default 1.0, <1 wider / >1 zoom)
  calibration --camera [--rectify on|off] [--fov-scale <s>]
                                 toggle rectify without re-typing intrinsics: reads the
                                 current calibration, changes only the named flag(s), resends
                                 (idle only; e.g. `calibration --camera --rectify on`)
  calibration [--camera|--imu] --reset   reset calibration to factory default
  calibration --imu --mode <raw|calibrated|both>   how recordings carry IMU calibration:
                                 raw (default) = params as metadata; calibrated = pre-applied
                                 on device; both = raw plus a pre-applied *_calibrated stream
  storage                        free/total space on the recording store (/userdata)
  state                          current DEVICE_STATE
  health [--deep]                run the on-device health sweep (add --deep for the stress tier)
  record start [name] [--location LAT,LON[,ALT]]
                                 start a device-local (eMMC) recording (survives host disconnect;
                                 --location overrides the LocationFix for this recording only)
  record stop                    stop the current device recording
  record status [name]           session status (+ storage + upload)
  record list                    list device recordings
  record delete <name>           delete a device recording
  download <name> [dest]         pull a recording over USB/BLE (default <name>.mcap)
  upload <name> <url>            device uploads a recording to a pre-signed URL (over WiFi)
  stop-upload <name>             cancel a running upload
  check-update                   ask the update service what this device should run
  update                         update to whatever the service offers
  update --url <url>             update from an explicit URL, skipping the service
  update --file <update.eff>     update from a local bundle, pushed over USB
  abort-update                   cancel an update in progress
  wifi add <ssid> <psk> [country]  save + connect a WiFi network (country = ISO regdomain code)
  wifi select <ssid>               force a specific saved network (overrides the auto-best pick)
  wifi list                        list saved networks (marks the connected one)
  wifi remove <ssid>               forget a saved network (disconnects it if it's the current one)
  wifi scan                        access points in range, strongest first (top 10)
  wifi status                      current association (connecting / connected / not connected)
  set-password <new>             rekey the BLE password (over USB, no old password)
  set-password <old> <new>       rekey the BLE password (over BLE)
  sync-time                      set the device clock from the host
  time                           read the device wall clock
  location                       read the device's current location (session_meta.json, else default)
  location set <lat> <lon> [alt] persist the device location -> every recording's LocationFix
  reboot                         reboot the device
```

### `efference-viewer`: live viewer

```text
efference-viewer [--codec raw|h264|h265|h264hq|h265hq] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto] [--headless]
```
Opens an OpenCV window with the decoded video; a status bar above the video
shows the live frame number and the latest **IMU values** (accel m/s², gyro
rad/s), and the same values are echoed to the console once per second. `--flip on` rotates the image
180° host-side (for an upside-down camera; `auto` decides from the IMU). `Q`,
`Esc`, window-close, or `Ctrl-C` quit. (A 3-D IMU orientation gizmo is on the
[roadmap](#roadmap).)

`--headless` (alias `--no-window`) skips the window and just holds the session,
printing a once-per-second heartbeat. Pair it with `--udp <host>` to make the
device forward video+IMU to a remote host with no local display; `--flip` is a
display-only transform and has no effect on the forwarded stream (the receiver
applies its own).

---

## Transports

| Transport | Control | Video + IMU | How |
|---|---|---|---|
| **USB** | ✓ (bulk ep1/ep2) | ✓ live isoc ep3/ep4 | default; `open()` over USB |
| **Bluetooth LE** | ✓ (GATT, password-gated) | n/a (control only) | `--ble <MAC>` / `InitParameters::input_type = STREAM` |
| **WiFi / UDP** | via USB or BLE | ✓ live UDP | `--udp <host[:port]>` over USB or `--ble`; device pushes to that host (yours, or a remote) |
| **MCAP replay** | n/a | ✓ from file | `InitParameters::input_type = MCAP`, `mcap_path` |

USB isoc video is sized to the negotiated link speed automatically (SuperSpeed
32 KB/interval, high-speed 1 KB); live streaming works at both.

---

## Common workflows

**Identity / info**
```sh
ef-cli info            # serial, hw rev, firmware, camera geometry, IMU, WiFi, MACs
ef-cli state           # CLOSED / IDLE / STREAMING / UPDATING
                   #   STREAMING = moving data: live host stream, recording, upload, OR calibration
```

**Diagnostics (health)**
```sh
ef-cli health          # quick sweep: camera, IMU, WiFi, BT, eMMC, USB, services
ef-cli health --deep   # + stress tier (mem/cpu/thermal/sensor-sync)
```

**Capture config & storage**
```sh
ef-cli config                      # current config + the ENABLED modes/codecs you may pick
ef-cli config set 1920 1200 30 h265  # change it (IDLE only, refused while streaming/recording)
ef-cli storage                     # free / total on the recording store (/userdata)
```

**Record to the device, then pull it off (no WiFi needed)**
```sh
ef-cli record start clip1
# ... let it run ...
ef-cli record stop
ef-cli record list                 # clip1  bytes=…  frames=…  dur=…  + remaining storage
ef-cli download clip1 clip1.mcap   # over the USB/BLE control plane
```

**Live view over the USB wire**
```sh
efference-viewer --codec h265           # live H.265 1200p30 in an OpenCV window
```

**Live view over WiFi (fully wireless: BLE control + UDP data)**
```sh
ef-cli wifi add <ssid> <psk> US    # provision once
efference-viewer --ble <MAC> --udp <this-host-ip>
```

**Provision WiFi**
```sh
ef-cli wifi scan                   # see nearby APs (strongest first) before picking one
ef-cli wifi add homenet hunter2 US # save + connect (optional 3rd arg = ISO regdomain code)
ef-cli wifi add cafe pass123       # add more, all saved nets sit at equal priority, so the
                               # device auto-connects to the best AP in range
ef-cli wifi select cafe            # choose a specific saved network (overrides the auto-best pick)
ef-cli wifi status                 # association + IP
ef-cli info                        # device snapshot, also lists every saved network
```
Re-adding the network you're already connected to is a no-op (it won't drop the
live link); `select` is how you switch to a different *saved* network on demand.

**Update firmware**
```sh
ef-cli update https://updates.example/latest.eff   # URL: device downloads over WiFi
ef-cli update path/to/update.eff                    # local file: sideload over the wire
```

**IMU field calibration**

Build and run the `calibrate_imu` tutorial (`tutorials/calibrate_imu/cpp`; see
[`tutorials/README.md`](../../tutorials/README.md)), then:
```sh
./build/calibrate_imu          # guided: hold still (gyro zero-bias), then tumble through
                               # all faces/edges/corners (accel ellipsoid). Solves + writes
                               # the calibration. A poor run (insufficient tumble coverage,
                               # or the device wasn't still) is REJECTED with guidance;
                               # re-run rather than persist a bad calibration.
ef-cli calibration --get           # read it back (bias, scale-misalignment, noise, dt)
ef-cli calibration --imu --mode calibrated   # device applies a*=M*S*(x-b) to recorded samples
                               #   raw (default): record uncalibrated + params as metadata
                               #   both:          raw + a pre-applied *_calibrated stream
```
Every recording carries the calibration as `efference.ImuCalibration` metadata
regardless of mode. `calibrated` pre-applies it on-device and marks the embedded
params `field-applied` (identity) so a downstream consumer never double-applies.

**Video quality / codec.** Pass a codec to the `grab` tutorial / `efference-viewer`, or set
`InitParameters::compression`. `H265_HQ` / `H264_HQ` request a
perceptually-lossless preset (device encodes at a high fixed-QP); plain
`H265` / `H264` use the device's default rate control; `RAW` is uncompressed
NV12, **USB only**. Raw NV12 runs about 830 Mbit/s at 1200p30, which the WiFi
link cannot carry, so requesting `RAW` with a `udp_host` set is rejected up front
with `INSUFFICIENT_WIFI_BANDWIDTH`. Over WiFi/UDP, use an encoded codec (`H264` /
`H265`); raw is available on the wired USB path, which has the bandwidth for it.

---

## Library API

```cpp
#include <ef/Device.hpp>

int main() {
    ef::Device dev;
    if (dev.open() != ef::ERROR_CODE::SUCCESS) return 1;   // capture starts on first grab()

    ef::Mat image;
    for (;;) {
        ef::ERROR_CODE ec = dev.grab();
        if (ec == ef::ERROR_CODE::GRAB_TIMEOUT) continue;   // no frame yet
        if (ec != ef::ERROR_CODE::SUCCESS) break;
        if (dev.retrieve_image(image) == ef::ERROR_CODE::SUCCESS) {
            // image.getPtr(), image.getWidth()/getHeight(), image.getStep(),
            // image.getTimestamp(), image.getFrameId()  (or image.copyTo(dst))
        }
    }
    dev.close();
    return 0;
}
```

Calls that touch the wire return `ef::ERROR_CODE`; results come back through out
parameters. `get_*` accessors serve values cached at `open()` and never block.
Full surface: [`include/ef/Device.hpp`](include/ef/Device.hpp),
[`Core.hpp`](include/ef/Core.hpp), [`Enums.hpp`](include/ef/Enums.hpp),
[`Parameters.hpp`](include/ef/Parameters.hpp).

Link with CMake: `find_package(ef REQUIRED)` (point `CMAKE_PREFIX_PATH` at
`sdk/linux/build`, which `env.sh` does) then
`target_link_libraries(app PRIVATE ef::ef)`. Or link `build/libef.a` and add
[`include/`](include/) to your include path. The tutorials under `tutorials/`
are the template.

### Tutorials

Runnable C++ tutorials live in [`tutorials/`](../../tutorials/), one topic per
directory: from a one-line serial read (`serial_number`) to a fully-wireless
BLE-control + WiFi/UDP livestream (`udp_livestream`), plus the `opencv_display`
and `calibrate_camera` OpenCV demos, `calibrate_imu`, and the `grab` debug loop.
Each builds standalone against the SDK via `find_package(ef)`.

Build them after the SDK:

```sh
./build.sh --tutorials              # from the repo root: build every tutorial
tutorials/serial_number/cpp/build.sh   # or build just one, from its folder
```

See [`tutorials/README.md`](../../tutorials/README.md).

---

## Roadmap

- `efference-diagnostics`, standalone health-check tool (wraps `ef-cli health`).
- `efference-viewer` 3-D IMU orientation gizmo (visual attitude from the IMU,
  like the older ISOC viewer) alongside the current numeric readout.
- Python bindings (`sdk/python`).
