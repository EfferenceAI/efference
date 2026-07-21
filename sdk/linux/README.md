# Efference SDK: Linux (C++)

The host-side C++ library and tools for the Efference **M1** sensor. A single
`ef::Device` handle exposes the control plane (identity, configuration, health,
WiFi, recording, updates) and a grab-and-retrieve data plane for live video + IMU
(`open` / `grab` / `retrieve`), over USB, Bluetooth LE, or WiFi/UDP.

This README is the one-stop reference for **everything the SDK can do**: the
library API, the command-line tools, the transports, and the common workflows.

---

## Requirements

Linux · C++17 · CMake ≥ 3.16 · pkg-config · `libusb-1.0`.

## Build

```sh
./build.sh            # configure + build into build/
./build.sh --deps     # first run on Debian/Ubuntu: also apt-get the deps and
                      # install the udev rule (needs sudo), then build
```

Or plain CMake: `cmake -S . -B build && cmake --build build`. Binaries land in
`build/`: `ef`, `ef-grab`, `efference-viewer`.

---

## Command-line tools

| Tool | Purpose |
|---|---|
| `ef` | Control CLI, one subcommand per SDK verb (info, health, record, wifi, update, …). |
| `ef-grab` | Live capture demo (the open/grab/retrieve loop); headless frame/IMU counter, optional raw record. |
| `efference-viewer` | Live decoded video window + live IMU values (accel/gyro). |

> **Tool naming:** the health check is `ef health` (a dedicated `efference-diagnostics`
> wrapper is planned, see [Roadmap](#roadmap)); the viewer is `efference-viewer`.

### `ef`: control CLI

```
ef [--ble <MAC>] [--device <id>] [--password <pw>] [--udp <host[:port]>] [--verbose] <command> [args]

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
  check-update                   report whether newer firmware is available
  update [url|file.eff]          download+apply firmware; a local .eff path sideloads over the wire
  abort-update                   cancel an update in progress
  wifi add <ssid> <psk> [country]  save + connect a WiFi network (country = ISO regdomain code)
  wifi select <ssid>               force a specific saved network (overrides the auto-best pick)
  wifi list                        list saved networks (marks the connected one)
  wifi remove <ssid>               forget a saved network (disconnects it if it's the current one)
  wifi status                      current association (connecting / connected / not connected)
  set-password <new>             rekey the BLE password (over USB, no old password)
  set-password <old> <new>       rekey the BLE password (over BLE)
  sync-time                      set the device clock from the host
  time                           read the device wall clock
  location                       read the device's current location (session_meta.json, else default)
  location set <lat> <lon> [alt] persist the device location -> every recording's LocationFix
  reboot                         reboot the device
```

### `ef-grab`: data-plane demo (headless)

```
ef-grab [seconds] [--codec raw|h264|h265|h264hq|h265hq] [--record out.mcap]
        [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto] [--verbose]
```
Runs `open → grab → retrieve_image/retrieve_imu` and prints a frame/IMU/fps
summary. `--record` tees the session to a host `.mcap`. `--udp HOST` (with
`--ble`) streams over WiFi instead of the USB wire. `--flip` rotates the frame
180° host-side (`on`), or decides from the IMU at stream start (`auto`). Use it
when the camera is mounted upside-down (host view only; not on-device recordings).

### `efference-viewer`: live viewer

```
efference-viewer [--codec raw|h264|h265|h264hq|h265hq] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto]
```
Opens an OpenCV window with the decoded video; a status bar above the video
shows the live frame number and the latest **IMU values** (accel m/s², gyro
rad/s), and the same values are echoed to the console once per second. `--flip on` rotates the image
180° host-side (for an upside-down camera; `auto` decides from the IMU). `Q`,
`Esc`, window-close, or `Ctrl-C` quit. (A 3-D IMU orientation gizmo is on the
[roadmap](#roadmap).)

---

## Transports

| Transport | Control | Video + IMU | How |
|---|---|---|---|
| **USB** | ✓ (bulk ep1/ep2) | ✓ live isoc ep3/ep4 | default; `open()` over USB |
| **Bluetooth LE** | ✓ (GATT, password-gated) | n/a (control only) | `--ble <MAC>` / `InitParameters::input_type = STREAM` |
| **WiFi / UDP** | via USB or BLE | ✓ live UDP | `--udp <host[:port]>` with `--ble` (device pushes to your host) |
| **MCAP replay** | n/a | ✓ from file | `InitParameters::input_type = MCAP`, `mcap_path` |

USB isoc video is sized to the negotiated link speed automatically (SuperSpeed
32 KB/interval, high-speed 1 KB); live streaming works at both.

---

## Common workflows

**Identity / info**
```sh
ef info            # serial, hw rev, firmware, camera geometry, IMU, WiFi, MACs
ef state           # CLOSED / IDLE / STREAMING / UPDATING
                   #   STREAMING = moving data: live host stream, recording, upload, OR calibration
```

**Diagnostics (health)**
```sh
ef health          # quick sweep: camera, IMU, WiFi, BT, eMMC, USB, services
ef health --deep   # + stress tier (mem/cpu/thermal/sensor-sync)
```

**Capture config & storage**
```sh
ef config                      # current config + the ENABLED modes/codecs you may pick
ef config set 1920 1080 30 h265  # change it (IDLE only, refused while streaming/recording)
ef storage                     # free / total on the recording store (/userdata)
```

**Record to the device, then pull it off (no WiFi needed)**
```sh
ef record start clip1
# ... let it run ...
ef record stop
ef record list                 # clip1  bytes=…  frames=…  dur=…  + remaining storage
ef download clip1 clip1.mcap   # over the USB/BLE control plane
```

**Live view over the USB wire**
```sh
efference-viewer --codec h265           # live H.265 1080/1200p30 in an OpenCV window
ef-grab 5 --codec h265         # headless: prints frames/fps
```

**Live view over WiFi (fully wireless: BLE control + UDP data)**
```sh
ef wifi add <ssid> <psk> US    # provision once
efference-viewer --ble <MAC> --udp <this-host-ip>
```

**Provision WiFi**
```sh
ef wifi add homenet hunter2 US # save + connect (optional 3rd arg = ISO regdomain code)
ef wifi add cafe pass123       # add more, all saved nets sit at equal priority, so the
                               # device auto-connects to the best AP in range
ef wifi select cafe            # choose a specific saved network (overrides the auto-best pick)
ef wifi status                 # association + IP
ef info                        # device snapshot, also lists every saved network
```
Re-adding the network you're already connected to is a no-op (it won't drop the
live link); `select` is how you switch to a different *saved* network on demand.

**Update firmware**
```sh
ef update https://updates.example/latest.eff   # URL: device downloads over WiFi
ef update path/to/update.eff                    # local file: sideload over the wire
```

**Video quality / codec.** Pass a codec to `ef-grab`/`efference-viewer`, or set
`InitParameters::compression`. `H265_HQ` / `H264_HQ` request a
perceptually-lossless preset (device encodes at a high fixed-QP); plain
`H265` / `H264` use the device's default rate control; `RAW` is uncompressed
NV12 (USB only).

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

Link with CMake (`add_subdirectory` this dir and
`target_link_libraries(app PRIVATE ef::ef)`), or link `build/libef.a` and add
[`include/`](include/) to your include path.

### Examples

Runnable C++ examples live in [`examples/`](examples/) (`wired/` + `wireless/`)
and build with the library into `build/` as `wired_*` / `wireless_*`, from a
one-line serial read to a fully-wireless BLE-control + WiFi/UDP livestream. See
[`examples/README.md`](examples/README.md).

---

## Roadmap

- `efference-diagnostics`, standalone health-check tool (wraps `ef health`).
- `efference-viewer` 3-D IMU orientation gizmo (visual attitude from the IMU,
  like the older ISOC viewer) alongside the current numeric readout.
- Python bindings (`sdk/python`).
