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
- `build/ef-config` — list caps / change the session configuration
- `build/ef-stream` — capture the device's MCAP stream to a `.mcap`
- `build/example1_serial_number` — the repo's example 1, verbatim

## Run

Plug in the device, then:

```sh
./build/example1_serial_number      # prints the serial number
./build/ef-info                     # dumps the whole DeviceInformation
./build/ef-config --list            # lists the advertised modes + current selection
./build/ef-config --set width=1280 height=720 fps=30 codec=h264   # change the config
./build/ef-stream out.mcap 10       # records 10s of the MCAP stream to out.mcap
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

### Capabilities + configuration (control plane)

`get_device_information().caps` advertises what the hardware *can* do as a list of
explicit modes — each a `width x height @ fps` with a `binning` label and a
`usable` flag. A mode with `usable == false` is advertised but **not selectable**
(e.g. 60 fps on the IMX415); `configure()` rejects it.

`configure()` applies a (partial) session configuration. Only the fields you set
are sent — an unset field keeps the device's current value. The camera tuple
(`width`/`height`/`fps`/`codec`) is validated on the device against the advertised
modes. Configuration is **IDLE-only** (applied at the next session start): stop any
recording first. A transport failure throws; a device *rejection* is a normal
result with `applied == false` and a `reason`.

Codecs are `h264`, `h265`, and `raw` (uncompressed NV12, no encoder). `raw` is
**streaming-only** — it requires `capture_mode = "usb_sdk"` and is rejected
otherwise; `ef-view` renders it directly with no decoder. (MJPEG is not currently
offered.)

```cpp
ef::DeviceInformation info = device.get_device_information();
for (const auto& m : info.caps.modes)
    if (m.usable) /* offer m.width x m.height @ m.fps to the user */;

ef::Configuration cfg;
cfg.width = 1280; cfg.height = 720; cfg.fps = 30; cfg.codec = "h264";
ef::ConfigureResult r = device.configure(cfg);
if (r.applied)
    /* accepted — config_version == r.config_version, takes effect next session */;
else
    /* rejected — r.reason is the offending key / "camera:not_usable" /
       "camera:not_advertised" / "not_idle". Pick another mode. */;
```

### MCAP streaming (data plane)

`stream_mcap` records the device's MCAP byte stream (its 2nd bulk IN, `ep3`) to a
file. It blocks until the duration elapses, your `should_stop()` returns true, or
the device signals a **terminal** end. It does not throw — the reason comes back
in the result (and an optional `on_end` callback).

```cpp
// Record 10 seconds to out.mcap.
ef::StreamResult r = device.stream_mcap("out.mcap", 10.0);
if (r.end == ef::STREAM_END::STOPPED)
    /* complete .mcap (footer captured) — opens in Foxglove */;
else
    /* r.end is terminal: PRODUCER_RESTART / DROPPED / HOST_GONE.
       out.mcap is a valid truncated PREFIX. The SDK does NOT resume —
       inspect r and decide whether to start a new stream. */;

// Or run until Ctrl-C / a terminal end, with a stop flag + end callback:
std::atomic<bool> stop{false};
device.stream_mcap("out.mcap", [&]{ return stop.load(); },
                   [](const ef::StreamResult& r){ /* notified once at the end */ });
```

Notes:
- **Clean stop is two-phase** so the `.mcap` is complete: the SDK finalizes the
  recording, drains the trailing MCAP footer off `ep3`, then tears the stream
  down. A terminal end yields a footer-less prefix by design.
- **The reader is decoupled from any viewer.** Bytes are persisted to the file at
  line rate; a live viewer must read the file independently (e.g. tail it into
  Foxglove). A slow viewer can therefore never back-pressure the device into a
  ring overrun.
- Pre-M0 firmware (no `ep3`) returns `STREAM_END::NO_STREAM_ENDPOINT`.
