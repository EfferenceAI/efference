---
title: "Record and download"
description: "Record to the device's eMMC, then pull the .mcap back over USB."
---

Starts a device-local recording, lets it run briefly, stops it, and downloads
the resulting `.mcap` over the USB control link. No WiFi is involved.

## Walkthrough

Configure a recording that targets the device's own storage.
`RECORDING_TARGET::DEVICE_LOCAL` writes to the M1's eMMC and keeps running even
if the host disconnects. Give it an explicit name so you can address it later (a
duplicate name is rejected, so a re-run needs `ef-cli record delete <name>` first):

```cpp
const std::string name = "example_recording";

RecordingParameters rp;
rp.target = RECORDING_TARGET::DEVICE_LOCAL;
rp.name   = name;

ec = device.enable_recording(rp);
```

Let it run, then stop it. `disable_recording()` ends the device session:

```cpp
std::this_thread::sleep_for(std::chrono::seconds(secs));
device.disable_recording();
```

Pull the file back over the control link. `download_recording()` streams it over
USB (or BLE) in chunks, so no WiFi is needed:

```cpp
const std::string dest = name + ".mcap";
ec = device.download_recording(name, dest);
```

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/record_and_download        # record 3 s, then download
./build/record_and_download 10     # record 10 s
```

## Expected output

```text
recording "example_recording" for 3 s...
downloading "example_recording" -> example_recording.mcap ...
saved example_recording.mcap
```
