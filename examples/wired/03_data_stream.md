---
title: "Data stream (wired)"
description: "The core grab and retrieve loop for video and IMU over USB."
---

The heart of the SDK: the capture loop. It pulls a set number of frames over
USB, decoding each one and counting the IMU samples that arrived with it. Every
streaming program is a variation on this loop.

## Walkthrough

Ask for IMU data in `InitParameters`, then open. Over USB the data plane starts
on the first `grab()`, so frames flow right after `open()`:

```cpp
InitParameters init;              // USB, native resolution, H265
init.enable_imu = true;

Device device;
ERROR_CODE status = device.open(init);
```

Each `grab()` waits for and latches one frame. `GRAB_TIMEOUT` is non-fatal, so
`continue` past it rather than treating it as an error; a real error breaks the
loop:

```cpp
while (frames < target) {
    ERROR_CODE g = device.grab();
    if (g == ERROR_CODE::GRAB_TIMEOUT) continue;   // non-fatal: keep looping
    if (g != ERROR_CODE::SUCCESS) {
        std::cerr << "grab failed: " << to_string(g) << "\n";
        break;
    }
```

Retrieve is separate from grab. After a good `grab()`, `retrieve_image()` hands
back the decoded pixels and `retrieve_imu()` hands back every IMU sample captured
since the previous grab:

```cpp
    device.retrieve_image(frame, VIEW::NV12);          // decoded frame (NV12)
    device.retrieve_imu(imu, TIME_REFERENCE::IMAGE);
    imu_total += imu.samples.size();
    ++frames;
}
```

## Run it

```sh
./build/wired_03_data_stream          # 150 frames (about 5 s at 30 fps)
./build/wired_03_data_stream 300      # a specific frame count
```

## Expected output

A progress line every 30 frames with the frame id, resolution, IMU counts, and
motion state, then a summary of total frames and IMU samples.
