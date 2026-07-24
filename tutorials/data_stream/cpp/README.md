---
title: "Data stream"
description: "The core grab and retrieve loop for video and IMU over USB."
---

The core capture loop you copy from. It pulls a set number of frames over USB,
decoding each one and counting the IMU samples that arrived with it. Every
streaming program is a variation on this loop. For a bare version with CLI flags
for throughput checks and debugging, see the `grab` tutorial.

## Walkthrough

Ask for IMU data in `InitParameters`, then open. Over USB `open()` auto-starts
the data plane, so `grab()` returns frames right away:

```cpp
InitParameters init;              // USB, native resolution, H265
init.enable_imu = true;

Device device;
ERROR_CODE ec = device.open(init);
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

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/data_stream          # 150 frames (about 5 s at 30 fps)
./build/data_stream 300      # a specific frame count
```

## Expected output

A progress line every 30 frames, then a summary.

```text
frame 30  1920x1200  imu +28 (840 total)  motion=MOVING
frame 60  1920x1200  imu +28 (1680 total)  motion=MOVING
frame 90  1920x1200  imu +28 (2520 total)  motion=MOVING
frame 120  1920x1200  imu +28 (3360 total)  motion=MOVING
frame 150  1920x1200  imu +28 (4200 total)  motion=MOVING
captured 150 frames, 4200 IMU samples
```
