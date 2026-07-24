---
title: "OpenCV display"
description: "Show live M1 frames in an OpenCV window with zero-copy interop."
---

Opens over USB, grabs frames, and shows them in an OpenCV window. It is the
starting point for dropping the M1 into an existing OpenCV pipeline.

## Walkthrough

Open over USB with the defaults:

```cpp
InitParameters init;              // USB, native resolution

Device device;
if (device.open(init) != ERROR_CODE::SUCCESS) {
    std::cerr << "open failed\n";
    return 1;
}
```

Run the grab loop and ask `retrieve_image()` for `VIEW::BGR`, which is already
OpenCV's channel order, so no colour conversion happens per frame:

```cpp
ERROR_CODE g = device.grab();
if (g == ERROR_CODE::GRAB_TIMEOUT) continue;      // non-fatal: keep looping
if (g != ERROR_CODE::SUCCESS) break;
if (device.retrieve_image(frame, VIEW::BGR) != ERROR_CODE::SUCCESS) continue;
```

`ef::toCvMat()` (from `include/ef/OpenCV.hpp`) wraps the grabbed `ef::Mat` as a
`cv::Mat` with no copy, so it goes straight to `cv::imshow()`:

```cpp
cv::imshow("efference opencv", ef::toCvMat(frame));   // zero-copy
int key = cv::waitKey(1);
if (key == 27 || key == 'q') break;
```

This example builds only when both OpenCV and FFmpeg are present, because it
decodes H.265 over USB.

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first to build the SDK)
./build/opencv_display
```

Press `Esc` or `q` to quit.

## Expected output

A live video window.
