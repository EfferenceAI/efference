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

Device dev;
if (dev.open(init) != ERROR_CODE::SUCCESS) {
    std::cerr << "open failed\n";
    return 1;
}
```

Run the grab loop and ask `retrieve_image()` for `VIEW::BGR`, which is already
OpenCV's channel order, so no colour conversion happens per frame:

```cpp
ERROR_CODE g = dev.grab();
if (g == ERROR_CODE::GRAB_TIMEOUT) continue;      // non-fatal: keep looping
if (g != ERROR_CODE::SUCCESS) break;
if (dev.retrieve_image(frame, VIEW::BGR) != ERROR_CODE::SUCCESS) continue;
```

`ef::toCvMat()` (from `include/ef/opencv.hpp`) wraps the grabbed `ef::Mat` as a
`cv::Mat` with no copy, so it goes straight to `cv::imshow()`:

```cpp
cv::imshow("efference opencv", ef::toCvMat(frame));   // zero-copy
int key = cv::waitKey(1);
if (key == 27 || key == 'q') break;
```

This example builds only when both OpenCV and FFmpeg are present, because it
decodes H.265 over USB.

## Run it

```sh
./build/opencv_01_opencv_display
```

Press `Esc` or `q` to quit.

## Expected output

A live video window. Once a frame is a `cv::Mat` it drops straight into any
OpenCV pipeline.
