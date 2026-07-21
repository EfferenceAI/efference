---
title: "Serial number (wired)"
description: "The smallest SDK program: open the M1 over USB and read its identity."
---

The minimal end-to-end program. It opens the device over USB, reads its cached
identity, and closes. Start here to confirm your build and USB permissions work.

## Walkthrough

Construct a `Device` and open it. A default-constructed `InitParameters` uses
USB, so `open()` with no arguments is the wired path. Every call that touches the
device returns an `ERROR_CODE`, so check it:

```cpp
Device device;

ERROR_CODE status = device.open();   // USB is the default
if (status != ERROR_CODE::SUCCESS) {
    std::cerr << "open failed: " << to_string(status) << "\n";
    return 1;
}
```

`get_device_information()` returns a `DeviceInformation` populated at `open()`. It
is cached, so reading it does not touch the device. Use `to_string()` to print
enum fields like the model:

```cpp
DeviceInformation info = device.get_device_information();
std::cout << "Serial:   " << info.serial << "\n"
          << "Model:    " << to_string(info.model) << "\n"
          << "Firmware: " << info.firmware_version << "\n";
```

Always `close()` when you are done:

```cpp
device.close();
```

## Run it

```sh
./build/wired_01_serial_number
```

## Expected output

```
Serial:   <device serial>
Model:    M1
Firmware: <version int>
```
