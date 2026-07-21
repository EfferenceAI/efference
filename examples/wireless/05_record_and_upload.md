---
title: "Record and upload (wireless)"
description: "Record on the device, then have it upload the .mcap over WiFi."
---

Bluetooth LE carries the control plane while the M1 uploads the finished
recording to your URL over its own WiFi. Nothing streams to this host: you tell
the device where to send the file and poll its progress. This is the offload
path for field devices that are not tethered.

## Walkthrough

Open over BLE and record to the device (see the wired record example for the
recording basics), then stop:

```cpp
RecordingParameters rp;
rp.target = RECORDING_TARGET::DEVICE_LOCAL;
rp.name   = name;
device.enable_recording(rp);
std::this_thread::sleep_for(std::chrono::seconds(secs));
device.disable_recording();
```

Hand the device the destination. `upload_recording()` returns immediately; the M1
does the transfer over WiFi in the background. The URL is a pre-signed AWS S3 URL
or a plain `http://<host>:<port>/<path>` receiver:

```cpp
status = device.upload_recording(name, url);   // WIFI_NOT_CONNECTED if not on WiFi
```

Poll the session's upload state until it leaves `RUNNING`.
`get_recording_status()` reports `upload` (`RUNNING`/`OFF`), the bytes sent and
total, and `last_error`:

```cpp
RecordingStatus rs;
for (int i = 0; i < 120; ++i) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (device.get_recording_status(rs, name) != ERROR_CODE::SUCCESS) continue;
    if (rs.upload != UPLOAD_STATE::RUNNING) break;   // done or failed
    std::cout << "  " << rs.upload_bytes_sent / MB << " / "
              << rs.upload_bytes_total / MB << " MB\n";
}
if (rs.last_error != ERROR_CODE::SUCCESS) { /* upload failed */ }
```

## Run it

```sh
# pre-signed AWS S3 URL
./build/wireless_05_record_and_upload AA:BB:CC:DD:EE:FF "https://bucket.s3.amazonaws.com/key?X-Amz-..."

# a plain HTTP receiver on your network
./build/wireless_05_record_and_upload AA:BB:CC:DD:EE:FF "http://192.168.1.50:8098/upload.mcap"
```

## Expected output

A recording line, upload progress in MB, and a final "upload complete" line, or
the upload error if it failed.
