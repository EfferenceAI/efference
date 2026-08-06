---
title: "Grab stress (session-start)"
description: "Repeat open/grab/close and measure the startup frame cadence to reproduce the 1200p H.265 stream-startup pause."
---

`grab_stress_v3` repeatedly opens a fresh session, captures for a short window
(long enough to cover the startup transient), and closes. For every cycle it
records the `open()` → first-frame latency and the largest inter-frame gap, and
flags any cycle whose startup gap looks like the reported stall — a multi-frame
pause followed by a jump in frame IDs.

It exists to reproduce and characterize the observed behavior: at 1920x1200
H.265, some sessions deliver a few frames, pause for ~1.9 s, skip ~57 frame IDs,
then resume at a steady 30 fps. The harness's host-side view (gap width, frame-ID
skip) lines up against the SDK's own `[ef.diag]` timeline when `--debug` is set.

It changes **no** SDK behavior; `--debug` only sets `InitParameters::verbose = 2`,
which turns on the SDK's opt-in stream diagnostics (see below).

## Options

| Flag | Effect |
|---|---|
| `--cycles N` | Number of open/grab/close cycles (default 10). |
| `--secs S` | Capture window per cycle, seconds (default 3). Startup is what matters. |
| `--gap-ms MS` | Flag a startup gap wider than this (default 100; 30 fps ≈ 33 ms). |
| `--debug` | Turn on the SDK `[ef.diag]` stream diagnostics (`verbose = 2`). |
| `--res 1200\|1080\|svga` | Capture resolution (default 1200 = 1920x1200). |
| `--codec raw\|h264\|h265\|h264hq\|h265hq` | Capture codec (default h265). |
| `--udp HOST[:PORT]` | Stream video/IMU over WiFi/UDP (add `--ble` for BLE control). |
| `--ble MAC` / `--password PW` | Control over Bluetooth. |

## Build and run

```sh
./build.sh                                   # build this tutorial (run repo-root ./build.sh first)
./build/grab_stress_v3                        # 10 cycles, 1200p H.265, 3 s each
./build/grab_stress_v3 --cycles 20 --debug    # + SDK [ef.diag] logging
./build/grab_stress_v3 --secs 5 --gap-ms 200  # flag gaps over 200 ms
```

## Reading the diagnostics

With `--debug`, the SDK writes one `[ef.diag] ...` line per event to **stderr**.
Redirect it to a file and grep the milestones:

```sh
./build/grab_stress_v3 --cycles 5 --debug 2> diag.log
grep -E 'ev=(open_requested|stream_start_requested|first_encoded_packet|first_decoded_frame|first_frame_returned|grab_timeout|frame_gap)' diag.log
```

Key fields:

- `first_encoded_packet` → `first_decoded_frame` → `first_frame_returned`: localizes
  a startup stall to transport vs decode vs delivery.
- `rx ... key=1`: a received IDR/IRAP keyframe. The spacing between keyframe IDs is
  the encoder GOP; if the startup gap equals one GOP, the host is waiting a full
  keyframe interval (drop-until-IDR resync, no PLI back-channel over USB isoc).
- `grab_timeout ... resync=1 frames_dropped=N`: frames were withheld by the
  drop-until-IDR gate while waiting for the next keyframe.
- `frame_gap ... skipped=N`: the device's frame IDs jumped by N.
