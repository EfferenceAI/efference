# Tools

| | |
|---|---|
| `mcap_to_video.py` | pull the video out of a recorded `.mcap` |

`ef-cli`, `efference-viewer` and `ef-decrypt` are built from `ef.cpp`,
`efference_viewer.cpp` and `ef_decrypt.c` here; all three are documented in the
[SDK README](../README.md). `ef-decrypt` compiles against the device's own format
sources under [`vendor/`](vendor/README.md); read that before touching them.

---

## mcap_to_video.py

Extract the video from a recording, either as an MP4 or as the raw elementary
stream. It reads a plain container, so run an encrypted recording through
`ef-decrypt` first.

The output extension picks the format:

```sh
python3 tools/mcap_to_video.py recording.mcap out.mp4     # muxed
python3 tools/mcap_to_video.py recording.mcap out.h265    # bare elementary stream
```

The video is copied, not re-encoded, so an h265 recording stays h265 inside the
MP4. The container is what changes.

Use `--format mp4` or `--format raw` to override that. Whether the recording is
H.264 or H.265 is read from the recording itself, so the name you give the
elementary stream is yours to choose.

It needs the `mcap` package (`pip install mcap`), and MP4 output also needs
`ffmpeg` on `PATH`.

| | |
|---|---|
| `--format {auto,mp4,raw}` | override the format implied by the output extension |
| `--fps N` | frame rate written into the MP4 (default `30`) |
| `--fps auto` | measure the rate from the recording instead |
| `--allow-mid-gop` | keep leading frames that arrive before the first parameter set |

A recording carries no frame rate of its own, so MP4 output has to be given one.
The default of 30 matches the shipped configuration; set `--fps` to match if you
recorded at another rate, or use `--fps auto` to measure it from the recording.

By default the converter drops any frames before the first parameter set, since a
decoder handed a mid-GOP start produces garbage rather than an error. Nothing is
lost when the recording starts on a keyframe, and at most one GOP is dropped when
it does not. `--allow-mid-gop` keeps them.

An incomplete recording still converts: everything readable is written, and what
was missing is reported.

H.265 does not always play on Linux out of the box, and the MP4 still holds
H.265. If a player will not take it, record `h264` instead
(`ef-cli config set <W> <H> <fps> h264`) rather than fighting the decoder.
