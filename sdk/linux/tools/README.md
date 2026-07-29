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

Extract the raw video elementary stream from a recording. It reads a plain
container, so run an encrypted recording through `ef-decrypt` first. The output is
the coded bitstream, not a container, so name it to match how the recording was
made:

```sh
python3 tools/mcap_to_video.py recording.mcap out.h265
python3 tools/mcap_to_video.py recording.mcap out.h264
```

Play it directly with `ffplay out.h265`, or wrap it in a container first if you
need one (`ffmpeg -i out.h265 -c copy out.mp4`).

H.265 does not always play on Linux out of the box. If it will not, record `h264`
instead (`ef-cli config set <W> <H> <fps> h264`) rather than fighting the decoder.
