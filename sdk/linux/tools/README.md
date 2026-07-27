# Tools

| | |
|---|---|
| `ef-smoke-test.sh` | exercise the whole control-plane CLI against a live device |
| `mcap_to_video.py` | pull the video out of a recorded `.mcap` |

`ef-cli` and `efference-viewer` are built from `ef.cpp` and `efference_viewer.cpp`
here; both are documented in the [SDK README](../README.md).

---

## ef-smoke-test.sh

Runs every `ef-cli` verb against a connected device and prints each command, its
output, and its exit code, so a run can be read start to finish. Deliberate error
cases are labelled EXPECT-FAIL and assert the specific error code, which is what
catches a verb that fails for a new reason.

```sh
./tools/ef-smoke-test.sh                 # full run
./tools/ef-smoke-test.sh --help          # every flag and env knob
```

Common variations:

```sh
EF=/path/to/ef-cli ./tools/ef-smoke-test.sh      # test a specific binary
EFARGS="--verbose" ./tools/ef-smoke-test.sh      # show the control-plane traffic
./tools/ef-smoke-test.sh --skip-ble              # skip the BLE section
SKIP_WIFI=1 SKIP_DEEP=1 ./tools/ef-smoke-test.sh # faster: no wifi churn, no deep health
```

To exercise wifi provisioning for real, give it a network. Without one it uses a
throwaway SSID that will not associate, which tests the plumbing but not a join:

```sh
./tools/ef-smoke-test.sh --wifi-ssid MyNet --wifi-psk secret --wifi-country US
```

Opt-in sections, off by default because they change device state:

```sh
TEST_LOCK=1 ./tools/ef-smoke-test.sh        # USB lock + key read (self-restoring)
TEST_ENCRYPTION=1 ./tools/ef-smoke-test.sh  # at-rest encryption on/off
TEST_REBOOT=1 ./tools/ef-smoke-test.sh      # reboot the device as the last step
```

What it deliberately does **not** do: apply a firmware update, or run
`factory-reset`. The reset clears wifi credentials, recordings and the encryption
key, none of which the harness can put back.

If a run dies partway through the lock section, later verbs will fail asking for a
password. Clear it with `ef-cli --password <pw> lock off`.

---

## mcap_to_video.py

Extract the raw video elementary stream from a recording. The output is the coded
bitstream, not a container, so name it to match how the recording was made:

```sh
python3 tools/mcap_to_video.py recording.mcap out.h265
python3 tools/mcap_to_video.py recording.mcap out.h264
```

Play it directly with `ffplay out.h265`, or wrap it in a container first if you
need one (`ffmpeg -i out.h265 -c copy out.mp4`).

H.265 does not always play on Linux out of the box. If it will not, record `h264`
instead (`ef-cli config set <W> <H> <fps> h264`) rather than fighting the decoder.
