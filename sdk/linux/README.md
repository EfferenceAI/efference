# Efference SDK: Linux (C++)

The host-side C++ library and tools for the Efference **M1** sensor. A single
`ef::Device` handle exposes the control plane (identity, configuration, health,
WiFi, recording, updates) and a grab-and-retrieve data plane for live video + IMU
(`open` / `grab` / `retrieve`), over USB, Bluetooth LE, or WiFi/UDP.

This README is the one-stop reference for **everything the SDK can do**: the
library API, the command-line tools, the transports, and the common workflows.

---

## Requirements

Linux · C++17 · CMake ≥ 3.16 · pkg-config · `libusb-1.0` · `libcurl`.

## Build

```sh
./build.sh            # configure + build into build/
./build.sh --deps     # first run on Debian/Ubuntu: also apt-get the deps and
                      # install the udev rule (needs sudo), then build
```

Or plain CMake: `cmake -S . -B build && cmake --build build`. Binaries land in
`build/`: `ef-cli`, `efference-viewer` and `ef-decrypt`. The last two are built
only when their optional dependencies are present, and the configure output says
which were found. Tutorials build separately; see
[`tutorials/README.md`](../../tutorials/README.md).

## USB device permissions (required)

The M1 connects over USB, and your user must have permission to open it.
**This is a required one-time setup step.** Without it the SDK cannot access the
device: the CLI prints `INSUFFICIENT_PERMISSIONS` even though the device is
plugged in, and you would have to run everything as root.

Install the USB permission rule once (needs sudo), then unplug and replug the
device:

```sh
sdk/linux/build.sh --udev
```

`./build.sh --deps` installs the same rule as part of first-run setup. Either way
it is done once per machine.

## Running the CLI

The CLI is `sdk/linux/build/ef-cli`. Three ways to run it, all from the repo
root, in increasing order of commitment:

```sh
sdk/linux/build/ef-cli info     # run it by its full path; nothing to set up
source env.sh                   # add ef-cli to PATH for this shell, then: ef-cli info
sdk/linux/build.sh --install    # put ef-cli on PATH for every shell (sudo)
```

`--install` places `ef-cli` and `ef-decrypt` on PATH, and installs the library,
its headers, and the calibration shared objects so `find_package(ef)` works.
`efference-viewer` is **not** installed: run it from `sdk/linux/build/` by path.

It copies binaries rather than linking them, so re-run it after every
rebuild. A plain `./build.sh` writes only to `sdk/linux/build/`, and an `ef-cli`
installed earlier goes on running the older code without complaining. If you are
changing the SDK and rebuilding often, `source env.sh` puts `build/` on PATH
directly and is current as soon as a build finishes.

`source env.sh` also sets `CMAKE_PREFIX_PATH` so the tutorials resolve
`find_package(ef)`, and it works from any directory if you give its full path
(`source /path/to/efference/env.sh`). To source it in every new shell, add that
line to your `~/.bashrc`.

---

## Command-line tools

| Tool | Purpose |
|---|---|
| `ef-cli` | Control CLI, one subcommand per SDK verb (info, health, record, wifi, update, …). |
| `efference-viewer` | Live decoded video window + live IMU values (accel/gyro). |
| `ef-decrypt` | Turn an encrypted recording back into the plain container. Needs libssl-dev. |

> **Tool naming:** the health check is `ef-cli health`; the viewer is `efference-viewer`.

### `ef-cli`: control CLI

```text
ef-cli [--ble <MAC>] [--device <id>] [--password <pw>] [--admin-password <pw>]
       [--udp <host[:port]>] [--verbose] <command> [args]
```

**Global flags**

```text
  --ble <MAC>            connect over Bluetooth LE instead of USB
  --device <id>          pick one of several USB devices
  --password <pw>        control password (factory default 123456)
  --admin-password <pw>  administrator password, when the device has one
  --udp <host[:port]>    with --ble, stream video and IMU to this host (default port 5005)
  --verbose              print the control-plane traffic
```

`--password` is always needed over BLE, and over USB once the device is locked.
`--admin-password` is sent only for the verbs that can demand it, never on others;
`set-admin-password` and `clear-admin-password` take the current password positionally
and ignore the flag.

**Discover and inspect**

```text
  list [--scan-ble]                    discover devices (add --scan-ble for BLE)
  info [--json]                        device snapshot; --json emits machine-readable state
  state                                current DEVICE_STATE
  storage                              free and total space on the recording store
  health [--deep]                      on-device health sweep; --deep adds the stress tier
  time                                 read the device wall clock
  sync-time                            set the device clock from the host
  reboot                               reboot the device
```

**Capture configuration**

```text
  config                               list the enabled modes and codecs
  config set <W> <H> <fps> <codec>     codec: raw|h264|h264hq|h265|h265hq
  calibration [--get]                  show camera and IMU calibration
  calibration --camera --set <fx> <fy> <cx> <cy> <xi> <alpha> <W> <H>
  calibration --camera [--rectify on|off] [--fov-scale <s>]
  calibration --imu --mode <raw|calibrated|both>
  calibration [--camera|--imu] --reset reset to factory default
  location                             read the device location
  location set <lat> <lon> [alt]       store it, and stamp it into every recording
  location clear                       drop it, so recordings carry no location
```

A device has no location until you set one, and it does not update as the device
moves. Until then `location` reports `not set` and recordings carry no location.
Setting one stamps it into every subsequent recording; `location clear` stops that
again.

All of the commands above are idle-only. Camera intrinsics are published as
recording metadata.
`--rectify on` ships rectilinear frames instead of raw fisheye, and `--fov-scale`
sets the rectified field of view (default 1.0, below 1 is wider). The second
`calibration --camera` form changes only the named flags and keeps the stored
intrinsics, so there is no need to retype `--set`.

For IMU, `raw` (the default) carries the calibration as metadata, `calibrated`
applies it on the device, and `both` adds a pre-applied `*_calibrated` stream
alongside the raw one.

**Recordings**

```text
  record start [name] [--location LAT,LON[,ALT]]
  record stop
  record status [name]                 session status, storage and upload progress
  record list                          device recordings, oldest first
  record delete <name>                 returns at once; large files are reclaimed in the background
  download <name> [dest]               pull a recording over USB or BLE
```

A device-local recording writes to eMMC and survives host disconnect. `record status`
says why a completed session ended, for example `complete (disk full)`, and while a
recording or upload runs its duration and byte counts advance each time you ask.
`--location` sets the location for that recording only, whether or not one is
stored. For `download`, `dest` is a file path or a directory to write into,
defaulting to `<name>.mcap`.

**The recordings drive** (device firmware >= v00.09.19). Connecting the device to a
computer also presents the recordings as a removable drive named `EFFERENCE`, with
each session as `<name>.mcap`. No driver or SDK is needed: copy them off with the
file manager.
Nothing is duplicated on the device to do this, so presenting the drive costs no
storage.

Things to know about it:

- **Deleting a file on the drive deletes the recording on the device.** There is no
  recycle bin, and moving a file to the Trash or Recycle Bin is not an undo: the
  recording is removed from the device, and the copy the Trash appears to hold
  disappears the next time the drive is attached. Eject the drive before you
  unplug it, so a delete finishes. Copy anything you want to keep off the drive
  first.
- **You cannot rename files on the drive.** The names are produced from the session
  names, and a rename does not stick. The recording itself is untouched. Copy a
  file off first and rename the copy.
- **A recording still in progress does not appear** until it completes. Recordings
  that finish while the drive is connected appear a few seconds later.
- **The drive follows the USB lock.** `lock on` removes it and `lock off` brings it
  back with no need to unplug. `ef-cli info` reports it as `recordings drive`, and
  `DeviceInformation::recordings_volume_attached` carries the same to an SDK caller,
  and `recordings_volume_files` the number of recordings the drive publishes, which
  `ef-cli info --json` emits under those names. Call `refresh_device_information()`
  first: the drive moves on its own and the snapshot is taken at `open()`. Ejecting it
  from the host also removes it, until the recordings change or you reconnect.
  `recordings_volume_files` counts what the drive publishes, so it can differ from
  `record list`: the drive omits a session still being written, and `record list`
  shows at most 48 entries.

The files on the drive are byte-for-byte the same as `download` produces. If at-rest
encryption is on, they are ciphertext; see [Reading an encrypted
recording](#reading-an-encrypted-recording).

**Uploads**

```text
  upload <name> <url>                  device uploads to a pre-signed URL over WiFi
  upload <name> <url> --resumable      <url> is a resumable-session URI
  stop-upload <name>                   abort the transfer and detach the URL
```

Without `--resumable` the recording goes up as one PUT, and an interrupted transfer
restarts from the first byte. With it, the file goes up in 32 MiB chunks and an
interrupted transfer continues from the byte the server confirms holding. Mint the
session URI yourself; the device never does.

`stop-upload` succeeds whether or not a transfer was running. With a resumable URI the
destination keeps what it already committed, so re-attaching the same URI continues
rather than restarting.

**WiFi**

```text
  wifi add <ssid> [psk [country]] [--band auto|2.4|5]
  wifi select <ssid> [--band auto|2.4|5]
  wifi remove <ssid>                   forget a network, disconnecting it if current
  wifi list                            saved networks, marking the connected one
  wifi scan                            access points in range, strongest first
  wifi status                          current association
```

Quote an SSID containing spaces, and omit the PSK for a hidden prompt. `country` is an
ISO code; leave it off and the device infers the regulatory domain from nearby beacons,
which decides whether channels 12 and 13 and the 5 GHz band are usable at all.

A dual-band AP publishes one SSID per radio, so `--band` chooses between them and `auto`
clears the pin. A band the AP does not offer is refused without disturbing the current
link. `wifi status` reports `connected`, `connecting`, `disconnected` or `auth_failed`,
the last two both meaning there is no link right now, and shows `internet` when the last
upload or time sync got an answer from its destination.

**`connected` means associated, not usable.** A device can finish the WPA handshake and
still hold no address, have no resolver, or reach nothing, and `connected` is true in all
three cases. `wifi_health` is the field to branch on. It is layered, each level requiring
every level below it, and it increases with usability. The test for "this link can carry
traffic" is:

```cpp
wifi_health == WIFI_HEALTH::UNSPECIFIED || wifi_health >= WIFI_HEALTH::UNVERIFIED
```

`UNSPECIFIED` is `0`, so it sorts *below* every real verdict: a bare `>=` test would read
every device on firmware older than this field as unusable. It means the firmware does not
report the field, which asserts nothing either way, so treat it as usable.

| `wifi_health` | meaning |
|---|---|
| `DISCONNECTED` | not associated |
| `NO_ADDRESS` | associated, but holds no address |
| `NO_DNS` | addressed, but no resolver is configured |
| `UNREACHABLE` | the last transfer got no answer |
| `UNVERIFIED` | every layer the device can check locally is good, but nothing has sent traffic to prove the path |
| `OK` | a real transfer got an answer |

`UNVERIFIED` and `OK` are both usable; they differ only in whether anything has proven it
end to end. Both `ef-cli wifi status` and `ef-cli info` print a `NOT USABLE:` line naming
the failing layer. Firmware predating the field reports `UNSPECIFIED`, which asserts
nothing.

`wifi scan` returns `DEVICE_BUSY` during a recording, livestream or update. The other
WiFi verbs are accepted then, but `wifi add` and `wifi select` reassociate and can
interrupt a transfer in progress.

**Updates**

```text
  check-update                         ask the update service what this device should run
  update                               update to whatever the service offers
  update --url <url>                   update from an explicit URL, skipping the service
  update --file <update.eff>           update from a local bundle, pushed over USB
  abort-update                         cancel an update in progress
```

**Access control**

```text
  lock on|off [--session]              lock or unlock the USB control plane
  set-password <new>                   rekey over unlocked USB
  set-password <old> <new>             rekey over BLE, or on locked USB
  set-admin-password <new>             set the administrator password
  set-admin-password <old> <new>       change it
  clear-admin-password <current>       remove it
  forget-ble-bonds                     clear every BLE pairing (USB only)
  factory-reset [--yes]                restore defaults
```

`lock --session` applies to this power session only and re-locks when power is lost.
The administrator password has no factory default; a device without one uses the control
password for the encryption verbs. It must be at least 8 characters, and longer is
better. On an unlocked USB link `set-password <new>` needs no old password, and with
`--admin-password` it is the rescue path for a forgotten control password.

`forget-ble-bonds` clears the **device** side only. Each phone must also forget the device
before it will pair again (on iOS, Settings > Bluetooth > the device > Forget This Device);
until it does, its connection attempts fail. Passwords and every other setting are left
alone.

`factory-reset` destroys the encryption key and removes the administrator password. It also
clears BLE bonds, saved WiFi networks, recordings and operator calibration. Over BLE it
requires the password like any other verb.

**At-rest encryption**

```text
  encryption on|off                    AES-256 encrypt new recordings
  encryption create                    generate the device key
  encryption delete                    show the key and how to destroy it
  encryption delete --confirm <key_id> [--yes]
  key show [--out <file>]              print the key, or write it to a new 0600 file
  key set --in <file> | - | <64-hex>   install a key you already hold
```

Every verb here takes the administrator password when one is set. `encryption on` is
refused with no key. `encryption create` shows the key once, so save it then. Plain
`encryption delete` destroys nothing; the `--confirm` form does, and recordings written
under that key become unreadable. `key set` is refused while a key exists, so delete the
existing one first, and prefer `--in` or `-` over passing the key as an argument.

### `efference-viewer`: live viewer

```text
efference-viewer [--codec raw|h264|h265|h264hq|h265hq] [--h264|--h265] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto] [--headless|--no-window] [--stats]
```
Opens an OpenCV window with the decoded video; a status bar above the video
shows the live frame number and the latest **IMU values** (accel m/s², gyro
rad/s), and the same values are echoed to the console once per second. `--flip on` rotates the image
180° host-side (for an upside-down camera; `auto` decides from the IMU). `Q`,
`Esc`, window-close, or `Ctrl-C` quit. (A 3-D IMU orientation gizmo is on the
[roadmap](#roadmap).)

`--h264` and `--h265` are shorthand for the matching `--codec`. Run
`efference-viewer --help` for the current flag list.

`--stats` adds frame accounting to the status bar: how many frames the device put on
the stream against how many arrived whole. The same numbers are available to an SDK
caller through `get_stream_stats(StreamStats&)`, which separates frames lost in
transit from frames that arrived incomplete, frames the host superseded before
`grab()` took them, and frames held back until the next keyframe. Counts reset when
streaming starts, and the call returns `INVALID_FUNCTION_CALL` before the first
`grab()`. Useful for telling a device-side problem from a host-side one: if
`device_frames` keeps climbing while `received_whole` does not, the loss is on the
wire or in the host, not in the camera.

`--headless` (alias `--no-window`) skips the window and just holds the session,
printing a once-per-second heartbeat. Pair it with `--udp <host>` to make the
device forward video+IMU to a remote host with no local display; `--flip` is a
display-only transform and has no effect on the forwarded stream (the receiver
applies its own).

---

## Transports

| Transport | Control | Video + IMU | How |
|---|---|---|---|
| **USB** | yes | yes, live | default; `open()` over USB |
| **Bluetooth LE** | yes (GATT, password-gated) | n/a (control only) | `--ble <MAC>` / `InitParameters::input_type = STREAM` |
| **WiFi / UDP** | via USB or BLE | yes, live UDP | `--udp <host[:port]>` over USB or `--ble`; device pushes to that host (yours, or a remote) |
| **MCAP replay** | n/a | yes, from file | `InitParameters::input_type = MCAP`, `mcap_path` |

USB isoc video is sized to the negotiated link speed automatically (SuperSpeed
32 KB/interval, high-speed 1 KB); live streaming works at both.

USB control is **one client per cable**: while an SDK app or `ef-cli` holds the
device open, a second USB open is refused with `DEVICE_BUSY` ("in use by another
process"). BLE stays available in parallel, so a long-running USB app does not
lock an operator out of the device. `ef-cli info` reports whether a BLE central
holds the link, which is otherwise only visible as a `DEVICE_BUSY` on a verb that
needs the radio.

**Pairing and the password are independent.** A BLE bond survives disconnect and
reboot, so a phone that has paired once reconnects without pairing again. Every
session still authenticates with the control password, so changing the password
does not drop bonds and dropping bonds does not change the password. If a phone
that paired before will no longer connect and re-pairing from the phone does not
fix it, `forget-ble-bonds` clears them all and every phone pairs again on its
next connect.

**A snapshot is not a subscription.** `get_device_information()` is a cached
accessor: it never talks to the device. The snapshot is taken at `open()` and
retaken by the wifi verbs, so a handle held open across a WiFi drop keeps
reporting the association it saw at `open()`. Call `refresh_device_information()`
to retake it, the same way `poll_fault()` refreshes device state. And when the
transport goes away, the association and the BLE link are cleared to unknown
rather than left at their last value, so a lost device never reads as still
connected. The health status is deliberately kept: it records a sweep that did
happen, and blanking it would report a healthy device as failed.

---

## Access control and at-rest encryption

There are two credentials, and the difference matters.

The **control password** (factory default `123456`, passed as `--password`) covers
everything a camera does day to day: recording, uploading, configuration, WiFi. One
value serves both transports.

The **administrator password** is separate, stored separately, and guards the
encryption key: creating, reading and destroying it, toggling encryption on or off,
and rekeying itself. A field worker holding the control password can run the camera
and never touch the key that decrypts the archive.

**It is optional, and whether it exists IS the policy.** Unlike the control password
there is no factory default for it: a device either has one or it does not.

**It must be at least 8 characters, and longer is better.** The device enforces the
minimum and refuses anything shorter. Treat it like any other credential guarding data at
rest: prefer a long passphrase, and do not reuse the control password.

| | |
|---|---|
| **No administrator password** | Those verbs are served at the control-password tier. You get at-rest encryption without managing a second credential. `ef-cli info` reports the key as `UNPROTECTED`, because anyone holding the control password can read it. |
| **Administrator password set** | The same verbs require it, including for a key that already existed. |

Set one when you want that separation:

```sh
ef-cli set-admin-password <new>        # provisions; no old password, none exists yet
ef-cli set-admin-password <old> <new>  # changes an existing one
ef-cli clear-admin-password <current>  # removes it, back to the control tier
```

Once set, pass `--admin-password <pw>` on any later `key show`, `encryption` verb or
rekey: each `ef-cli` run is a fresh process, so a device with one set answers
`INVALID_PASSWORD` without it. `set-admin-password` and `clear-admin-password` are the
exceptions, since the current password is already one of their arguments.

Changing or removing it requires **holding** it: an administrator grant plus the
current password. There is no physical-access reset. **If it is lost, the only
recovery is `factory-reset`, which destroys the encryption key and every recording
made under it.** Manage it centrally.

The reverse case is cheap: the device accepts an administrator grant in place of the
old control password, so an administrator can reset a *worker* who forgot theirs
without destroying anything. The alternative is `factory-reset`, which takes the
encryption key with it.

From the command line:

```sh
ef-cli --admin-password <admin-pw> set-password <new-worker-pw>
```

It needs an administrator password to be set, so the rescue works on a device that has
one and not on a device that does not. Only the one-argument form takes this path: with
`<old> <new>` the old password is the proof, and the administrator grant would be spent
for nothing.

### The USB lock

BLE always requires the control password. USB is fully open until you lock it:

```sh
ef-cli lock on                       # USB now gates exactly like BLE
ef-cli --password 123456 record list
ef-cli --password 123456 lock off
```

A locked device still reports its identity, and can always be recovered. `ef-cli info`
shows `usb access`, `encryption` and `recordings drive`; `ef-cli config` shows
`encryption` and `rectify` alongside the capture mode.

Authentication is per link, not per command: authenticate once and the rest of
that session is authenticated. USB and BLE authenticate independently of each
other.

A grant lasts for the link. It ends on a cable cycle, on a new authentication
attempt on the same link, when `lock on` is issued, or when the password that
earned it is changed. An administrator grant is single-use and ends on its first.
The SDK re-runs the handshake and retries transparently, so a long-lived `Device`
never sees this; a client speaking the wire protocol itself must be ready to
re-authenticate on `AUTH_REQUIRED`.

The recordings drive follows the lock; see the recordings drive under
[`ef-cli`](#ef-cli-control-cli).

**When you have finished with a locked device, run `lock on` or unplug it.**

If you are going to work on a locked device for a while, open it deliberately
instead of passing a password to every command:

```sh
ef-cli lock off --session      # open until re-locked or the device loses power
ef-cli lock on  --session      # close it again
```

This does not change the stored policy. The device still reports `usb_locked`, so
losing power closes it with nothing to remember, and `ef-cli info` shows
`LOCKED, open for this session` rather than claiming to be either. It is refused
on a device that is not locked, and a plain `lock on` from any transport ends it.

Authenticating never does this implicitly: opening the device is a separate,
deliberate step.

Recordings can be AES-256-GCM encrypted at rest. The device generates a key itself and
returns it exactly once, at creation:

```sh
ef-cli encryption create             # generates the key and PRINTS IT, once
ef-cli encryption on                 # applies to the NEXT recording
ef-decrypt clip1.mcap key.txt clip1.plain.mcap
```

**Save the key when `create` prints it.** The device keeps a working copy, but
nothing ever prints it again except `key show`, and a factory reset destroys it.
Without your copy, recordings made under that key cannot be read by anyone.

### Bringing your own key

`create` gives every device its own key, which means a fleet has as many keys as
units. If you would rather run one archive key across several devices, or hold the key
in your own key management system and hand it to the device, install it instead:

```sh
ef-cli key set --in fleet.key        # 64 hex characters, a 32-byte AES-256 key
ef-cli key set - < fleet.key         # or on stdin
ef-cli key set 0f1e2d3c...           # or positionally, for scripting
```

The reply names the installed key by its `key_id` but does not echo the bytes back —
you supplied them. `key show` still reads it back afterwards, and a key installed this
way is otherwise indistinguishable from a generated one.

Use `key set --in <file>` or `key set -`. Pass the key as an argument only where
the command line is not retained.

Like `create`, `key set` is **refused while a key already exists** and names the
installed one, because replacing a key silently would strand every recording written
under it. Rotation is three deliberate steps, and the third is not optional:

```sh
ef-cli --admin-password <pw> encryption delete --confirm <key_id> --yes
ef-cli --admin-password <pw> key set <new-64-hex>
ef-cli --admin-password <pw> encryption on
```

**`encryption delete` turns encryption OFF, and installing a key does not turn it
back on.** Without the third step the device holds the new key and writes every
subsequent recording in the clear. `key set` warns when it lands in that state, and
`ef-cli info` reports it.

`key set` also requires the device to be **IDLE**, as `encryption delete` does.

Reading it back needs the administrator password, if one is set:

```sh
ef-cli --admin-password <pw> key show                   # print it (pipeable)
ef-cli --admin-password <pw> key show --out device1.key # or a 0600 file, never echoed
```

`--out` refuses to overwrite an existing file rather than truncating it: the file
it would destroy may be the only copy of another device's key.

`key show`, `key set`, `encryption create|delete`, `encryption on|off`,
`set-admin-password` and `clear-admin-password` need the administrator password **when
one is set**, on either transport and whatever the USB lock state. With none set they
are served at the control-password tier, and `ef-cli info` says so rather than letting
you assume the key is protected.

`key set` is gated for a different reason than the others. Its *reply* withholds the
key, but choosing what future recordings are encrypted under decides who can read them,
which is the same argument that gates `encryption off`.


**Setting it works on either transport and needs no cable.** The first
`set-admin-password` takes one argument; changing or removing it afterwards needs the
administrator password itself.

Order does not matter: set it after creating a key and that key is protected too, and
removing it exposes that key again.

```sh
ef-cli encryption create          # allowed while no administrator password is set
ef-cli set-admin-password <new>   # the key above is now behind it
ef-cli encryption on
```

`ef-cli info` reports which credential the key is behind:

```
key access       : control password (no administrator password set; run
                   'set-admin-password' to require one for the encryption key)
```
Once one is set it reads `administrator password`. With a key present and none set it
is more emphatic, because that key is readable by anyone holding the control password:

```
key access       : control password only, key UNPROTECTED (no administrator password is
                   set, so anyone holding the control password can read this key; run
                   'set-admin-password' to protect it)
```

Against firmware older than this feature it instead reads `operator password (this
firmware has no admin tier)`. Do not assume a camera is protected without checking
that line.

**What this protects.** If a device leaves your physical control, rotate the key.

`encryption on` is refused when no key exists, so a session can never be told it
is encrypting while it records in the clear. `ef-cli info` reports the key's ID
(the first four bytes of its SHA-256, never the key), which is how you tell which
of your saved keys opens a given recording; the ID is also written into each
file's header. On a locked device the ID needs the control password like any other
gated field. Whether a key exists at all is always readable. Note the ID is not the key: it needs only the control password, while the
key itself needs the administrator password.

`encryption create` needs the **administrator** password, like `key show` and
`encryption delete`. It returns all 32 bytes of a brand-new key, which is the same
disclosure those two are gated for. There is no chicken-and-egg problem: with no
administrator password set, `create` is served at the
control-password tier, so a device can always make its first key.

`record list` marks each recording `[encrypted]` or `[unencrypted]` by reading
the container magic off the file, so it reports what is on disk rather than the
current setting.

### Reading an encrypted recording

Download it as usual, then decrypt on the host:

```sh
ef-cli download clip1 clip1.enc      # download writes bytes as-is; name it yourself
ef-decrypt clip1.enc device1.key clip1.mcap
```

`download` does not decrypt and does not rename: with no destination it writes
`<name>.mcap` whatever the contents are, so an encrypted recording lands in a
file called `.mcap` that no MCAP reader will open. Give it a destination, as
above, when the recording is encrypted.

The key file is what `key show --out` wrote: 32 raw bytes, or 64 hex characters
with an optional trailing newline. `ef-decrypt` prints the file's `device_id`,
`key_id`, algorithm and chunk count to stderr and the plaintext to the named
output, so it can be read in a pipeline without the summary contaminating it.

Its exit status distinguishes the two failures worth acting on, because a
recording cut short by power loss is a normal outcome and must not look like
corruption:

| Exit | Meaning |
|---|---|
| 0 | Clean: the end marker was present and every chunk verified. |
| 1 | Truncated, or a chunk failed its tag. Everything written before that point is valid and is kept. The warning says which happened: a clean early end reads as truncation (normal after power loss), a failed tag as corrupt or tampered ciphertext. |
| 2 | Unusable input: bad header, wrong key, an algorithm this build does not implement, or I/O failure. No output file is left behind. |

A wrong key is caught at the header rather than as a wall of failing chunks, and
the error names the `key_id` the file actually wants, so you can tell which of
your saved keys it needs.

`ef-decrypt` is built from the device's own format sources, vendored under
`tools/vendor/` so the host reads exactly what the device writes; see the README
there. It needs libcrypto (`apt install libssl-dev`), which
BLE also uses, and is skipped by the build when that is missing.

### Destroying the key

Two commands destroy it, and both are irreversible for every recording written
under it, including copies already uploaded elsewhere.

```sh
ef-cli encryption delete             # shows the key and what will happen; destroys NOTHING
ef-cli encryption delete --confirm <key_id>   # actually destroys it
```

The first form is there so you can still save the key if you need to read old
recordings. The device requires the `key_id` to match, so an SDK caller cannot
destroy a key it never identified either, and it refuses unless the device is
idle: a running session holds the key in memory and would otherwise keep writing
under a key the device had just reported destroyed. Do not start a recording while
`encryption delete` is running.

Both destructive commands prompt for typed confirmation on a terminal. With no
terminal (`ssh box 'ef-cli …'`, cron, redirected stdin) they **refuse** instead of
prompting, so add `--yes` to mean it in a script:

```sh
ef-cli encryption delete --confirm <key_id> --yes
ef-cli factory-reset --yes
```

`factory-reset` restores defaults (password, lock, encryption, wifi, calibration,
capture config, recordings) and **destroys the encryption key too**. It does not
show you the key first, deliberately. Save the key
with `--admin-password <pw> key show --out` before resetting if you still need it.
It puts the control password back to `123456` and **removes** the administrator
password: there is no factory default to restore it to, so the device returns to having
none. Over BLE the reset requires the password like any other verb.

A full reflash destroys the key as surely as a reset does. Your saved copy is the
only thing that survives either.

---

## Common workflows

**Identity / info**
```sh
ef-cli info            # serial, hw rev, firmware, state, camera geometry, IMU, WiFi,
                   #   MACs, and whether a BLE central is holding the link right now
ef-cli info --json     # the machine-readable subset, including "state"
ef-cli state           # CLOSED / IDLE / STREAMING / UPDATING
                   #   STREAMING = moving data: live host stream, recording, upload, OR calibration
                   #   CLOSED = not open, or open but not ready to accept camera work
                   #   To tell a recording from a livestream or an upload, read `record status`
                   #     rather than the state alone
```

`info` reports the same four-value state as `state`, so one call answers both what the
device is and what it is doing. `--json` carries it as `"state"`; the JSON field names
are a contract, so scripts should read that rather than grepping the prose.

In the SDK, `get_state()` is a cached read: free, non-blocking, and kept current by
`open()` and by every call that moves the device. A program that has been idle and wants
to see a change it did not cause calls `refresh_state()` first, which re-reads from the
device and returns an `ERROR_CODE`. On failure the cached value is kept rather than
replaced with a guess, so an error there means what `get_state()` serves is old, not that
the device is gone.

**Diagnostics (health)**
```sh
ef-cli health          # quick sweep: camera, IMU, WiFi, BT, eMMC, USB, services
ef-cli health --deep   # + stress tier (mem/cpu/thermal/sensor-sync)
```

**Capture config & storage**
```sh
ef-cli config                      # current config + the ENABLED modes/codecs you may pick
ef-cli config set 1920 1200 30 h265  # change it (IDLE only, refused while streaming/recording)
ef-cli storage                     # free / total on the recording store (/userdata)
```

**Record to the device, then pull it off (no WiFi needed)**
```sh
ef-cli record start clip1
# ... let it run ...
ef-cli record stop
ef-cli record list                 # clip1  bytes=…  frames=…  dur=…  + remaining storage
ef-cli download clip1 clip1.mcap   # over the USB/BLE control plane
ef-cli download clip1 ~/captures/  # a directory destination writes clip1.mcap into it
# or copy it off the EFFERENCE drive with your file manager, no SDK needed
```

`download`'s destination is a file path, or an existing directory to write
`<name>.mcap` into. With no destination it writes `<name>.mcap` in the current
directory. A destination that cannot be written (missing directory, no
permission, host disk full) fails with `DESTINATION_NOT_WRITABLE`; the recording
on the device is untouched and the pull can simply be re-run.

On a terminal, a long pull reports progress on a line that redraws in place. The
rate is whatever the link is achieving, so it varies with transport:

```
[download   ]  42%  69.3/164.6 MB  16.0 MB/s
```

The line is erased when the transfer ends, leaving `saved <path>` as the result.
Nothing is printed for a recording that finishes in under a moment, and nothing
is printed at all when output is redirected, so a script still reads exactly the
one `saved <path>` line.

An interrupted pull is resumed by re-running the same command, and the progress
it reports opens at the fraction already on disk. Until a run returns success the
file left behind is a partial, not a readable `.mcap`; finish the pull before
handing it to a reader.

**One command at a time.** The USB claim is exclusive for the life of each
process, so a second command issued while the first is still running fails
immediately with `DEVICE_BUSY` from `open()`. There is no minimum delay to
observe between commands: once one returns, the next may be issued at once.
Scripts that background commands, or run two shells against one device, need to
serialize them.

A recording runs until `record stop`, storage reaches the device's reserve
(10% of the recording store; the session then finalizes cleanly on its own), or
power is lost. After a power loss the device recovers the interrupted recording
on its next boot: everything flushed up to ~250 ms before the cut is salvaged
into a normal, listed, downloadable `.mcap` (device firmware >= v00.09.16).
Encrypted recordings interrupted by power loss cannot be repaired on the device
(the ciphertext is opaque to it), so they are listed as
`partial: recovered after power loss` and served as-is: `download` and `upload`
move the partial bytes, and `ef-decrypt` recovers everything up to the cut with
a truncated-tail warning (exit 1); see "Reading an encrypted recording".

`record list` reports the **48 most recent** sessions, oldest first, so the newest
is the last line printed. The reply carries 48 rows; a device holding more keeps
the newest by session mtime rather than whichever the filesystem happened to
return first. `record status <name>` and `download <name>` resolve the name on the
device itself, so they reach sessions older than that window (firmware >=
v00.09.19; against older firmware `record status` falls back to the listed set).

`record status`/`record list` report why a completed session ended
(`RecordingStatus::stopped_reason`: USER, DISK_FULL, WRITE_ERROR, or
INTERRUPTED for a power-loss recovery; firmware >= v00.09.16). DEVICE means the
device ended the session rather than an operator — a shutdown, or a service
restart while it was recording. The file is complete, exactly as with USER; the
value exists so an application does not read those as somebody pressing stop.

**Live view over the USB wire**
```sh
efference-viewer --codec h265           # live H.265 1200p30 in an OpenCV window
```

**Live view over WiFi (fully wireless: BLE control + UDP data)**
```sh
ef-cli wifi add <ssid> <psk> US    # provision once
efference-viewer --ble <MAC> --udp <this-host-ip>
```

**Provision WiFi**
```sh
ef-cli wifi scan                   # see nearby APs (strongest first) before picking one
ef-cli wifi add homenet hunter2 US # save + connect (optional 3rd arg = ISO regdomain code)
ef-cli wifi add cafe pass123       # add more, all saved nets sit at equal priority, so the
                               # device auto-connects to the best AP in range
ef-cli wifi add "My iPhone"        # SSIDs with spaces need quotes; with no <psk> on the
                               # command line the password is typed at a hidden prompt
ef-cli wifi select cafe            # choose a specific saved network (overrides the auto-best pick)
ef-cli wifi status                 # association + IP
ef-cli info                        # device snapshot; reports how many networks are saved
```
Re-adding the network you're already connected to is a no-op (it won't drop the
live link); `select` is how you switch to a different *saved* network on demand.

**Update firmware**
```sh
ef-cli update                                          # ask the update service (device needs WiFi)
ef-cli update --url https://updates.example/latest.eff # device downloads this URL over WiFi
ef-cli update --file path/to/update.eff                # push a local bundle over USB
```
Name the source explicitly. A bare path still works for compatibility, but it is
inferred: the CLI treats it as a local bundle only if it exists on disk, so a
mistyped path is fetched as a URL instead of being reported.

**IMU field calibration**

Build and run the `calibrate_imu` tutorial (`tutorials/calibrate_imu/cpp`; see
[`tutorials/README.md`](../../tutorials/README.md)), then:
```sh
./build/calibrate_imu          # guided: hold still (gyro zero-bias), then tumble through
                               # all faces/edges/corners (accel ellipsoid). Solves + writes
                               # the calibration. A poor run (insufficient tumble coverage,
                               # or the device wasn't still) is REJECTED with guidance;
                               # re-run rather than persist a bad calibration.
ef-cli calibration --get           # read it back (bias, scale-misalignment, noise, dt)
ef-cli calibration --imu --mode calibrated   # device applies a*=M*S*(x-b) to recorded samples
                               #   raw (default): record uncalibrated + params as metadata
                               #   both:          raw + a pre-applied *_calibrated stream
```
Every recording carries the calibration as `efference.ImuCalibration` metadata
regardless of mode. `calibrated` pre-applies it on-device and marks the embedded
params `field-applied` (identity) so a downstream consumer never double-applies.

**Video quality / codec.** Pass a codec to the `grab` tutorial / `efference-viewer`, or set
`InitParameters::compression`. `H265_HQ` / `H264_HQ` request a
perceptually-lossless preset (device encodes at a high fixed-QP); plain
`H265` / `H264` use the device's default rate control; `RAW` is uncompressed
NV12, **USB only**. Raw NV12 runs about 830 Mbit/s at 1200p30, which the WiFi
link cannot carry, so requesting `RAW` with a `udp_host` set is rejected up front
with `INSUFFICIENT_WIFI_BANDWIDTH`. Over WiFi/UDP, use an encoded codec (`H264` /
`H265`); raw is available on the wired USB path, which has the bandwidth for it.

---

## Library API

```cpp
#include <ef/Device.hpp>

int main() {
    ef::Device dev;
    if (dev.open() != ef::ERROR_CODE::SUCCESS) return 1;   // capture starts on first grab()

    ef::Mat image;
    for (;;) {
        ef::ERROR_CODE ec = dev.grab();
        if (ec == ef::ERROR_CODE::GRAB_TIMEOUT) continue;   // no frame yet
        if (ec != ef::ERROR_CODE::SUCCESS) break;
        if (dev.retrieve_image(image) == ef::ERROR_CODE::SUCCESS) {
            // image.getPtr(), image.getWidth()/getHeight(), image.getStep(),
            // image.getTimestamp(), image.getFrameId()  (or image.copyTo(dst))
        }
    }
    dev.close();
    return 0;
}
```

Calls that touch the wire return `ef::ERROR_CODE`; results come back through out
parameters. `get_*` accessors serve values cached at `open()` and never block, so
a long-lived handle refreshes them explicitly: `refresh_device_information()` for
the device snapshot, `poll_fault()` for device state.
Full surface: [`include/ef/Device.hpp`](include/ef/Device.hpp),
[`Core.hpp`](include/ef/Core.hpp), [`Enums.hpp`](include/ef/Enums.hpp),
[`Parameters.hpp`](include/ef/Parameters.hpp).

A call the device does not serve returns `COMMAND_NOT_FOUND`, which usually means the
SDK is newer than the firmware; update the device. Firmware predating that code reports
the same call as `INVALID_FUNCTION_CALL`, or as `UNSUPPORTED` on recording calls — so a
version fallback has to accept both.

**Update the SDK alongside the device, not after it.** An SDK older than the firmware
has no case for `COMMAND_NOT_FOUND` and falls through to a generic per-context result —
`UNKNOWN_FAILURE` on control calls — so a device answer that is precise becomes one that
is not. Nothing breaks, but a diagnosable error stops being diagnosable.

Link with CMake: `find_package(ef REQUIRED)` (point `CMAKE_PREFIX_PATH` at
`sdk/linux/build`, which `env.sh` does) then
`target_link_libraries(app PRIVATE ef::ef)`. Or link `build/libef.a` and add
[`include/`](include/) to your include path. The tutorials under `tutorials/`
are the template.

### Tutorials

Runnable C++ tutorials live in [`tutorials/`](../../tutorials/), one topic per
directory: from a one-line serial read (`serial_number`) to a fully-wireless
BLE-control + WiFi/UDP livestream (`udp_livestream`), plus the `opencv_display`
and `calibrate_camera` OpenCV demos, `calibrate_imu`, and the `grab` debug loop.
Each builds standalone against the SDK via `find_package(ef)`.

Build them after the SDK:

```sh
./build.sh --tutorials              # from the repo root: build every tutorial
tutorials/serial_number/cpp/build.sh   # or build just one, from its folder
```

See [`tutorials/README.md`](../../tutorials/README.md).

---

## Roadmap

- `efference-viewer` 3-D IMU orientation gizmo (visual attitude from the IMU,
  like the older ISOC viewer) alongside the current numeric readout.
- Python bindings (`sdk/python`).
