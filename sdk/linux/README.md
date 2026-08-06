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

global flags
  --ble <MAC>            connect over Bluetooth LE instead of USB
  --device <id>          pick one of several USB devices
  --password <pw>        control password (factory default 123456); always needed on BLE,
                         and on USB once locked
  --admin-password <pw>  administrator password, if one is set on the device. Sent only
                         for the verbs that can demand it -- `key show`, `key set`,
                         `encryption create|delete`, `encryption on|off` -- never on
                         others. `set-admin-password` and `clear-admin-password` take
                         the current password positionally and ignore this flag.
                         Unset by default; there is no factory default (below)
  --udp <host[:port]>    with --ble: device streams video+IMU to this host over WiFi/UDP (default port 5005)
  --verbose              print the control-plane traffic; over BLE, also print
                         phase-by-phase connect timing (`[ble] +NNNN ms <phase>`)

commands
  list [--scan-ble]              discover devices (USB; add BLE with --scan-ble)
  info [--json]                  device information snapshot (serial, fw, camera, IMU, WiFi, MACs);
                                 --json emits only the machine-checkable state, for scripts
  config                         list the ENABLED capture modes + codecs (+ current config)
  config set <W> <H> <fps> <codec>  set capture config (idle only; codec: raw|h264|h264hq|h265|h265hq)
  calibration [--get]            show camera + IMU calibration
  calibration --camera --set <fx> <fy> <cx> <cy> <xi> <alpha> <W> <H> [--rectify on|off] [--fov-scale <s>]
                                 set camera intrinsics (idle only; published as recording
                                 metadata; --rectify on|off toggles on-device rectification,
                                 default off; when on, recordings ship rectified (rectilinear) frames,
                                 else raw fisheye (double_sphere); --fov-scale sets the rectified
                                 output FOV, default 1.0, <1 wider / >1 zoom)
  calibration --camera [--rectify on|off] [--fov-scale <s>]
                                 toggle rectify without re-typing intrinsics: reads the
                                 current calibration, changes only the named flag(s), resends
                                 (idle only; e.g. `calibration --camera --rectify on`)
  calibration [--camera|--imu] --reset   reset calibration to factory default
  calibration --imu --mode <raw|calibrated|both>   how recordings carry IMU calibration:
                                 raw (default) = params as metadata; calibrated = pre-applied
                                 on device; both = raw plus a pre-applied *_calibrated stream
  storage                        free/total space on the recording store (/userdata)
  state                          current DEVICE_STATE
  health [--deep]                run the on-device health sweep (add --deep for the stress tier)
  record start [name] [--location LAT,LON[,ALT]]
                                 start a device-local (eMMC) recording (survives host disconnect;
                                 --location overrides the LocationFix for this recording only)
  record stop                    stop the current device recording
  record status [name]           session status (+ storage + upload); says why a completed
                                 session ended, e.g. "complete (disk full)"
  record list                    list device recordings, oldest first (each marked
                                 [encrypted]/[unencrypted])
  record delete <name>           delete a device recording (returns immediately;
                                 large files are reclaimed in the background)
  download <name> [dest]         pull a recording over USB/BLE; dest is a file path or a
                                 directory to write into (default <name>.mcap)
  upload <name> <url>            device uploads a recording to a pre-signed URL (over WiFi)
  stop-upload <name>             cancel a running upload
  check-update                   ask the update service what this device should run
  update                         update to whatever the service offers
  update --url <url>             update from an explicit URL, skipping the service
  update --file <update.eff>     update from a local bundle, pushed over USB
  abort-update                   cancel an update in progress
  wifi add <ssid> [psk [country]] [--band auto|2.4|5]
                                   save + connect a WiFi network (quote an SSID with spaces;
                                   omit <psk> for a hidden prompt). country is an ISO code;
                                   leave it off and the device reads the regulatory domain
                                   from nearby beacons, which decides whether channels 12-13
                                   and 5 GHz are usable at all.
  wifi select <ssid> [--band auto|2.4|5]
                                   force a specific saved network (overrides the auto-best
                                   pick). A dual-band AP publishes one SSID per radio, so
                                   --band is how you choose between them; auto clears the
                                   pin. A band the AP does not offer is refused without
                                   disturbing the current link.
  wifi list                        list saved networks (marks the connected one)
  wifi remove <ssid>               forget a saved network (disconnects it if it's the current one)
  wifi scan                        access points in range, strongest first (top 10);
                                   a dual-band AP appears once per band
  wifi status                      current association (connected / connecting / disconnected /
                                   auth_failed, the last two both meaning no link right now)
  set-password <new>             rekey the control password (over UNLOCKED USB, no old
                                 password; or on a locked device with --admin-password,
                                 the administrator rescue for a forgotten one)
  set-password <old> <new>       rekey the control password (over BLE, or locked USB)
  set-admin-password <new>       set the administrator password (there is no factory
                                 default; a device without one uses the control password;
                                 minimum 8 characters)
  set-admin-password <old> <new> change it; needs the current one
  clear-admin-password <current> remove it; the encryption verbs return to the control
                                 password [administrator password]
  lock on|off [--session]        lock/unlock the USB control plane (needs the current password;
                                 --session applies to this power session only, and re-locks
                                 when power is lost)
  encryption on|off              AES-256 encrypt new recordings (refused with no key)
                                 [administrator password, when one is set]
  encryption create              generate the device's key; SHOWN ONCE, save it
                                 [administrator password, when one is set]
  encryption delete              show the key and how to destroy it (destroys nothing)
                                 [administrator password, when one is set]
  encryption delete --confirm <key_id> [--yes]
                                 destroy the key; recordings under it become unreadable
                                 [administrator password, when one is set]
  key show [--out <file>]        print the key, or write it to a new 0600 file
                                 [administrator password, when one is set]
  key set --in <file> | - | <64-hex>
                                 install a key you already hold, instead of letting the
                                 device generate one; refused while a key exists.
                                 Prefer --in or - over a positional key
                                 [administrator password, when one is set]
  factory-reset [--yes]          restore defaults; DESTROYS the key and REMOVES the
                                 administrator password. Unauthenticated use is USB-only
  sync-time                      set the device clock from the host
  time                           read the device wall clock
  location                       read the device's current location (the persisted value, else the default)
  location set <lat> <lon> [alt] persist the device location -> every recording's LocationFix
  reboot                         reboot the device
```

### `efference-viewer`: live viewer

```text
efference-viewer [--codec raw|h264|h265|h264hq|h265hq] [--h264|--h265] [--ble MAC] [--password PW] [--udp HOST[:PORT]] [--flip on|off|auto] [--headless|--no-window]
```
Opens an OpenCV window with the decoded video; a status bar above the video
shows the live frame number and the latest **IMU values** (accel m/s², gyro
rad/s), and the same values are echoed to the console once per second. `--flip on` rotates the image
180° host-side (for an upside-down camera; `auto` decides from the IMU). `Q`,
`Esc`, window-close, or `Ctrl-C` quit. (A 3-D IMU orientation gizmo is on the
[roadmap](#roadmap).)

`--h264` and `--h265` are shorthand for the matching `--codec`. Run
`efference-viewer --help` for the current flag list.

`--headless` (alias `--no-window`) skips the window and just holds the session,
printing a once-per-second heartbeat. Pair it with `--udp <host>` to make the
device forward video+IMU to a remote host with no local display; `--flip` is a
display-only transform and has no effect on the forwarded stream (the receiver
applies its own).

---

## Transports

| Transport | Control | Video + IMU | How |
|---|---|---|---|
| **USB** | ✓ | ✓ live | default; `open()` over USB |
| **Bluetooth LE** | ✓ (GATT, password-gated) | n/a (control only) | `--ble <MAC>` / `InitParameters::input_type = STREAM` |
| **WiFi / UDP** | via USB or BLE | ✓ live UDP | `--udp <host[:port]>` over USB or `--ble`; device pushes to that host (yours, or a remote) |
| **MCAP replay** | n/a | ✓ from file | `InitParameters::input_type = MCAP`, `mcap_path` |

USB isoc video is sized to the negotiated link speed automatically (SuperSpeed
32 KB/interval, high-speed 1 KB); live streaming works at both.

USB control is **one client per cable**: while an SDK app or `ef-cli` holds the
device open, a second USB open is refused with `DEVICE_BUSY` ("in use by another
process"). BLE stays available in parallel, so a long-running USB app does not
lock an operator out of the device. `ef-cli info` reports whether a BLE central
holds the link, which is otherwise only visible as a `DEVICE_BUSY` on a verb that
needs the radio. There is deliberately no "is it paired" counterpart: the device
purges the BLE bond on every disconnect, so no lasting pairing state exists to
report.

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

`info`, `state`, `storage` and `factory-reset` answer regardless, so a locked
device still tells you what it is and can always be recovered. `ef-cli info`
shows `usb access` and `encryption`; `ef-cli config` shows `encryption` and
`rectify` alongside the capture mode.

Authentication is per link, not per command: authenticate once and the rest of
that session is authenticated. USB and BLE authenticate independently of each
other.

A grant lasts as long as the link and expires on no clock. It ends when the cable
is cycled or the UDC is rebound, on a new authentication attempt on the same link,
when `lock on` is issued, or when the password that earned it is changed. An
administrator grant additionally ends on its first use, because it is single-use.
The SDK re-runs the handshake and retries transparently, so a long-lived `Device`
never sees this; a client speaking the wire protocol itself must be ready to
re-authenticate on `AUTH_REQUIRED`.

⚠ **When you have finished with a locked device, run `lock on` or unplug it.**

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

⚠ **A key passed as an argument lands in your shell history.** Use `key set --in
<file>` or `key set -` instead when that matters.

Like `create`, `key set` is **refused while a key already exists** and names the
installed one, because replacing a key silently would strand every recording written
under it. Rotation is three deliberate steps, and the third is not optional:

```sh
ef-cli --admin-password <pw> encryption delete --confirm <key_id> --yes
ef-cli --admin-password <pw> key set <new-64-hex>
ef-cli --admin-password <pw> encryption on
```

⚠ **`encryption delete` turns encryption OFF, and installing a key does not turn it
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


⚠ **Setting it works on either transport and needs no cable.** The first
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

**What this protects.** The administrator password is an access control on the
control plane, not on the storage medium. If a device is physically compromised,
rotate the key.

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
show you the key first, deliberately: over USB possession authorises it, and handing a key
to an unauthenticated caller is exactly what destroying it prevents. Save the key
with `--admin-password <pw> key show --out` before resetting if you still need it.
It puts the control password back to `123456` and **removes** the administrator
password: there is no factory default to restore it to, so the device returns to having
none. Over BLE the reset requires the password like any other verb.

The key lives on `/userdata`, so a full reflash destroys it as surely as a reset
does. Your saved copy is the only thing that survives either.

---

## Common workflows

**Identity / info**
```sh
ef-cli info            # serial, hw rev, firmware, state, camera geometry, IMU, WiFi,
                   #   MACs, and whether a BLE central is holding the link right now
ef-cli info --json     # the machine-readable subset, including "state"
ef-cli state           # CLOSED / IDLE / STREAMING / UPDATING
                   #   STREAMING = moving data: live host stream, recording, upload, OR calibration
```

`info` reports the same four-value state as `state`, so one call answers both what the
device is and what it is doing. `--json` carries it as `"state"`; the JSON field names
are a contract, so scripts should read that rather than grepping the prose.

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
```

`download`'s destination is a file path, or an existing directory to write
`<name>.mcap` into. With no destination it writes `<name>.mcap` in the current
directory. A destination that cannot be written (missing directory, no
permission, host disk full) fails with `DESTINATION_NOT_WRITABLE`; the recording
on the device is untouched and the pull can simply be re-run.

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
device itself, so they reach sessions older than that window (firmware >= this
release; against older firmware `record status` falls back to the listed set).

`record status`/`record list` report why a completed session ended
(`RecordingStatus::stopped_reason`: USER, DISK_FULL, WRITE_ERROR, or
INTERRUPTED for a power-loss recovery; firmware >= v00.09.16).

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
