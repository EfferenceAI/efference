# Prebuilt binaries (closed source)

Unlike the rest of the SDK, the calibration solvers ship as prebuilt,
per-architecture binaries (the source lives under `calibration/` and is fenced
from the public mirror by `copy.bara.sky`):

- `libcamera_cal.so` provides checkerboard detection + wide-angle Double Sphere
  camera calibration. Public API in `include/ef/Checkerboard.h` (`ckb_*`).
- `libimu_cal.so` provides IMU field calibration: gyro zero-bias + accelerometer
  ellipsoid fit. Public API in `include/ef/ImuCalib.h` (`imc_*`).

Drop the matching `.so`s here:

```text
lib/linux-x86_64/libcamera_cal.so    lib/linux-x86_64/libimu_cal.so     # desktop / server Linux
lib/linux-aarch64/libcamera_cal.so   lib/linux-aarch64/libimu_cal.so    # arm64 Linux
```

CMake picks the directory from `CMAKE_SYSTEM_PROCESSOR` at configure time; when a
binary is missing, everything that needs it is skipped and the rest of the SDK
builds as usual.

## How these are built

The camera/IMU solver **source lives in a separate private repo**
(`EfferenceAI/calibration`), pulled in here as the `sdk/linux/calibration` submodule.
The **publish pipeline builds them fresh** from that submodule for every shipped
arch (host + aarch64) right before Copybara mirrors the tree, so the public repo
always carries binaries that match the release's pinned calibration source, never a
hand-built or stale artifact. See `.github/workflows/publish.yml` (and `ci/preview.sh`,
which mirrors it locally). The committed `.so` here are the copies used for local
private-repo development between releases.

To (re)build them by hand on a Linux host (run from `sdk/linux/`, both libs per arch):

```sh
git submodule update --init calibration                 # get the solver source
make -C calibration/cport ship-libs SHIP_TAG=linux-x86_64 LIBOUT=$PWD/lib/linux-x86_64
make -C calibration/cport ship-libs SHIP_TAG=linux-aarch64 \
     CROSS=aarch64-linux-gnu- LIBOUT=$PWD/lib/linux-aarch64
```

The aarch64 build needs an aarch64 cross-toolchain (`apt install gcc-aarch64-linux-gnu`,
or point `CROSS`/`AARCH64_CROSS` at a pinned prefix). The **toolchain sets the shipped
`.so`'s minimum glibc**: an Ubuntu 22.04 `aarch64-linux-gnu-gcc` floors it at `GLIBC_2.35`
(won't run on older arm64 systems with glibc 2.31); an older-glibc toolchain (one targeting
`GLIBC_2.29`) reaches more of them. Pick to match the oldest arm64 target you support.

The build produces **two independent libs**, `libcamera_cal.so` and `libimu_cal.so`,
each **exporting only its own public ABI**: `ckb_version`, `ckb_default_params`,
`ckb_detect_board`, `ckb_calibrate_double_sphere` (camera) and `imc_version`,
`imc_gyro_bias`, `imc_accel_ellipsoid` (IMU). Every internal kernel (`lin_*` linalg,
`ckb_*` helpers) is hidden via a linker version script
(`calibration/cport/{camera_cal,imu_cal}.map`),
and the binaries are fully stripped (no `.symtab`, no debug), so neither `nm -D` nor `nm`
reveals anything but those entry points. The source itself never ships. Keep the `*.map`
export lists in sync with `Checkerboard.h` / `ImuCalib.h`.

**License note:** these binaries are proprietary (closed source) and are NOT
covered by the SDK's open-source license.
