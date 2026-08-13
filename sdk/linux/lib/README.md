# Prebuilt binaries (closed source)

Unlike the rest of the SDK, the calibration solvers ship as prebuilt,
per-architecture binaries. The solver source is not distributed.

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

## Requirements

The binaries are built per release for every supported architecture, so they always
match the release they ship with.

They need **glibc 2.29 or newer**. Check a host with:

```sh
ldd --version
objdump -T lib/linux-aarch64/libcamera_cal.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1
```

On an older host the libraries will not load, and CMake skips everything that needs
them rather than failing the build.

The build produces **two independent libs**, `libcamera_cal.so` and `libimu_cal.so`,
each **exporting only its own public ABI**: `ckb_version`, `ckb_default_params`,
`ckb_detect_board`, `ckb_calibrate_double_sphere` (camera) and `imc_version`,
`imc_gyro_bias`, `imc_accel_ellipsoid` (IMU). Every internal kernel (`lin_*` linalg,
`ckb_*` helpers) is not exported
and the binaries are stripped. Keep the export lists in sync with
`Checkerboard.h` / `ImuCalib.h`.

**License note:** these binaries are proprietary (closed source) and are NOT
covered by the SDK's open-source license.
