#!/usr/bin/env bash
# Build the Efference SDK (Linux host).
#
# A plain ./build.sh only configures + builds, it never runs sudo or writes
# outside this tree. The steps that need root are opt-in and each say so.
#
#   ./build.sh          configure + build
#   ./build.sh clean    wipe the build dir, then configure + build
#   ./build.sh --deps   install system deps + the USB udev rule (sudo), then build
#   ./build.sh --udev   install just the USB udev rule (sudo), then exit
#   ./build.sh --install install ef-cli to /usr/local (sudo), after building
#   ./build.sh --tutorials build every tutorial (../../tutorials) after the SDK
set -euo pipefail

# The Linux SDK is a self-contained CMake project rooted in this directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

# The USB access rule for the M1 (VID 39c5 / PID 0001). Without it, a non-root
# user cannot open the device and the SDK returns INSUFFICIENT_PERMISSIONS.
UDEV_RULE="/etc/udev/rules.d/51-efference.rules"
UDEV_LINE='SUBSYSTEM=="usb", ATTR{idVendor}=="39c5", ATTR{idProduct}=="0001", MODE="0660", TAG+="uaccess"'

# install_udev: write the rule and reload udev. Needs sudo; honours EF_SKIP_UDEV=1.
install_udev() {
    [ "${EF_SKIP_UDEV:-0}" = 1 ] && return 0
    if [ "$(cat "$UDEV_RULE" 2>/dev/null)" = "$UDEV_LINE" ]; then
        echo ">> USB udev rule already installed at $UDEV_RULE"
        return 0
    fi
    echo ">> installing USB udev rule at $UDEV_RULE (needs sudo)"
    printf '%s\n' "$UDEV_LINE" | sudo tee "$UDEV_RULE" >/dev/null
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo ">> udev rule installed. Unplug and replug the device if it is connected."
}

INSTALL_DEPS="${EF_INSTALL_DEPS:-0}"
INSTALL_CLI=0
BUILD_TUTORIALS=0
for arg in "$@"; do
    case "$arg" in
        clean)       echo ">> removing $BUILD"; rm -rf "$BUILD" ;;
        --deps)      INSTALL_DEPS=1 ;;
        --udev)      install_udev; exit 0 ;;
        --install)   INSTALL_CLI=1 ;;
        --tutorials) BUILD_TUTORIALS=1 ;;
    esac
done

# --- dependencies -----------------------------------------------------------
# Core (required): a C++17 compiler, cmake, pkg-config, libusb-1.0, libcurl.
# Optional (features degrade gracefully, the core build still succeeds):
#   * OpenCV                                   -> efference-viewer + opencv tutorials
#   * FFmpeg (libavcodec/libavutil/libswscale) -> H264/H265 decode in retrieve_image
#   * GLib/GIO (libglib2.0-dev) + OpenSSL      -> BLE (--ble) connections
missing_core=0
command -v cmake      >/dev/null 2>&1      || missing_core=1
command -v pkg-config >/dev/null 2>&1      || missing_core=1
command -v c++        >/dev/null 2>&1      || missing_core=1
pkg-config --exists libusb-1.0 2>/dev/null || missing_core=1
pkg-config --exists libcurl 2>/dev/null    || missing_core=1

opt_missing=""
pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null \
    || opt_missing="$opt_missing OpenCV(viewer+opencv-tutorials)"
pkg-config --exists libavcodec libavutil libswscale 2>/dev/null \
    || opt_missing="$opt_missing FFmpeg(H264/H265-decode)"
# BLE needs GLib/GIO and OpenSSL together (the challenge-response auth uses libssl).
if pkg-config --exists glib-2.0 gio-2.0 2>/dev/null \
   && { pkg-config --exists libssl 2>/dev/null || pkg-config --exists openssl 2>/dev/null; }; then :; \
else opt_missing="$opt_missing GLib/OpenSSL(--ble)"; fi

if [ "$INSTALL_DEPS" = 1 ]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo ">> installing build dependencies (needs sudo)"
        sudo apt-get update
        sudo apt-get install -y build-essential cmake pkg-config \
            libusb-1.0-0-dev libcurl4-openssl-dev libssl-dev libopencv-dev \
            libavcodec-dev libavutil-dev libswscale-dev libglib2.0-dev
    else
        echo "!! --deps needs apt-get, which is not on this system." >&2
        echo "!! install dev headers for: a C++17 compiler, cmake, pkg-config, libusb-1.0," >&2
        echo "!!   libcurl (required) and OpenCV, FFmpeg, libglib2.0 + libssl (optional)." >&2
        exit 1
    fi
elif [ "$missing_core" = 1 ]; then
    echo "!! missing core build dependencies (cmake, pkg-config, a C++17 compiler," >&2
    echo "!!   libusb-1.0, libcurl). Install them, or re-run with --deps on Debian/Ubuntu." >&2
    exit 1
fi
# Skip the note after --deps: the install above just satisfied the optional ones.
[ "$INSTALL_DEPS" != 1 ] && [ -n "$opt_missing" ] && \
    echo ">> note: optional deps missing:$opt_missing (those features are disabled)." >&2

# --- USB device permissions (udev) ------------------------------------------
# --deps also installs the udev rule (it already has sudo in hand). On its own,
# use ./build.sh --udev. Both need root; a plain build never touches /etc.
[ "$INSTALL_DEPS" = 1 ] && install_udev

# --- build ------------------------------------------------------------------
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

# --- optional install -------------------------------------------------------
# Copy ef-cli (and the library + headers) to /usr/local so ef-cli is on PATH
# everywhere. Needs sudo. Skip it to keep everything inside the repo.
if [ "$INSTALL_CLI" = 1 ]; then
    echo ">> installing ef-cli to /usr/local (needs sudo)"
    sudo cmake --install "$BUILD"
fi

# --- optional: build every tutorial -----------------------------------------
# Each tutorial links the SDK we just built. Failures (usually a missing optional
# dep like OpenCV) are reported and skipped rather than aborting the run.
if [ "$BUILD_TUTORIALS" = 1 ]; then
    TUT_DIR="$ROOT/../../tutorials"
    echo ">> building all tutorials in $TUT_DIR"
    built=0 skipped=""
    for d in "$TUT_DIR"/*/cpp; do
        [ -f "$d/CMakeLists.txt" ] || continue
        name="$(basename "$(dirname "$d")")"
        if cmake -S "$d" -B "$d/build" -DCMAKE_PREFIX_PATH="$BUILD" >/dev/null 2>&1 \
           && cmake --build "$d/build" >/dev/null 2>&1; then
            echo "   built  $name"; built=$((built + 1))
        else
            echo "   SKIP   $name (build failed, likely a missing optional dep)"
            skipped="$skipped $name"
        fi
    done
    echo ">> tutorials built: $built; skipped:${skipped:- none}"
fi

cat <<EOF

>> build complete. binaries in $BUILD/:
   ef-cli             the control CLI, one subcommand per SDK verb (info, record, wifi, ...)
   efference-viewer   live decoded video + IMU viewer

   REQUIRED once per machine: give your user permission to open the USB device.
   Without it the CLI prints INSUFFICIENT_PERMISSIONS even with the M1 plugged in.
     sdk/linux/build.sh --udev       install the USB permission rule (sudo), then replug

   Three ways to run the CLI, all from the repo root:
     sdk/linux/build/ef-cli info     run it by its full path; nothing to set up
     source env.sh                   add ef-cli to PATH for this shell only, then: ef-cli info
     sdk/linux/build.sh --install    install ef-cli to /usr/local so it is always on PATH (sudo once)

   Tutorials are standalone example programs under tutorials/:
     sdk/linux/build.sh --tutorials       build all of them now
     tutorials/<topic>/cpp/build.sh       build just one (run from anywhere)
   Then run a tutorial by path, e.g. tutorials/serial_number/cpp/build/serial_number.
   See tutorials/README.md.
EOF
