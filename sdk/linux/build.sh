#!/usr/bin/env bash
# Build the Efference SDK (Linux host).
#
# A plain ./build.sh only configures + builds, it never runs sudo or writes to
# /etc. Pass --deps (or set EF_INSTALL_DEPS=1) to also apt-get the dependencies
# and install the USB udev rule; that is the only mode that needs root.
#
#   ./build.sh          configure + build
#   ./build.sh clean    wipe the build dir, then configure + build
#   ./build.sh --deps   install system deps + udev rule (sudo), then build
set -euo pipefail

# The Linux SDK is a self-contained CMake project rooted in this directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

INSTALL_DEPS="${EF_INSTALL_DEPS:-0}"
for arg in "$@"; do
    case "$arg" in
        clean)  echo ">> removing $BUILD"; rm -rf "$BUILD" ;;
        --deps) INSTALL_DEPS=1 ;;
    esac
done

# --- dependencies -----------------------------------------------------------
# Core (required): a C++17 compiler, cmake, pkg-config, libusb-1.0.
# Optional (features degrade gracefully, the core build still succeeds):
#   * OpenCV                                   -> efference-viewer + opencv examples
#   * FFmpeg (libavcodec/libavutil/libswscale) -> H264/H265 decode in retrieve_image / ef-grab
#   * GLib/GIO (libglib2.0-dev) + OpenSSL      -> BLE (--ble) connections
missing_core=0
command -v cmake      >/dev/null 2>&1      || missing_core=1
command -v pkg-config >/dev/null 2>&1      || missing_core=1
command -v c++        >/dev/null 2>&1      || missing_core=1
pkg-config --exists libusb-1.0 2>/dev/null || missing_core=1

opt_missing=""
pkg-config --exists opencv4 2>/dev/null || pkg-config --exists opencv 2>/dev/null \
    || opt_missing="$opt_missing OpenCV(viewer+opencv-examples)"
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
            libusb-1.0-0-dev libssl-dev libopencv-dev \
            libavcodec-dev libavutil-dev libswscale-dev libglib2.0-dev
    else
        echo "!! --deps needs apt-get, which is not on this system." >&2
        echo "!! install dev headers for: a C++17 compiler, cmake, pkg-config, libusb-1.0" >&2
        echo "!!   (required) and OpenCV, FFmpeg, libglib2.0 + libssl (optional)." >&2
        exit 1
    fi
elif [ "$missing_core" = 1 ]; then
    echo "!! missing core build dependencies (cmake, pkg-config, a C++17 compiler," >&2
    echo "!!   libusb-1.0). Install them, or re-run with --deps on Debian/Ubuntu." >&2
    exit 1
fi
# Skip the note after --deps: the install above just satisfied the optional ones.
[ "$INSTALL_DEPS" != 1 ] && [ -n "$opt_missing" ] && \
    echo ">> note: optional deps missing:$opt_missing (those features are disabled)." >&2

# --- USB device permissions (udev) ------------------------------------------
# Install the libusb access rule for the M1. Only under --deps/EF_INSTALL_DEPS
# (it needs sudo and writes /etc); still honours EF_SKIP_UDEV=1 to opt out.
UDEV_RULE="/etc/udev/rules.d/51-efference.rules"
UDEV_LINE='SUBSYSTEM=="usb", ATTR{idVendor}=="39c5", ATTR{idProduct}=="0001", MODE="0660", TAG+="uaccess"'
if [ "$INSTALL_DEPS" = 1 ] && [ "${EF_SKIP_UDEV:-0}" != 1 ] \
   && [ "$(cat "$UDEV_RULE" 2>/dev/null)" != "$UDEV_LINE" ]; then
    echo ">> installing USB udev rule at $UDEV_RULE (needs sudo)"
    printf '%s\n' "$UDEV_LINE" | sudo tee "$UDEV_RULE" >/dev/null
    sudo udevadm control --reload-rules && sudo udevadm trigger
    echo ">> udev rule installed. Re-plug the device if it is already connected."
fi

# --- build ------------------------------------------------------------------
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

cat <<EOF

>> build complete. binaries in $BUILD/:
   ef                       # control CLI (one subcommand per SDK verb)
   ef-grab                  # live capture demo (open/grab/retrieve)
   efference-viewer         # live video + IMU viewer
   wired_01_serial_number   # + the other <group>_<name> example programs

   Plug in the device and run one. If the device is not accessible, run
   ./build.sh --deps once to install the USB udev rule (or add it yourself).
EOF
