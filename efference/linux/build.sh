#!/usr/bin/env bash
# Build the Efference SDK (Linux host).
#
# Installs the build dependencies on first run (apt-based distros), then
# configures + builds with CMake.
#
#   ./build.sh          configure + build
#   ./build.sh clean    wipe the build dir, then configure + build
set -euo pipefail

# The Linux SDK is a self-contained CMake project rooted in this directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$ROOT/build"

if [ "${1:-}" = "clean" ]; then
    echo ">> removing $BUILD"
    rm -rf "$BUILD"
fi

# --- dependencies -----------------------------------------------------------
need_dep=0
command -v cmake      >/dev/null 2>&1            || need_dep=1
command -v pkg-config >/dev/null 2>&1            || need_dep=1
command -v c++        >/dev/null 2>&1            || need_dep=1
pkg-config --exists libusb-1.0 2>/dev/null       || need_dep=1

if [ "$need_dep" = 1 ]; then
    if command -v apt-get >/dev/null 2>&1; then
        echo ">> installing build dependencies (needs sudo)"
        sudo apt-get update
        sudo apt-get install -y build-essential cmake pkg-config libusb-1.0-0-dev
    else
        echo "!! missing build dependencies and no apt-get on this system." >&2
        echo "!! please install: a C++17 compiler, cmake, pkg-config, libusb-1.0 dev headers" >&2
        exit 1
    fi
fi

# --- build ------------------------------------------------------------------
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

cat <<EOF

>> build complete. binaries:
   $BUILD/example1_serial_number    # prints the device serial number
   $BUILD/ef-info                   # dumps the full DeviceInformation

   plug in the device and run either one. (USB permissions: see
   README.md if you hit ACCESS_DENIED.)
EOF
