#!/usr/bin/env bash
# Build this tutorial against the SDK build tree. Run the SDK build first
# (from the repo root: ./build.sh). Each .cpp here becomes ./build/<name>.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SDK_BUILD="$HERE/../../../sdk/linux/build"
if [ ! -d "$SDK_BUILD" ]; then
    echo "!! SDK not built yet. From the repo root, run: ./build.sh" >&2
    exit 1
fi
cmake -S "$HERE" -B "$HERE/build" -DCMAKE_PREFIX_PATH="$SDK_BUILD"
cmake --build "$HERE/build" -j"$(nproc)"
echo ">> built into $HERE/build/ :"
find "$HERE/build" -maxdepth 1 -type f -executable -printf '   %f\n'
