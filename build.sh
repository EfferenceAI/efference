#!/usr/bin/env bash
# Efference SDK top-level build. Detects the platform and dispatches to the
# matching SDK under sdk/. All arguments pass through, e.g.:
#
#   ./build.sh            configure + build (installs deps on first run)
#   ./build.sh clean      wipe the build dir, then configure + build
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    Linux)
        exec "$ROOT/sdk/linux/build.sh" "$@"
        ;;
    Darwin)
        echo "!! macOS is not supported yet. The SDK currently targets Linux." >&2
        exit 1
        ;;
    *)
        echo "!! unsupported platform: $(uname -s)" >&2
        exit 1
        ;;
esac
