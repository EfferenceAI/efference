# Source this from the repo root to use the SDK in the current shell:
#
#   source ./env.sh
#
# It puts ef-cli on PATH and sets CMAKE_PREFIX_PATH so tutorials resolve
# find_package(ef) against the in-repo build tree. This affects only the current
# shell and is undone when the shell closes. To make it permanent, add the line
# above to your ~/.bashrc (or ~/.zshrc), or install ef-cli with ./build.sh --install.

_ef_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
_ef_build="$_ef_root/sdk/linux/build"

if [ ! -d "$_ef_build" ]; then
    echo "efference: build tree not found at $_ef_build. Run ./build.sh first." >&2
else
    case ":$PATH:" in
        *":$_ef_build:"*) ;;                       # already present
        *) PATH="$_ef_build:$PATH"; export PATH ;;
    esac
    export CMAKE_PREFIX_PATH="$_ef_build${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
    echo "efference: ef-cli on PATH and CMAKE_PREFIX_PATH set for this shell."
fi

unset _ef_root _ef_build
