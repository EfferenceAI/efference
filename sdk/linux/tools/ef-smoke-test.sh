#!/usr/bin/env bash
# ef-smoke-test.sh: exercise the whole `ef` control-plane CLI against a live
# device (USB) and print every command, its output, and exit code for review.
# Includes intentional-error cases (labeled EXPECT-FAIL) to confirm the new
# granular error codes. Does NOT reboot or apply an OTA update.
#
# Usage:
#   ./tools/ef-smoke-test.sh                 # full run (record cycle + wifi + deep health)
#   EF=/path/to/ef ./tools/ef-smoke-test.sh  # point at a specific ef binary
#   EFARGS="--verbose" ./tools/ef-smoke-test.sh
#   ./tools/ef-smoke-test.sh --wifi-ssid MyNet --wifi-psk secret --wifi-country US
#   ./tools/ef-smoke-test.sh --skip-ble --reboot        # (see --help for all flags)
#   SKIP_WIFI=1 ./tools/ef-smoke-test.sh     # skip the wifi-mutation section
#   SKIP_DEEP=1 ./tools/ef-smoke-test.sh     # skip the deep health sweep
#
# Env knobs:
#   EF               ef-cli binary (default: <script>/../build/ef-cli)
#   EFARGS           extra global flags (e.g. "--verbose", "--device <id>")
#   TEST_WIFI_SSID   throwaway SSID to add/select/remove (default: ef-smoke-fake)
#   TEST_WIFI_PSK    its PSK (default: bogus-password-123)
#   TEST_WIFI_COUNTRY regdomain (default: US)
#   TEST_WIFI_AUTHFAIL set to 1 (with a real --wifi-ssid/--wifi-psk) to also test
#                    wrong-password detection: joins with a bad PSK, expects
#                    auth_failed, then restores the correct PSK (drops WiFi briefly)
#   EFGRAB           grab-tutorial binary (default: built from tutorials/grab/cpp on demand)
#   GRAB_SECS        seconds for the data-plane live-capture smoke (default: 5)
#   GRAB_BASE        base path for the saved capture .mcap/.h265/.mp4/.png (default: ./efsmoke_grab,
#                    overwritten each run, NOT cleaned up)
#   SKIP_WIFI/SKIP_BLE/SKIP_DEEP/SKIP_RECORD/SKIP_GRAB  set to 1 to skip a section
#   TEST_REBOOT      set to 1 to reboot the device as the LAST step (off by default;
#                    OTA update/apply is out of scope, test it separately)
#   TEST_LOCK        set to 1 to exercise the USB lock + key-read verbs (off by
#                    default; a run that dies while locked leaves later verbs
#                    needing --password). factory-reset is never run here: it
#                    clears wifi credentials and recordings the harness cannot
#                    restore.
#   TEST_ENCRYPTION  set to 1 to exercise at-rest encryption: the on/off toggle,
#                    the encrypted/unencrypted marker, and a full decrypt round
#                    trip through ef-decrypt back to a parseable MCAP (off by
#                    default; it records, then deletes what it recorded, and
#                    leaves encryption OFF). The round trip is skipped when
#                    ef-decrypt was not built, which needs libssl-dev.

set -u
export LC_ALL=C   # stable %.2f / strtod formatting so the float greps below don't drift
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EF="${EF:-$HERE/../build/ef-cli}"
# Built next to ef-cli, but only when libcrypto was found, so the encryption
# section checks for it rather than assuming it.
EFDEC="${EFDEC:-$HERE/../build/ef-decrypt}"
EFARGS="${EFARGS:-}"
GARGS="$EFARGS"                          # global flags the helpers use (USB by default; BLE section swaps this)
TIMEOUT="${TIMEOUT:-330}"                # per-command cap (s): covers deep health ~3min; a hung verb is killed so we move on
EFX() { timeout -k 5 "$TIMEOUT" "$EF" "$@"; }   # every ef call goes through this so nothing can stall the run
TEST_BLE_PASSWORD="${TEST_BLE_PASSWORD:-123456}"
# For negative auth tests. Must never equal the real one, hence the pid suffix.
WRONG_PW="ef-smoke-wrong-$$"
TEST_WIFI_SSID="${TEST_WIFI_SSID:-ef-smoke-fake}"
TEST_WIFI_PSK="${TEST_WIFI_PSK:-bogus-password-123}"
TEST_WIFI_COUNTRY="${TEST_WIFI_COUNTRY:-US}"
SESSION="efsmoke$$"          # unique throwaway recording name
PSESSION="efsmokeps$$"       # second throwaway session for the per-session --location test
DLDIR=""                     # download scratch dir; created after the cleanup trap is armed

usage() {
    cat <<EOF
Usage: ef-smoke-test.sh [options]
  --wifi-ssid SSID      real SSID to join (default: throwaway -> only tests connecting/disconnected)
  --wifi-psk PSK        WiFi password (required for a real join). Visible in \`ps\`
                        and your shell history -- prefer --wifi-psk-stdin, or set
                        TEST_WIFI_PSK in the environment.
  --wifi-psk-stdin      read the WiFi password from stdin instead of argv
  --wifi-country CC     regdomain (default $TEST_WIFI_COUNTRY; needed for 5 GHz)
  --ble-password PW     BLE control password (default 123456)
  --grab-secs N         data-plane live-capture seconds (default 5)
  --timeout N           per-command timeout seconds (default $TIMEOUT)
  --ef PATH             ef binary to use
  --efargs "ARGS"       extra global ef flags (e.g. "--verbose")
  --reboot              reboot the device as the final step
  --test-calibration    run the destructive calibration set/reset tests (WIPES the
                        device calibration; snapshots + restores best-effort). Off by
                        default: without it, only a read-only calibration --get runs.
  --test-upload         record + upload to a built-in local receiver, verify it lands
  --upload-host IP      host IP the device reaches over WiFi (required for --test-upload)
  --upload-port N       receiver port (default 8099)
  --open-firewall       ufw-allow the receiver port for the test, auto-reverted on exit
                        (only if not already open; needs sudo)
  --skip-wifi | --skip-ble | --skip-deep | --skip-record | --skip-grab
  -h, --help
All options also accept the matching env var (see header comment).
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --wifi-ssid)    TEST_WIFI_SSID="$2";    shift 2;;
        --wifi-psk)     TEST_WIFI_PSK="$2";     shift 2;;
        --wifi-psk-stdin)
            # -r so a backslash in the PSK survives; no echo.
            IFS= read -r TEST_WIFI_PSK || true
            shift;;
        --wifi-country) TEST_WIFI_COUNTRY="$2"; shift 2;;
        --ble-password) TEST_BLE_PASSWORD="$2"; shift 2;;
        --grab-secs)    GRAB_SECS="$2";         shift 2;;
        --timeout)      TIMEOUT="$2";           shift 2;;
        --ef)           EF="$2";                shift 2;;
        --efargs)       EFARGS="$2";            shift 2;;
        --reboot)       TEST_REBOOT=1;          shift;;
        --test-calibration) TEST_CALIB=1;       shift;;
        --test-upload)  TEST_UPLOAD=1;          shift;;
        --upload-host)  UPLOAD_HOST="$2";       shift 2;;
        --upload-port)  UPLOAD_PORT="$2";       shift 2;;
        --skip-wifi)    SKIP_WIFI=1;            shift;;
        --skip-ble)     SKIP_BLE=1;             shift;;
        --skip-deep)    SKIP_DEEP=1;            shift;;
        --skip-record)  SKIP_RECORD=1;          shift;;
        --skip-grab)    SKIP_GRAB=1;            shift;;
        --open-firewall) OPEN_FIREWALL=1;       shift;;
        -h|--help)      usage; exit 0;;
        *) echo "unknown option: $1" >&2; usage; exit 2;;
    esac
done
GARGS="$EFARGS"              # re-sync after --efargs (helpers target USB via $GARGS)

pass=0 fail=0 xfail_ok=0 xfail_bad=0
FW_OPENED=0 FW_PORT=""       # set if --open-firewall opened a ufw rule we must revert
RXPID="" RXDIR="" UPSESS=""  # upload-test receiver pid / temp dir / session (for cleanup)
# Arg-only knobs default here so `set -u` doesn't abort when a flag is omitted.
UPLOAD_HOST="${UPLOAD_HOST:-}" TEST_UPLOAD="${TEST_UPLOAD:-0}" TEST_REBOOT="${TEST_REBOOT:-0}"
TEST_CALIB="${TEST_CALIB:-0}"       # destructive calibration tests: opt-in (wipes calib)
CAL_SNAP=""                          # snapshot of the real calibration, for restore

c_hdr=$'\033[1;36m'; c_cmd=$'\033[1;33m'; c_ok=$'\033[1;32m'
c_err=$'\033[1;31m'; c_dim=$'\033[2m'; c_off=$'\033[0m'
[ -t 1 ] || { c_hdr= c_cmd= c_ok= c_err= c_dim= c_off=; }

banner() { printf '\n%s========== %s ==========%s\n' "$c_hdr" "$1" "$c_off"; }

# run "<description>" <ef args...>. Expect SUCCESS (rc 0). Uses $GARGS (USB or BLE).
run() {
    local desc="$1"; shift
    printf '\n%s# %s%s\n%s$ ef-cli %s %s%s\n' "$c_dim" "$desc" "$c_off" "$c_cmd" "$GARGS" "$*" "$c_off"
    local out rc
    out="$(EFX $GARGS "$@" 2>&1)"; rc=$?
    printf '%s\n' "$out"
    if [ $rc -eq 0 ]; then printf '%s[exit %d OK]%s\n' "$c_ok" "$rc" "$c_off"; pass=$((pass+1))
    else printf '%s[exit %d UNEXPECTED-FAIL]%s\n' "$c_err" "$rc" "$c_off"; fail=$((fail+1)); fi
}

# xfail "<description>" "<expected code substring>" <ef args...>. Expect FAILURE
xfail() {
    local desc="$1" want="$2"; shift 2
    printf '\n%s# EXPECT-FAIL (%s): %s%s\n%s$ ef-cli %s %s%s\n' \
        "$c_dim" "$want" "$desc" "$c_off" "$c_cmd" "$GARGS" "$*" "$c_off"
    local out rc
    out="$(EFX $GARGS "$@" 2>&1)"; rc=$?
    printf '%s\n' "$out"
    if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "$want"; then
        printf '%s[correctly failed with %s]%s\n' "$c_ok" "$want" "$c_off"; xfail_ok=$((xfail_ok+1))
    elif [ $rc -ne 0 ]; then
        printf '%s[failed, but not with %s (rc %d)]%s\n' "$c_err" "$want" "$rc" "$c_off"; xfail_bad=$((xfail_bad+1))
    else
        printf '%s[UNEXPECTEDLY SUCCEEDED, expected %s]%s\n' "$c_err" "$want" "$c_off"; xfail_bad=$((xfail_bad+1))
    fi
}

# xfail_rc "<description>" <ef args...>. Like xfail but asserts only a non-zero
# exit, with no substring match on the message. Use when any failure mode is fine.
xfail_rc() {
    local desc="$1"; shift
    printf '\n%s# EXPECT-FAIL (any error): %s%s\n%s$ ef-cli %s %s%s\n' \
        "$c_dim" "$desc" "$c_off" "$c_cmd" "$GARGS" "$*" "$c_off"
    local out rc
    out="$(EFX $GARGS "$@" 2>&1)"; rc=$?
    printf '%s\n' "$out"
    if [ $rc -ne 0 ]; then
        printf '%s[correctly failed (rc %d)]%s\n' "$c_ok" "$rc" "$c_off"; xfail_ok=$((xfail_ok+1))
    else
        printf '%s[UNEXPECTEDLY SUCCEEDED, expected non-zero exit]%s\n' "$c_err" "$c_off"; xfail_bad=$((xfail_bad+1))
    fi
}

# Snapshot the active calibration into CAL_SNAP as "fx fy cx cy xi alpha W H hw fov"
# (best-effort, from calibration --get display precision). Empty => uncalibrated.
snapshot_calibration() {
    local g; g="$(EFX $GARGS calibration --get 2>&1)"
    printf '%s' "$g" | grep -q "uncalibrated" && { CAL_SNAP=""; return; }
    local fx fy cx cy xi al hw fov wh w h
    fx=$(printf '%s' "$g" | sed -n 's/.*fx=\([-0-9.eE+]*\).*/\1/p')
    fy=$(printf '%s' "$g" | sed -n 's/.*fy=\([-0-9.eE+]*\).*/\1/p')
    cx=$(printf '%s' "$g" | sed -n 's/.*cx=\([-0-9.eE+]*\).*/\1/p')
    cy=$(printf '%s' "$g" | sed -n 's/.*cy=\([-0-9.eE+]*\).*/\1/p')
    xi=$(printf '%s' "$g" | sed -n 's/.*xi=\([-0-9.eE+]*\).*/\1/p')
    al=$(printf '%s' "$g" | sed -n 's/.*alpha=\([-0-9.eE+]*\).*/\1/p')
    hw=$(printf '%s' "$g" | sed -n 's/.*rectify: \(o[nf]*\).*/\1/p')
    fov=$(printf '%s' "$g" | sed -n 's/.*fov-scale : \([0-9.]*\).*/\1/p')
    # Calibration resolution isn't shown by --get; use the largest supported mode
    # (calibration is always at full sensor res), fall back to 1920x1200.
    wh=$(EFX $GARGS config 2>&1 | grep -oE '[0-9]+x[0-9]+' | sort -t x -k1,1n -k2,2n | tail -1)
    w=${wh%x*}; h=${wh#*x}; [ -n "$w" ] || { w=1920; h=1200; }
    [ -n "$fx" ] && CAL_SNAP="$fx $fy $cx $cy $xi $al $w $h ${hw:-off} ${fov:-1.0}" || CAL_SNAP=""
}

# Restore the CAL_SNAP snapshot (best-effort). Empty snapshot => reset to factory.
restore_calibration() {
    if [ -z "$CAL_SNAP" ]; then
        EFX $EFARGS calibration --camera --reset >/dev/null 2>&1 && echo "calibration reset (was uncalibrated)"
        return
    fi
    # shellcheck disable=SC2086
    set -- $CAL_SNAP
    EFX $EFARGS calibration --camera --set "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" \
        --rectify "$9" --fov-scale "${10}" >/dev/null 2>&1 \
        && echo "restored calibration snapshot (best-effort, display precision)"
}

cleanup_done=0
cleanup() {
    [ "$cleanup_done" = 1 ] && return; cleanup_done=1     # idempotent (INT then EXIT)
    banner "CLEANUP"
    # Calibration: restore the pre-test snapshot and drop the embed-test recording
    # (only the destructive path touches these).
    [ "${TEST_CALIB:-0}" = 1 ] && restore_calibration
    # A run that dies mid-lock would otherwise leave the device gated and every
    # later verb failing for a reason unrelated to the code under test. Both are
    # no-ops when the lock test never ran.
    if [ "${TEST_LOCK:-0}" = 1 ]; then
        EFX $EFARGS --password "$TEST_BLE_PASSWORD" lock on --session >/dev/null 2>&1
        EFX $EFARGS --password "$TEST_BLE_PASSWORD" lock off >/dev/null 2>&1 \
            && echo "restored USB to unlocked"
    fi
    EFX $EFARGS record delete ef-smoke-cal >/dev/null 2>&1 && echo "deleted ef-smoke-cal"
    EFX $EFARGS record stop            >/dev/null 2>&1
    EFX $EFARGS record delete "$SESSION" >/dev/null 2>&1 && echo "deleted test session $SESSION"
    EFX $EFARGS record delete "$PSESSION" >/dev/null 2>&1 && echo "deleted test session $PSESSION"
    # Restore the persistent location to the compiled SF default (the smoke test
    # mutated session_meta.json). No "reset" verb exists, so we re-set the default.
    EFX $EFARGS location set 37.7749 -122.4194 16 >/dev/null 2>&1 && echo "restored location to SF default"
    [ -n "$UPSESS" ] && EFX $EFARGS record delete "$UPSESS" >/dev/null 2>&1 && echo "deleted upload session $UPSESS"
    # only forget the FAKE throwaway, never a real network the user passed in
    [ "${SKIP_WIFI:-0}" = 1 ] || [ "$TEST_WIFI_SSID" != "ef-smoke-fake" ] || { \
        EFX $EFARGS wifi remove "$TEST_WIFI_SSID" >/dev/null 2>&1 \
        && echo "removed throwaway wifi $TEST_WIFI_SSID"; }
    # upload receiver + temp dir (set only while the upload test is running)
    [ -n "$RXPID" ] && { kill "$RXPID" 2>/dev/null && echo "stopped upload receiver (pid $RXPID)"; }
    [ -n "$RXDIR" ] && rm -rf "$RXDIR"
    # revert the ufw rule iff WE opened it
    [ "$FW_OPENED" = 1 ] && { sudo ufw delete allow "$FW_PORT/tcp" >/dev/null 2>&1 \
        && echo "ufw: reverted $FW_PORT/tcp"; }
    [ -n "$DLDIR" ] && rm -rf "$DLDIR"
}

# mcap_location <file> -> prints "lat lon" (%.4f) from the first /camera/location
# LocationFix message, or nothing. Minimal varint-aware protobuf walk (fields
# latitude=1, longitude=2 are 64-bit doubles). No-op if python3/mcap are absent.
mcap_location() {
    python3 - "$1" <<'PY' 2>/dev/null
import sys, struct
try:
    from mcap.reader import make_reader
except Exception:
    sys.exit(0)
def walk(b):
    d={}; i=0; n=len(b)
    while i < n:
        tag=b[i]; i+=1; fn=tag>>3; wt=tag&7
        if wt==0:
            while i<n and b[i]&0x80: i+=1
            i+=1
        elif wt==1:
            d[fn]=struct.unpack_from('<d', b, i)[0]; i+=8
        elif wt==2:
            ln=0; sh=0
            while i<n:
                c=b[i]; i+=1; ln|=(c&0x7f)<<sh
                if not c&0x80: break
                sh+=7
            i+=ln
        elif wt==5: i+=4
        else: break
    return d
try:
    with open(sys.argv[1],'rb') as f:
        for _s,ch,m in make_reader(f).iter_messages():
            if ch.topic=='/camera/location':
                d=walk(m.data)
                print(f"{d.get(1,0.0):.4f} {d.get(2,0.0):.4f}")
                break
except Exception:
    pass
PY
}

# check_mcap_loc <file> <lat-substr> <lon-substr>. Assert the MCAP's LocationFix
# contains the expected lat/lon (substring match tolerates float formatting).
check_mcap_loc() {
    if ! have python3 || ! python3 -c "import mcap" >/dev/null 2>&1; then
        echo "${c_dim}(skip MCAP location check: need python3 + the mcap lib)${c_off}"; return
    fi
    local loc; loc="$(mcap_location "$1")"
    printf '  MCAP /camera/location = [%s]  (want ~%s, %s)\n' "$loc" "$2" "$3"
    if printf '%s' "$loc" | grep -q -- "$2" && printf '%s' "$loc" | grep -q -- "$3"; then
        printf '%s[LocationFix matches]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
    else
        printf '%s[LocationFix MISMATCH]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
    fi
}
# Full cleanup on normal exit AND on Ctrl-C / kill: INT/TERM clear the EXIT trap,
# run cleanup once, then exit with the conventional signal code.
trap 'trap - EXIT; cleanup; exit 130' INT
trap 'trap - EXIT; cleanup; exit 143' TERM
trap cleanup EXIT

# Private 0700 dir for downloaded captures (never predictable world-writable /tmp
# paths: symlink/temp race). Created now that cleanup() is armed to remove it.
DLDIR="$(mktemp -d "${TMPDIR:-/tmp}/efsmoke.XXXXXX")"

# ---------------------------------------------------------------------------
banner "PREFLIGHT"
echo "ef binary : $EF"
[ -x "$EF" ] || { echo "${c_err}ef not found/executable at $EF: build it first (cmake --build build)$c_off"; exit 1; }

# ---- Linux dependency check: hard-fail on required, warn on optional tools that
#      an ENABLED feature needs (missing optional -> that step skips, not aborts).
have() { command -v "$1" >/dev/null 2>&1; }
dep_req=""; dep_opt=""
have timeout || dep_req="$dep_req coreutils"     # EFX wraps every ef call in `timeout`
[ "${SKIP_GRAB:-0}"    = 1 ] || { have ffmpeg || dep_opt="$dep_opt ffmpeg"; have python3 || dep_opt="$dep_opt python3"; }
[ "${TEST_UPLOAD:-0}"  = 1 ] && { have python3 || dep_opt="$dep_opt python3"; }
[ "${OPEN_FIREWALL:-0}" = 1 ] && { have ufw || dep_opt="$dep_opt ufw"; }
if [ -n "$dep_req" ]; then
    echo "${c_err}missing required:$dep_req  ->  sudo apt install$dep_req$c_off"; exit 1
fi
if [ -n "$dep_opt" ]; then
    dep_opt=$(printf '%s\n' $dep_opt | sort -u | tr '\n' ' ')
    echo "${c_err}missing tools for enabled features: $dep_opt$c_off"
    echo "  install:  sudo apt install $dep_opt   (otherwise those steps skip/degrade)"
else
    echo "deps ok   : all tools for the enabled features are present"
fi

EFX $EFARGS list 2>&1
if ! EFX $EFARGS info >/dev/null 2>&1; then
    echo "${c_err}No device reachable (ef info failed). Flash + connect the board over USB, then re-run.$c_off"
    exit 1
fi

# ---------------------------------------------------------------------------
banner "IDENTITY / QUERIES (read-only)"
run "device discovery"                 list
run "device information snapshot"      info
run "current DEVICE_STATE (4-state)"   state
run "enabled capture modes + current"  config
run "recording store free/total"       storage
run "align device clock to host"       sync-time
run "read device wall clock"           time
run "read device location"             location

# ---------------------------------------------------------------------------
banner "HEALTH"
run "shallow health sweep"             health
if [ "${SKIP_DEEP:-0}" != 1 ]; then
    # Let the device firmware leave HEALTH_TEST before the next sweep; back-to-back
    # health checks race it and the second returns DEVICE_NOT_AVAILABLE. (ef state
    # reports IDLE even mid-health-test, so a settle is more reliable than polling.)
    sleep 3
    run "deep health sweep (stress ~2-3 min)"  health --deep
else
    echo "(skipped deep sweep: SKIP_DEEP=1)"
fi

# ---------------------------------------------------------------------------
banner "CONFIG: valid set (restore a known-good mode)"
# Settle after the health sweep: config set is IDLE-only and a just-finished
# sweep can leave the device briefly busy (INVALID_FUNCTION_CALL otherwise).
sleep 2
run "set 1920x1200@30 h265"            config set 1920 1200 30 h265

# ---------------------------------------------------------------------------
banner "LOCATION: persistent set/get (session_meta.json)"
run "set persistent location (NYC)"    location set 40.7128 -74.0060 10
LOC_AFTER="$(EFX $GARGS location 2>&1)"
printf '  location now: %s\n' "$LOC_AFTER"
if printf '%s' "$LOC_AFTER" | grep -q "40.7128" && printf '%s' "$LOC_AFTER" | grep -q -- "-74.006"; then
    printf '%s[persistent location applied]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[persistent location NOT applied]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
run "config after set"                 config

# ---------------------------------------------------------------------------
banner "CALIBRATION: read-only get (always) + set/reset/toggle (opt-in)"
run "read camera + IMU calibration"    calibration --get
if [ "${TEST_CALIB:-0}" != 1 ]; then
    echo "(skipped destructive calibration tests: pass --test-calibration to run;"
    echo " they WIPE the device calibration; snapshot+restore is best-effort)"
else
printf '%s! --test-calibration modifies the device calibration; best-effort restore at exit.%s\n' "$c_err" "$c_off"
snapshot_calibration
[ -n "$CAL_SNAP" ] && echo "calibration snapshot: $CAL_SNAP" || echo "calibration snapshot: (uncalibrated)"
# set intrinsics, confirm they persist + read back (host -> endpoint -> device firmware
# -> /var/lib/efference/calibration/camera.json -> read).
run "set camera intrinsics"            calibration --camera --set 702.5 703.1 960 600 0 0.61 1920 1200
CAL_GET="$(EFX $GARGS calibration --get 2>&1)"
printf '%s\n' "$CAL_GET"
if printf '%s' "$CAL_GET" | grep -q "fx=702\.5" && printf '%s' "$CAL_GET" | grep -q "alpha=0\.610000"; then
    printf '%s[intrinsics applied + read back]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[intrinsics NOT read back]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# ef info must show the same values + "rectify off" (the flag defaults off).
CAL_INFO="$(EFX $GARGS info 2>&1 | grep calibration)"
printf '  info: %s\n' "$CAL_INFO"
if printf '%s' "$CAL_INFO" | grep -q "702\.50" && printf '%s' "$CAL_INFO" | grep -q "rectify off"; then
    printf '%s[ef info reflects calibration, rectify off]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[ef info calibration mismatch]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# --rectify sets the on-device FEC rectify flag (persisted; rectification itself is not
# implemented yet). ef info must flip to "rectify on".
run "set intrinsics with --rectify on"  calibration --camera --set 702.5 703.1 960 600 0 0.61 1920 1200 --rectify on
if EFX $GARGS info 2>&1 | grep calibration | grep -q "rectify on"; then
    printf '%s[rectify flag on]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[rectify flag did not turn on]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# Standalone toggle: change rectify / fov-scale WITHOUT re-typing intrinsics
# (read-modify-write). The intrinsics (fx=702.5) must survive every round-trip.
run "toggle rectify OFF (standalone)"   calibration --camera --rectify off
TOG="$(EFX $GARGS calibration --get 2>&1)"
printf '%s\n' "$TOG"
if printf '%s' "$TOG" | grep -q "rectify: off" && printf '%s' "$TOG" | grep -q "fx=702\.5"; then
    printf '%s[rectify off, intrinsics preserved]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[standalone toggle flipped flag wrong or disturbed intrinsics]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
run "toggle rectify ON (standalone)"    calibration --camera --rectify on
if EFX $GARGS calibration --get 2>&1 | grep -q "rectify: on"; then
    printf '%s[rectify on (standalone)]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[standalone rectify on did not take]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# --fov-scale alone sets the rectified FOV without touching rectify or intrinsics.
run "set fov-scale 1.20 (standalone)"      calibration --camera --fov-scale 1.20
FOV="$(EFX $GARGS calibration --get 2>&1)"
printf '%s\n' "$FOV"
if printf '%s' "$FOV" | grep -q "fov-scale : 1.20" && printf '%s' "$FOV" | grep -q "fx=702\.5" \
   && printf '%s' "$FOV" | grep -q "rectify: on"; then
    printf '%s[fov-scale 1.20; rectify + intrinsics untouched]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[fov-scale not applied or it clobbered other fields]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# Both flags in one call; also restores fov-scale 1.0 so later steps aren't surprised.
run "toggle rectify+fov together"       calibration --camera --rectify on --fov-scale 1.00
if EFX $GARGS calibration --get 2>&1 | grep -q "fov-scale : 1.00"; then
    printf '%s[combined toggle applied; fov-scale back to 1.00]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[combined toggle / fov restore failed]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# A device-local recording must embed the intrinsics as foxglove.CameraCalibration
# (distortion_model="double_sphere") while calibration is set. Verify the recorded
# MCAP carries it (embed is ungated by the rectify flag).
if [ "${SKIP_RECORD:-0}" != 1 ]; then
    CAL_MCAP="./efsmoke_cal.mcap"
    run "record for calibration embed"     record start ef-smoke-cal
    sleep 2
    # calibration writes are IDLE-only: a set is rejected while a recording is
    # active (device is in COLLECT, not IDLE). The rejected set does NOT change the
    # active calibration, so the embed check below still sees double_sphere.
    xfail "set calibration while recording (IDLE-only)" INVALID_FUNCTION_CALL \
        calibration --camera --set 100 100 100 100 0 0.5 1920 1200
    # The standalone toggle is a calibration write too, so it's IDLE-only as well.
    xfail "rectify toggle while recording (IDLE-only)" INVALID_FUNCTION_CALL \
        calibration --camera --rectify off
    sleep 2
    run "stop calibration recording"       record stop
    run "download calibration recording"   download ef-smoke-cal "$CAL_MCAP"
    if [ -s "$CAL_MCAP" ] && grep -a -q "double_sphere" "$CAL_MCAP"; then
        printf '%s[MCAP embeds double_sphere intrinsics]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
    else
        printf '%s[MCAP missing double_sphere intrinsics]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
    fi
    run "delete calibration recording"     record delete ef-smoke-cal
    rm -f "$CAL_MCAP"
else
    echo "(skipped calibration MCAP-embed check: SKIP_RECORD=1)"
fi

# reset restores the golden factory default. Assert it moved AWAY from the test
# intrinsics (702.5) -- holds whether golden is zeroed or a real calibration.
run "reset camera calibration"         calibration --camera --reset
if EFX $GARGS calibration --get 2>&1 | grep -q "fx=702\.5"; then
    printf '%s[reset did NOT clear the test intrinsics]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
else
    printf '%s[reset cleared the test intrinsics]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
fi
fi   # end --test-calibration (destructive)

# ---------------------------------------------------------------------------
banner "CALIBRATION: IMU field-calibration mode + reset (IDLE-only)"
# The IMU calibration values are written by the field-calibration tutorial
# (tutorials/calibrate_imu: still + tumble capture -> solve -> set). Here
# we exercise the control surface the SDK exposes: the on-device capture-mode
# toggle, reset, report readback, and that a recording embeds the params.
run "select calibrated IMU capture"    calibration --imu --mode calibrated
run "select both (raw+calibrated)"     calibration --imu --mode both
run "reset IMU calibration"            calibration --imu --reset
# report readback: the IMU block must be present. This reads the field imu.json
# (device-firmware-owned), not a stale device.json -- the report-path fix.
IMU_GET="$(EFX $GARGS calibration --get 2>&1)"
printf '%s\n' "$IMU_GET"
if printf '%s' "$IMU_GET" | grep -qi "imu calibration"; then
    printf '%s[calibration --get reports the IMU block]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[calibration --get missing IMU block]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
# leave the device recording UNCALIBRATED (the data-collection default) for the rest.
run "back to raw (data-collection default)"  calibration --imu --mode raw
# A device-local recording must embed the full IMU calibration as
# efference.ImuCalibration on /camera/imu/0/calibration (params ride as metadata
# regardless of the capture mode).
if [ "${SKIP_RECORD:-0}" != 1 ]; then
    IMU_MCAP="$DLDIR/efsmoke_imucal.mcap"
    run "record for IMU-cal embed"         record start ef-smoke-imucal
    sleep 2
    run "stop IMU-cal recording"           record stop
    run "download IMU-cal recording"       download ef-smoke-imucal "$IMU_MCAP"
    if [ -s "$IMU_MCAP" ] && grep -a -q "efference.ImuCalibration" "$IMU_MCAP" \
       && grep -a -q "camera/imu/0/calibration" "$IMU_MCAP"; then
        printf '%s[MCAP embeds efference.ImuCalibration on /camera/imu/0/calibration]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
    else
        printf '%s[MCAP missing the IMU calibration message]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
    fi
    run "delete IMU-cal recording"         record delete ef-smoke-imucal
    rm -f "$IMU_MCAP"
else
    echo "(skipped IMU-cal MCAP-embed check: SKIP_RECORD=1)"
fi

# ---------------------------------------------------------------------------
if [ "${SKIP_GRAB:-0}" != 1 ]; then
banner "DATA PLANE: live stream smoke (grab tutorial: open/grab/retrieve) + save a viewable clip"
# The grab tutorial is a standalone project; build it against the SDK build tree
# on demand (unless EFGRAB points at a prebuilt binary).
GRABDIR="$HERE/../../../tutorials/grab/cpp"
if [ -z "${EFGRAB:-}" ] && [ -d "$GRABDIR" ]; then
    cmake -S "$GRABDIR" -B "$GRABDIR/build" -DCMAKE_PREFIX_PATH="$HERE/../build" >/dev/null 2>&1 \
        && cmake --build "$GRABDIR/build" >/dev/null 2>&1
    EFGRAB="$GRABDIR/build/grab"
fi
EFGRAB="${EFGRAB:-$GRABDIR/build/grab}"
GRAB_SECS="${GRAB_SECS:-5}"
# Fixed base name in the current folder, OVERWRITTEN each run (not cleaned up) so
# you always have the latest capture to eyeball. Override the location with GRAB_BASE.
GRAB_BASE="${GRAB_BASE:-./efsmoke_grab}"
if [ ! -x "$EFGRAB" ]; then
    echo "${c_err}grab tutorial not found/built at $EFGRAB, skipping data-plane smoke$c_off"
else
    printf '\n%s# %ss live capture over the wire (frames / IMU / fps), tee to %s.mcap%s\n%s$ grab %s --codec h265 --record %s.mcap%s\n' \
        "$c_dim" "$GRAB_SECS" "$GRAB_BASE" "$c_off" "$c_cmd" "$GRAB_SECS" "$GRAB_BASE" "$c_off"
    out="$(timeout -k 5 "$TIMEOUT" "$EFGRAB" "$GRAB_SECS" --codec h265 --record "$GRAB_BASE.mcap" 2>&1)"; rc=$?
    printf '%s\n' "$out"
    if [ $rc -eq 0 ]; then printf '%s[exit 0 OK]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
    else printf '%s[exit %d UNEXPECTED-FAIL]%s\n' "$c_err" "$rc" "$c_off"; fail=$((fail+1)); fi
    # Convert the captured frames to viewable artifacts (same names, overwritten).
    # mcap -> raw H.265 Annex-B (mcap_to_video.py) -> ffmpeg clip (.mp4) + still (.png).
    MCV="$HERE/mcap_to_video.py"
    if [ -s "$GRAB_BASE.mcap" ] && command -v ffmpeg >/dev/null 2>&1 && [ -f "$MCV" ] \
       && python3 "$MCV" "$GRAB_BASE.mcap" "$GRAB_BASE.h265" >/dev/null 2>&1 && [ -s "$GRAB_BASE.h265" ]; then
        ffmpeg -y -loglevel error -f hevc -i "$GRAB_BASE.h265" -c:v libx264 "$GRAB_BASE.mp4" 2>/dev/null \
            && echo "${c_ok}saved clip : $GRAB_BASE.mp4$c_off   (view: ffplay $GRAB_BASE.mp4)"
        ffmpeg -y -loglevel error -f hevc -i "$GRAB_BASE.h265" -frames:v 1 "$GRAB_BASE.png" 2>/dev/null \
            && echo "${c_ok}saved still: $GRAB_BASE.png$c_off   (view: ffplay $GRAB_BASE.png)"
    else
        echo "${c_dim}(no viewable clip: need frames in $GRAB_BASE.mcap + ffmpeg + mcap_to_video.py)$c_off"
    fi
fi
else
echo; echo "(skipped data-plane smoke: SKIP_GRAB=1)"
fi

# ---------------------------------------------------------------------------
banner "INTENTIONAL ERROR: raw codec over WiFi/UDP (INSUFFICIENT_WIFI_BANDWIDTH)"
# Raw NV12 (~830 Mbit/s @1200p30) overruns the WiFi link, so the SDK rejects
# RAW + a udp_host at open(), before any streaming. efference-viewer is the CLI
# entry that sets both (init.compression via --codec, plus --udp). 192.0.2.1 is
# TEST-NET-1 and is never contacted: the guard fires on open() over USB, before
# the data plane is touched. (The device enforces the same rule at StartStream.)
EFVIEW="${EFVIEW:-$HERE/../build/efference-viewer}"
if [ ! -x "$EFVIEW" ]; then
    echo "${c_dim}(skip raw-over-UDP check: efference-viewer not built at $EFVIEW)${c_off}"
else
    printf '\n%s# EXPECT-FAIL (INSUFFICIENT_WIFI_BANDWIDTH): raw over UDP%s\n%s$ efference-viewer --codec raw --udp 192.0.2.1 --headless%s\n' \
        "$c_dim" "$c_off" "$c_cmd" "$c_off"
    out="$(timeout -k 5 "$TIMEOUT" "$EFVIEW" --codec raw --udp 192.0.2.1 --headless 2>&1)"; rc=$?
    printf '%s\n' "$out"
    if [ $rc -ne 0 ] && printf '%s' "$out" | grep -q "INSUFFICIENT_WIFI_BANDWIDTH"; then
        printf '%s[correctly rejected raw-over-UDP at open]%s\n' "$c_ok" "$c_off"; xfail_ok=$((xfail_ok+1))
    elif [ $rc -ne 0 ]; then
        printf '%s[failed, but not with INSUFFICIENT_WIFI_BANDWIDTH (rc %d)]%s\n' "$c_err" "$rc" "$c_off"; xfail_bad=$((xfail_bad+1))
    else
        printf '%s[UNEXPECTEDLY SUCCEEDED, expected INSUFFICIENT_WIFI_BANDWIDTH]%s\n' "$c_err" "$c_off"; xfail_bad=$((xfail_bad+1))
    fi
fi

# ---------------------------------------------------------------------------
banner "INTENTIONAL ERRORS: config (P2/P3 granular codes)"
xfail "bad resolution"      INVALID_RESOLUTION      config set 9999 9999 30 h265
xfail "bad fps"             INVALID_FPS             config set 1920 1200 999 h265
# The CLI validates the codec client-side (raw|h264|h264hq|h265|h265hq), so a bad
# codec is rejected before the wire, every wire-valid codec is device-supported,
# so UNSUPPORTED_COMPRESSION isn't reachable via `config set`. Test the CLI guard.
xfail "unknown codec (CLI-side reject)" "unknown codec" config set 1920 1200 30 mjpeg
xfail "location set missing args (CLI reject)"     "need <lat>"  location set 40.0
xfail "record --location bad format (CLI reject)"  "LAT,LON"     record start --location not-a-coord
xfail "calibration set wrong arg count (CLI reject)" "usage"     calibration --camera --set 1 2 3
xfail "rectify bad value (CLI reject)"     "wants 'on' or 'off'"   calibration --camera --rectify maybe
xfail "fov-scale missing value (CLI reject)"  "wants a value"         calibration --camera --fov-scale
xfail "fov-scale non-positive (CLI reject)"   "positive number"       calibration --camera --fov-scale 0
xfail "fov-scale not a number (CLI reject)"    "positive number"       calibration --camera --fov-scale abc
xfail "rectify toggle on --imu (CLI reject)"  "apply to --camera only" calibration --imu --rectify on
xfail "rectify flags with stray args (CLI reject)" "unexpected arguments" calibration --camera --rectify on 1920 1080
xfail "imu bad capture mode (CLI reject)"  "unknown imu mode"    calibration --imu --mode bogus
xfail_rc "imu --set points to the field-cal tool" calibration --imu --set

# ---------------------------------------------------------------------------
banner "INTENTIONAL ERRORS: recording / state"
xfail "stop when not recording"  INVALID_FUNCTION_CALL  record stop
xfail "status of missing session" RECORDING_NOT_FOUND   record status no-such-session
xfail "delete missing session"    RECORDING_NOT_FOUND   record delete no-such-session
xfail "download missing session"  RECORDING_NOT_FOUND   download no-such-session /tmp/none.mcap

# upload URL scheme is validated host-side (CLI reject, no device round-trip)
xfail "upload with no scheme"     "must be an http"  upload no-such-session example.com/x
xfail "upload with ftp:// scheme" "must be an http"  upload no-such-session ftp://host/x
xfail "upload with file:// path"  "must be an http"  upload no-such-session file:///tmp/x

# ---------------------------------------------------------------------------
if [ "${SKIP_RECORD:-0}" != 1 ]; then
banner "RECORD LIFECYCLE (device-local, self-cleaning)"
run "start recording $SESSION"    record start "$SESSION"
sleep 3
run "recording status"            record status "$SESSION"
run "state during recording (STREAMING)"  state
xfail "location set while recording (must reject)"  INVALID_FUNCTION_CALL  location set 1.0 2.0
run "stop recording"              record stop
run "list recordings"             record list
# record list must say, per recording, whether the stored segments are encrypted.
# The device answers by reading the container magic off a segment, so this also
# proves the recorder and the reporter agree about what actually landed on disk.
LISTOUT=$(EFX $GARGS record list 2>&1 || true)
if printf '%s' "$LISTOUT" | grep -qE "\[(encrypted|unencrypted)\]"; then
    printf '%s[record list reports encryption state]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[record list is missing the encrypted/unencrypted marker]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
xfail "start with a duplicate name (must reject, not overwrite)" \
      RECORDING_ALREADY_EXISTS    record start "$SESSION"
run "download $SESSION"           download "$SESSION" "$DLDIR/$SESSION.mcap"
[ -s "$DLDIR/$SESSION.mcap" ] && echo "downloaded $(stat -c%s "$DLDIR/$SESSION.mcap" 2>/dev/null) bytes -> $DLDIR/$SESSION.mcap"
# this (no-override) recording must carry the persistent NYC set in the LOCATION section
check_mcap_loc "$DLDIR/$SESSION.mcap" "40.712" "-74.006"
run "delete $SESSION"             record delete "$SESSION"

# per-session --location override (London): the MCAP must carry London, and the
# persistent default must remain NYC afterwards (a per-session override never persists).
run "start recording $PSESSION (--location London)"  record start "$PSESSION" --location 51.5074,-0.1278
sleep 3
run "stop recording (per-session)"  record stop
run "download $PSESSION"            download "$PSESSION" "$DLDIR/$PSESSION.mcap"
check_mcap_loc "$DLDIR/$PSESSION.mcap" "51.507" "-0.127"
LOC_STILL="$(EFX $GARGS location 2>&1)"
printf '  persistent location after per-session run: %s\n' "$LOC_STILL"
if printf '%s' "$LOC_STILL" | grep -q "40.7128"; then
    printf '%s[per-session did NOT change the persistent default]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
else
    printf '%s[per-session leaked into persistent!]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
fi
run "delete $PSESSION"             record delete "$PSESSION"
else
echo; echo "(skipped record lifecycle: SKIP_RECORD=1)"
fi

# ---------------------------------------------------------------------------
if [ "${TEST_UPLOAD:-0}" = 1 ]; then
banner "UPLOAD (opt-in: --test-upload): record -> upload to a local receiver -> verify"
# We only stand up a RECEIVER at a ready URL; the create/finalize orchestration
# is the cloud/app side and out of scope. Device uploads over WiFi, so the device
# must be on WiFi with a route to --upload-host (this machine's reachable IP).
# If it's not connected and real creds were given, provision + wait for it first.
if [ -n "$UPLOAD_HOST" ] && ! EFX $EFARGS wifi status 2>/dev/null | grep -q "connected to" \
   && [ -n "$TEST_WIFI_PSK" ] && [ "$TEST_WIFI_SSID" != "ef-smoke-fake" ]; then
    echo "device not on WiFi, provisioning '$TEST_WIFI_SSID' for the upload..."
    EFX $EFARGS wifi add "$TEST_WIFI_SSID" "$TEST_WIFI_PSK" "$TEST_WIFI_COUNTRY" >/dev/null 2>&1
    j=0; while [ $j -lt 20 ]; do
        EFX $EFARGS wifi status 2>/dev/null | grep -q "connected to" && break
        sleep 2; j=$((j + 1))
    done
fi
UPLOAD_PORT="${UPLOAD_PORT:-8099}"
if [ -z "$UPLOAD_HOST" ]; then
    echo "${c_err}--upload-host <ip> required (host IP the device can reach over WiFi), skipping$c_off"
elif ! command -v python3 >/dev/null 2>&1; then
    echo "${c_err}python3 not found (needed for the receiver), skipping$c_off"
elif ! EFX $EFARGS wifi status 2>/dev/null | grep -q "connected to"; then
    echo "${c_err}device is NOT on WiFi. Upload runs over WiFi, so it can't work. Connect it, or"
    echo "${c_err}pass --wifi-ssid/--wifi-psk so the test provisions it. Skipping upload.$c_off"
elif ss -ltn 2>/dev/null | grep -q ":$UPLOAD_PORT "; then
    echo "${c_err}port $UPLOAD_PORT is already in use on this host, our receiver can't bind, so the"
    echo "${c_err}device's PUT would hit the wrong server (e.g. a plain file server that rejects PUT)."
    echo "${c_err}Pick a free --upload-port, or find + stop the owner:  ss -ltnp | grep :$UPLOAD_PORT . Skipping.$c_off"
else
    # Optionally ufw-allow the inbound port for the duration of the test. Only if
    # it isn't already allowed (so we never clobber existing config), and record
    # that WE opened it so cleanup() reverts it on exit (incl. Ctrl-C).
    if [ "${OPEN_FIREWALL:-0}" = 1 ] && command -v ufw >/dev/null 2>&1; then
        if sudo ufw status 2>/dev/null | grep -q "$UPLOAD_PORT/tcp"; then
            echo "ufw: $UPLOAD_PORT/tcp already allowed, leaving as-is"
        elif sudo ufw allow "$UPLOAD_PORT/tcp" >/dev/null 2>&1; then
            FW_OPENED=1; FW_PORT="$UPLOAD_PORT"
            echo "ufw: opened $UPLOAD_PORT/tcp (will auto-revert on exit)"
        else
            echo "${c_err}ufw: could not open $UPLOAD_PORT/tcp (sudo?), inbound PUT may be blocked$c_off"
        fi
    fi
    UPSESS="efup$$"
    RXDIR="$(mktemp -d)"
    cat > "$RXDIR/rx.py" <<'PYEOF'
import http.server, os, sys
D, P = sys.argv[1], int(sys.argv[2])
class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_PUT(self):
        n = int(self.headers.get('Content-Length', 0))
        with open(os.path.join(D, os.path.basename(self.path)), 'wb') as f:
            f.write(self.rfile.read(n))
        self.send_response(200); self.send_header('Content-Length', '0'); self.end_headers()
    def do_POST(self):  # tolerate any create/finalize the device may send
        self.send_response(200); self.send_header('Content-Length', '0'); self.end_headers()
http.server.HTTPServer(('0.0.0.0', P), H).serve_forever()
PYEOF
    python3 "$RXDIR/rx.py" "$RXDIR" "$UPLOAD_PORT" >/dev/null 2>&1 &
    RXPID=$!
    sleep 1
    URL="http://$UPLOAD_HOST:$UPLOAD_PORT/gcs/$UPSESS.mcap"
    echo "receiver pid $RXPID  dir $RXDIR   target url: $URL"
    echo "${c_dim}  NOTE: the device connects INBOUND to $UPLOAD_HOST:$UPLOAD_PORT. That port must be"
    echo "        allowed through THIS host's firewall or the PUT is silently dropped,"
    echo "        e.g.  sudo ufw allow $UPLOAD_PORT/tcp   (revert with 'ufw delete allow $UPLOAD_PORT/tcp').${c_off}"
    run "record a session to upload"  record start "$UPSESS"
    sleep 3
    run "stop it"                      record stop
    run "upload $UPSESS -> receiver"   upload "$UPSESS" "$URL"
    printf '\n%s# waiting up to 40s for the file to land at the receiver (device uploads async over WiFi)...%s\n' "$c_dim" "$c_off"
    got=0; i=0
    while [ $i -lt 20 ]; do
        [ -s "$RXDIR/$UPSESS.mcap" ] && { got=1; break; }
        sleep 2; i=$((i + 1))
    done
    if [ "$got" = 1 ]; then
        sz=$(stat -c%s "$RXDIR/$UPSESS.mcap" 2>/dev/null)
        printf '%s[received %s bytes at the receiver -> UPLOAD OK]%s\n' "$c_ok" "$sz" "$c_off"; pass=$((pass + 1))
    else
        printf '%s[nothing received after 40s -> UPLOAD FAILED]%s\n' "$c_err" "$c_off"; fail=$((fail + 1))
        echo "  likely: (1) inbound port $UPLOAD_PORT blocked by this host's firewall (open it), or"
        echo "          (2) device not on WiFi / no route to $UPLOAD_HOST. 'ef record status $UPSESS' shows the device-side upload state."
    fi
    # Normal-path teardown; clear the globals so cleanup() doesn't repeat it.
    # (The ufw revert is left to cleanup so it also covers a Ctrl-C before here.)
    EFX $EFARGS record delete "$UPSESS" >/dev/null 2>&1
    kill "$RXPID" 2>/dev/null; rm -rf "$RXDIR"
    RXPID="" RXDIR="" UPSESS=""
fi
else
echo; echo "(upload round-trip NOT run, pass --test-upload --upload-host <ip> to enable)"
fi

# ---------------------------------------------------------------------------
banner "INTENTIONAL ERROR: upload precondition (WIFI_NOT_CONNECTED)"
# Only meaningful if the device is NOT on WiFi; if it is connected this may
# instead reach the URL. Kept as EXPECT-FAIL either way (bad URL / no wifi).
xfail "upload without WiFi (or bad URL)"  "WIFI_NOT_CONNECTED\|RECORDING_NOT_FOUND\|FAIL" \
      upload no-such-session http://192.0.2.1/none

# ---------------------------------------------------------------------------
if [ "${SKIP_WIFI:-0}" != 1 ]; then
banner "WIFI MUTATION (throwaway '$TEST_WIFI_SSID', added then removed)"
[ "$TEST_WIFI_SSID" = "ef-smoke-fake" ] && \
    echo "(fake creds -> expect connecting then disconnected; pass --wifi-ssid/--wifi-psk for a real join to 'connected')"
run "wifi status before"          wifi status
run "list saved networks before"  wifi list
run "add throwaway network"       wifi add "$TEST_WIFI_SSID" "$TEST_WIFI_PSK" "$TEST_WIFI_COUNTRY"
run "list saved networks after add" wifi list
for i in 1 2 3 4; do
    printf '\n%s# wifi status poll %d/4 (watch for connecting -> connected/disconnected)%s\n' "$c_dim" "$i" "$c_off"
    EFX $EFARGS wifi status 2>&1
    sleep 3
done
run "select throwaway network"    wifi select "$TEST_WIFI_SSID"
run "wifi status after select"    wifi status
if [ "$TEST_WIFI_SSID" = "ef-smoke-fake" ]; then
    run "remove throwaway network"    wifi remove "$TEST_WIFI_SSID"
    run "wifi status after remove"    wifi status
else
    echo "(keeping '$TEST_WIFI_SSID', it's your real network; not removing so the device stays connected)"
fi
xfail "select a network not saved" INVALID_FUNCTION_CALL wifi select definitely-not-saved
# Removing a never-saved ssid must report not-saved, NOT a false "forgotten"
# success, and must never wipe the whole store (the bug this guards).
xfail "remove a network not saved" "not a saved network" wifi remove definitely-not-saved

# Wrong-password detection: add the REAL network with a deliberately bad PSK and
# confirm the device reports auth_failed within seconds (not an endless
# "connecting"), then restore the correct PSK so it reconnects. Opt-in
# (TEST_WIFI_AUTHFAIL=1) and needs a real --wifi-ssid/--wifi-psk, since it briefly
# drops the live WiFi link. Only the device (wpa_supplicant) can prove this.
if [ "${TEST_WIFI_AUTHFAIL:-0}" = 1 ] && [ "$TEST_WIFI_SSID" != "ef-smoke-fake" ] \
   && [ -n "$TEST_WIFI_PSK" ]; then
    banner "WIFI WRONG-PASSWORD (auth_failed) for real net '$TEST_WIFI_SSID'"
    run "forget real net (drop live link)" wifi remove "$TEST_WIFI_SSID"
    run "add real net with WRONG psk"      wifi add "$TEST_WIFI_SSID" definitely-wrong-psk-123 "$TEST_WIFI_COUNTRY"
    authfail_ok=0
    for i in 1 2 3 4 5 6; do
        printf '\n%s# auth_failed poll %d/6%s\n' "$c_dim" "$i" "$c_off"
        authfail_out=$(EFX $EFARGS wifi status 2>&1); printf '%s\n' "$authfail_out"
        printf '%s' "$authfail_out" | grep -qi "authentication failed" && { authfail_ok=1; break; }
        sleep 2
    done
    [ "$authfail_ok" = 1 ] && echo "${c_ok}PASS: reported auth_failed for wrong password$c_off" \
                           || echo "${c_err}FAIL: never reported auth_failed$c_off"
    run "restore correct psk" wifi add "$TEST_WIFI_SSID" "$TEST_WIFI_PSK" "$TEST_WIFI_COUNTRY"
fi
else
echo; echo "(skipped wifi mutation: SKIP_WIFI=1)"
fi

# ---------------------------------------------------------------------------
if [ "${SKIP_BLE:-0}" != 1 ]; then
banner "BLE CONTROL (MAC discovered from 'ef info' over USB, not assumed)"
# We don't know the BLE MAC up front, pull it from the USB identity snapshot,
# exactly as a real client would (ef info -> 'bt mac : <MAC>').
BLE_MAC="$(EFX $EFARGS info 2>/dev/null | sed -n 's/^bt mac *: *\([0-9A-Fa-f:]\{17\}\).*/\1/p')"
if [ -z "$BLE_MAC" ]; then
    echo "${c_err}no BT MAC in 'ef info' (bt unprovisioned / down), skipping BLE section$c_off"
else
    echo "discovered BT MAC : $BLE_MAC   (password: $TEST_BLE_PASSWORD)"
    run "BLE scan/discover" list --scan-ble
    # Re-run the read-only control verbs over the BLE transport. Same verbs,
    # different global flags, swap $GARGS so the helpers target BLE, then restore.
    GARGS="$EFARGS --ble $BLE_MAC --password $TEST_BLE_PASSWORD"
    run "BLE: device information"        info
    run "BLE: DEVICE_STATE"              state
    run "BLE: config"                    config
    run "BLE: storage"                   storage
    run "BLE: wifi status"               wifi status
    run "BLE: wifi list"                 wifi list
    run "BLE: recording list"            record list
    run "BLE: shallow health"            health
    # Explicit wrong-password check (bypasses $GARGS to force a bad password).
    # It MUST probe a gated verb: info/state/storage/factory-reset answer pre-auth
    # by design, so `state` here always succeeded and proved nothing about the gate.
    printf '\n%s# EXPECT-FAIL (INVALID_PASSWORD): BLE wrong password%s\n%s$ ef-cli --ble %s --password %s record list%s\n' \
        "$c_dim" "$c_off" "$c_cmd" "$BLE_MAC" "$WRONG_PW" "$c_off"
    if out="$(EFX --ble "$BLE_MAC" --password "$WRONG_PW" record list 2>&1)"; then
        printf '%s\n%s[UNEXPECTEDLY SUCCEEDED, expected INVALID_PASSWORD]%s\n' "$out" "$c_err" "$c_off"; xfail_bad=$((xfail_bad+1))
    else
        printf '%s\n' "$out"
        if printf '%s' "$out" | grep -q "INVALID_PASSWORD"; then
            printf '%s[correctly failed with INVALID_PASSWORD]%s\n' "$c_ok" "$c_off"; xfail_ok=$((xfail_ok+1))
        else
            printf '%s[failed, but not with INVALID_PASSWORD]%s\n' "$c_err" "$c_off"; xfail_bad=$((xfail_bad+1))
        fi
    fi
    GARGS="$EFARGS"                       # restore USB for anything after
fi
else
echo; echo "(skipped BLE section: SKIP_BLE=1)"
fi

# ---------------------------------------------------------------------------
# USB lock: opt-in, because a run that dies while locked leaves every later verb
# demanding --password. Recovery is always available (the default password, or
# the ungated factory-reset), but it should be a deliberate choice, not a
# surprise. factory-reset itself is NOT exercised here: it clears the device's
# wifi credentials, which the harness cannot restore.
if [ "${TEST_ENCRYPTION:-0}" = 1 ]; then
    banner "AT-REST ENCRYPTION (opt-in: TEST_ENCRYPTION=1)"
    # Self-restoring: leaves encryption OFF and the key exactly as it found it.
    # if it has to create one, it destroys that one at the end and never touches a
    # key that was already there, because destroying THAT would make every
    # recording on the device unreadable.
    MADE_KEY=""
    if ! EFX $GARGS key show 2>&1 | grep -qE '^[0-9a-f]{64}$'; then
        # No key: enabling must be refused before anything else, since "enabled"
        # must never quietly mean "recording in the clear".
        xfail "enabling encryption without a key" INVALID_FUNCTION_CALL encryption on
        run "create the encryption key"    encryption create
        MADE_KEY=$(EFX $GARGS info 2>&1 | sed -n 's/^encryption key *: \([0-9a-f]*\).*/\1/p')
    fi
    if EFX $GARGS key show 2>&1 | grep -qE '^[0-9a-f]{64}$'; then
        run "enable encryption"            encryption on
        run "record encrypted"             record start ef-smoke-enc
        sleep 3
        run "stop"                         record stop
        sleep 1
        ENCLIST=$(EFX $GARGS record list 2>&1 || true)
        if printf '%s' "$ENCLIST" | grep -q "ef-smoke-enc.*\[encrypted\]"; then
            printf '%s[recording reported as encrypted]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
        else
            printf '%s[encryption enabled but the recording is not marked encrypted]%s\n' \
                "$c_err" "$c_off"; fail=$((fail+1))
        fi
        # The round trip is the point of the feature: a recording nobody can read
        # back is indistinguishable from a lost one, so prove the ciphertext
        # actually decrypts to a parseable MCAP before deleting it.
        ENC_DL="$DLDIR/ef-smoke-enc.enc"
        ENC_KEY="$DLDIR/ef-smoke-enc.key"
        ENC_OUT="$DLDIR/ef-smoke-enc.mcap"
        run "download the encrypted recording" download ef-smoke-enc "$ENC_DL"
        # --out writes 0600 and never echoes the key, which is the form an
        # operator is told to use; it also covers the refuse-to-overwrite guard.
        run "write the key to a file"      key show --out "$ENC_KEY"
        if [ -f "$ENC_KEY" ]; then
            KEYMODE=$(stat -c%a "$ENC_KEY" 2>/dev/null)
            if [ "$KEYMODE" = 600 ]; then
                printf '%s[key file is 0600]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
            else
                printf '%s[key file mode %s, expected 600]%s\n' "$c_err" "$KEYMODE" "$c_off"
                fail=$((fail+1))
            fi
        fi
        # Overwriting would silently destroy what may be another device's only key.
        xfail_rc "key show --out refuses to overwrite" key show --out "$ENC_KEY"

        if [ ! -x "$EFDEC" ]; then
            echo "${c_dim}(skip decrypt round trip: no ef-decrypt at $EFDEC; needs libssl-dev)${c_off}"
        elif [ ! -s "$ENC_DL" ]; then
            echo "${c_dim}(skip decrypt round trip: the download produced nothing)${c_off}"
        else
            # Downloaded bytes are the stored bytes, so an encrypted recording must
            # NOT already be a plain MCAP. If it is, encryption did not happen.
            if head -c 5 "$ENC_DL" | grep -q 'MCAP'; then
                printf '%s[downloaded "encrypted" recording is plaintext MCAP]%s\n' \
                    "$c_err" "$c_off"; fail=$((fail+1))
            else
                printf '%s[stored recording is ciphertext, not MCAP]%s\n' "$c_ok" "$c_off"
                pass=$((pass+1))
            fi

            printf '\n%s# decrypt it with the saved key%s\n%s$ ef-decrypt %s %s %s%s\n' \
                "$c_dim" "$c_off" "$c_cmd" "$ENC_DL" "$ENC_KEY" "$ENC_OUT" "$c_off"
            DEC_OUT="$("$EFDEC" "$ENC_DL" "$ENC_KEY" "$ENC_OUT" 2>&1)"; DEC_RC=$?
            printf '%s\n' "$DEC_OUT"
            # 0 is a clean file; 1 means truncated, which a cleanly stopped
            # recording must never be, so only 0 passes here.
            if [ $DEC_RC -eq 0 ]; then
                printf '%s[exit 0, clean end marker]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
            else
                printf '%s[ef-decrypt exit %d, expected 0]%s\n' "$c_err" "$DEC_RC" "$c_off"
                fail=$((fail+1))
            fi
            # The header names the key, so a mismatch here means the device stamped
            # an id that does not match the key it handed out.
            DEC_KEYID=$(printf '%s' "$DEC_OUT" | sed -n 's/^key_id *: *\([0-9a-f]*\).*/\1/p')
            INFO_KEYID=$(EFX $GARGS info 2>&1 | sed -n 's/^encryption key *: \([0-9a-f]*\).*/\1/p')
            if [ -n "$DEC_KEYID" ] && [ "$DEC_KEYID" = "$INFO_KEYID" ]; then
                printf '%s[file key_id %s matches the device]%s\n' "$c_ok" "$DEC_KEYID" "$c_off"
                pass=$((pass+1))
            else
                printf '%s[file key_id "%s" != device "%s"]%s\n' \
                    "$c_err" "$DEC_KEYID" "$INFO_KEYID" "$c_off"; fail=$((fail+1))
            fi
            if head -c 5 "$ENC_OUT" 2>/dev/null | grep -q 'MCAP'; then
                printf '%s[decrypted output is a plain MCAP]%s\n' "$c_ok" "$c_off"; pass=$((pass+1))
            else
                printf '%s[decrypted output is not an MCAP]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
            fi
            # Strongest available check: the decrypted bytes parse as a real
            # recording, not merely a file that starts with the right magic.
            if have python3 && python3 -c "import mcap" >/dev/null 2>&1; then
                if python3 -c "
import sys
from mcap.reader import make_reader
with open(sys.argv[1],'rb') as f:
    n = sum(1 for _ in make_reader(f).iter_messages())
print('messages:', n)
sys.exit(0 if n > 0 else 1)
" "$ENC_OUT"; then
                    printf '%s[decrypted MCAP parses and carries messages]%s\n' "$c_ok" "$c_off"
                    pass=$((pass+1))
                else
                    printf '%s[decrypted MCAP did not parse]%s\n' "$c_err" "$c_off"; fail=$((fail+1))
                fi
            else
                echo "${c_dim}(skip MCAP parse: need python3 + the mcap lib)${c_off}"
            fi

            # A wrong key must be refused at the header (exit 2), not produce
            # garbage, and must leave no partial output behind to be mistaken for
            # a decrypt.
            printf '\n%s# EXPECT-FAIL: decrypt with the wrong key%s\n' "$c_dim" "$c_off"
            # A fixed 64-hex-char key: the device generates random ones, so this
            # is never the installed key, and it keeps the run reproducible.
            printf '%s' \
              'baddecafbaddecafbaddecafbaddecafbaddecafbaddecafbaddecafbaddecaf' \
              > "$DLDIR/wrong.key"
            BAD_OUT="$("$EFDEC" "$ENC_DL" "$DLDIR/wrong.key" "$DLDIR/wrong.mcap" 2>&1)"; BAD_RC=$?
            printf '%s\n' "$BAD_OUT"
            if [ $BAD_RC -eq 2 ] && [ ! -e "$DLDIR/wrong.mcap" ]; then
                printf '%s[correctly refused, exit 2, no output left behind]%s\n' "$c_ok" "$c_off"
                xfail_ok=$((xfail_ok+1))
            else
                printf '%s[wrong key: exit %d, output present=%s; want exit 2 and no file]%s\n' \
                    "$c_err" "$BAD_RC" "$([ -e "$DLDIR/wrong.mcap" ] && echo yes || echo no)" "$c_off"
                xfail_bad=$((xfail_bad+1))
            fi
        fi

        run "delete"                       record delete ef-smoke-enc
        # A toggle must apply to the NEXT recording with no Configure in between;
        # resolving it only at Configure once made "encryption on" silently record
        # unencrypted.
        run "disable encryption"           encryption off
        run "record unencrypted"           record start ef-smoke-plain
        sleep 3
        run "stop"                         record stop
        sleep 1
        PLAINLIST=$(EFX $GARGS record list 2>&1 || true)
        if printf '%s' "$PLAINLIST" | grep -q "ef-smoke-plain.*\[unencrypted\]"; then
            printf '%s[toggle applies to the next recording, no Configure needed]%s\n' \
                "$c_ok" "$c_off"; pass=$((pass+1))
        else
            printf '%s[encryption off did not take effect on the next recording]%s\n' \
                "$c_err" "$c_off"; fail=$((fail+1))
        fi
        # The mirror of the ciphertext check above: with encryption off the
        # download must already be a plain MCAP, needing no key at all.
        run "download the unencrypted recording" \
            download ef-smoke-plain "$DLDIR/ef-smoke-plain.mcap"
        if head -c 5 "$DLDIR/ef-smoke-plain.mcap" 2>/dev/null | grep -q 'MCAP'; then
            printf '%s[unencrypted recording downloads as plain MCAP]%s\n' "$c_ok" "$c_off"
            pass=$((pass+1))
        else
            printf '%s[unencrypted recording is not a plain MCAP]%s\n' "$c_err" "$c_off"
            fail=$((fail+1))
        fi
        run "delete"                       record delete ef-smoke-plain

        # The key_id guard is the device's, so a wrong id must be refused even
        # though the CLI would have caught it too.
        xfail "delete with the wrong key_id" INVALID_FUNCTION_CALL \
              encryption delete --confirm deadbeef --yes
    else
        echo "(skipped: no encryption key and could not create one)"
    fi
    if [ -n "$MADE_KEY" ]; then
        run "destroy the key this test created" \
            encryption delete --confirm "$MADE_KEY" --yes
    fi
fi

# ---------------------------------------------------------------------------
if [ "${TEST_LOCK:-0}" = 1 ]; then
    banner "USB LOCK + SESSION UNLOCK (opt-in: TEST_LOCK=1)"
    run "read the encryption key (unlocked)"  key show
    run "lock USB"                            lock on

    # ef-cli DEFAULTS --password to 123456 (InitParameters::ble_password), so
    # OMITTING it still authenticates on a factory-default device. A negative test
    # must therefore pass a deliberately WRONG password; "no --password" proves
    # nothing and silently tested the happy path instead.
    xfail "gated verb, wrong password"        INVALID_PASSWORD --password "$WRONG_PW" record list
    # info stays readable, which is how a host learns it must authenticate.
    run "info still readable while locked"    info
    run "gated verb with --password"          --password "$TEST_BLE_PASSWORD" record list
    run "read the encryption key (locked)"    --password "$TEST_BLE_PASSWORD" key show

    # Session unlock: open the locked device for this power session. The grant is
    # deliberately NOT tied to the client that asked, so the next call needs no
    # password -- that is the feature, and also why it is worth asserting.
    run "session unlock"                      --password "$TEST_BLE_PASSWORD" lock off --session
    run "gated verb, no password needed"      record list
    run "info reports the third state"        info
    run "end the session unlock"              lock on --session
    xfail "gated again after ending it"       INVALID_PASSWORD --password "$WRONG_PW" record list

    run "unlock USB"                          --password "$TEST_BLE_PASSWORD" lock off
    run "gated verb after unlock"             record list
    # Meaningless on an open device, and refusing it is what stops an unlocked link
    # (authed with no password) planting an override that defeats the next lock.
    xfail "session unlock on an unlocked device" INVALID_FUNCTION_CALL lock off --session
fi

# ---------------------------------------------------------------------------
banner "SUMMARY"
printf '%ssuccess-path OK : %d%s\n' "$c_ok"  "$pass"      "$c_off"
printf '%ssuccess-path FAIL: %d%s\n' "$c_err" "$fail"      "$c_off"
printf '%sexpected-fail correct : %d%s\n' "$c_ok"  "$xfail_ok"  "$c_off"
printf '%sexpected-fail wrong   : %d%s\n' "$c_err" "$xfail_bad" "$c_off"
echo
echo "NOTE: OTA (update/abort-update) is out of scope for this harness (test it separately)."
echo "      reboot is opt-in via TEST_REBOOT=1 (runs last; ends the session)."
[ $((fail + xfail_bad)) -eq 0 ] && echo "${c_ok}ALL CHECKS BEHAVED AS EXPECTED$c_off" || echo "${c_err}REVIEW THE FLAGGED LINES ABOVE$c_off"

# ---------------------------------------------------------------------------
# Reboot LAST and only when explicitly opted in, it drops the device off the
# bus, so nothing can run after it. Clean up first and disarm the EXIT trap so
# cleanup doesn't then flail against a rebooting device.
if [ "${TEST_REBOOT:-0}" = 1 ]; then
    banner "REBOOT (opt-in: TEST_REBOOT=1)"
    cleanup
    trap - EXIT
    printf '\n%s# reboot the device (session ends here)%s\n%s$ ef-cli %s reboot%s\n' \
        "$c_dim" "$c_off" "$c_cmd" "$EFARGS" "$c_off"
    EFX $EFARGS reboot 2>&1
    echo "reboot issued, the device is restarting; reconnect and re-run to verify it came back."
else
    echo; echo "(reboot NOT run, set TEST_REBOOT=1 to include it as the final step.)"
fi
