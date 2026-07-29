////////////////////////////////////////////////////////////////////////////////
//
// File:      ef.cpp
// Purpose:   The `ef` command-line tool, one subcommand per SDK verb.
// Author:    Gianluca Bencomo
//
// Copyright (c) 2026, Remnant Robotics, Inc. All rights reserved.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <fcntl.h>     // O_EXCL: never overwrite an existing key file
#include <sys/stat.h>  // stat: tell a local bundle from a URL before dispatching
#include <termios.h>   // hidden password prompt
#include <unistd.h>   // isatty

#include <ef/Device.hpp>

using namespace ef;

namespace {

// A key must not survive in this process longer than it is needed. Not a security
// guarantee (the bytes were already on a socket and possibly a terminal), but it
// keeps them out of a core dump or a later allocation of the same page.
void wipe(std::vector<uint8_t>& k) {
    volatile uint8_t* p = k.data();
    for (size_t i = 0; i < k.size(); i++) p[i] = 0;
    k.clear();
}

// "<hex>\n", the one representation of a key, shared by the terminal and the
// file path so they can never drift.
std::string key_hex_line(const std::vector<uint8_t>& k) {
    std::string s;
    s.reserve(k.size() * 2 + 1);
    char b[3];
    for (uint8_t byte : k) { std::snprintf(b, sizeof b, "%02x", byte); s += b; }
    s += '\n';
    return s;
}

// Key to stdout, everything else to stderr, so `ef-cli key show > k` yields a file
// holding the key and nothing else.
void print_key_hex(const std::vector<uint8_t>& k) {
    std::string line = key_hex_line(k);
    std::fwrite(line.data(), 1, line.size(), stdout);
    // Flush before the caller writes more to stderr. Piped, stdout is block
    // buffered and the key would otherwise surface AFTER the instructions that
    // are meant to follow it.
    std::fflush(stdout);
    std::memset(&line[0], 0, line.size());
}

// Write "<hex>\n" to `path`, 0600 at creation so it is never briefly world
// readable, and never echo the key itself.
//
// O_EXCL rather than truncate: overwriting an existing key file is far more likely
// a wrong path or the wrong device than an intended re-export, and the file it
// would destroy may be the only copy of another device's key.
int write_key_file(const std::string& path, const std::vector<uint8_t>& k) {
    int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (fd < 0) {
        if (errno == EEXIST)
            std::fprintf(stderr, "key show: %s already exists; "
                                 "remove it first if you mean to replace it\n", path.c_str());
        else
            std::fprintf(stderr, "key show: %s: %s\n", path.c_str(), std::strerror(errno));
        return 1;
    }
    std::string line = key_hex_line(k);
    bool ok = write(fd, line.data(), line.size()) == static_cast<ssize_t>(line.size());
    close(fd);
    std::memset(&line[0], 0, line.size());
    if (!ok) {
        std::fprintf(stderr, "key show: writing %s failed\n", path.c_str());
        unlink(path.c_str());
        return 1;
    }
    std::fprintf(stderr, "key written to %s (mode 0600)\n", path.c_str());
    return 0;
}

void usage() {
    std::puts(
        "ef-cli: Efference M1 control tool\n"
        "\n"
        "  ef-cli [--ble <MAC>] [--device <id>] [--password <pw>] [--udp <host[:port]>]\n"
        "     [--verbose] <command> [args]\n"
        "\n"
        "flags:\n"
        "  --ble <MAC>                       connect over Bluetooth instead of USB\n"
        "  --device <id>                     pick one of several USB devices\n"
        "  --password <pw>                   control password, default 123456; needed on\n"
        "                                    BLE always and on USB once locked\n"
        "  --udp <host[:port]>               with --ble: device streams video+IMU to\n"
        "                                    this host over WiFi/UDP (default port 5005)\n"
        "  --verbose                         print the control-plane traffic\n"
        "\n"
        "commands:\n"
        "  list [--scan-ble]                 discover devices (USB, and BLE with --scan-ble)\n"
        "  info                              device information snapshot\n"
        "  config                            list enabled capture modes + codecs\n"
        "  config set <W> <H> <fps> <codec>  set capture config (idle only; codec:\n"
        "                                    raw|h264|h264hq|h265|h265hq)\n"
        "  calibration [--get]               show camera + IMU calibration\n"
        "  calibration --camera --set <fx> <fy> <cx> <cy> <xi> <alpha> <W> <H> [--rectify on|off] [--fov-scale <s>]\n"
        "                                    set camera intrinsics (idle only; always\n"
        "                                    published as metadata; --rectify on|off toggles\n"
        "                                    on-device FEC rectification, default off, needs FEC fw;\n"
        "                                    --fov-scale sets the rectified FOV, default 1.0, <1 wider)\n"
        "  calibration --camera [--rectify on|off] [--fov-scale <s>]\n"
        "                                    change only those flags, keeping the stored\n"
        "                                    intrinsics (no need to retype --set)\n"
        "  calibration --imu --mode <raw|calibrated|both>\n"
        "                                    select on-device IMU handling for the session\n"
        "                                    (calibrated applies M*S*(x-b) per sample)\n"
        "  calibration [--camera|--imu] --reset\n"
        "                                    reset calibration to factory default\n"
        "  storage                           free/total space on the recording store\n"
        "  state                             current DEVICE_STATE\n"
        "  health [--deep]                   run the on-device health sweep\n"
        "  record start [name] [--location LAT,LON[,ALT]]\n"
        "                                    start a device-local (eMMC) recording\n"
        "                                    (--location overrides just this recording)\n"
        "  record stop                       stop it\n"
        "  record status [name]              session status (+ storage, upload)\n"
        "  record list                       list device recordings\n"
        "  record delete <name>              delete a device recording\n"
        "  download <name> [dest]            pull a recording over USB/BLE (default <name>.mcap)\n"
        "  upload <name> <url>               upload a recording to a pre-signed URL\n"
        "  stop-upload <name>                kill a running upload\n"
        "  check-update                      ask the update service what to run\n"
        "  update                            update to whatever the service offers\n"
        "  update --url <url>                update from an explicit URL (no service call)\n"
        "  update --file <update.eff>        update from a local bundle over USB\n"
        "  abort-update                      cancel an update in progress\n"
        "  wifi add <ssid> [psk [country]]   provision WiFi (quote an SSID with spaces;\n"
        "                                    omit <psk> for a hidden prompt; \"US\" unlocks 5 GHz)\n"
        "  wifi remove <ssid> | select <ssid>\n"
        "  wifi list                         saved networks (marks the connected one)\n"
        "  wifi scan                         access points in range (not while recording)\n"
        "  wifi status                       current association\n"
        "  set-password <new>                rekey the control password (unlocked USB)\n"
        "  set-password <old> <new>          rekey the control password (BLE or locked USB)\n"
        "  lock on|off [--session]           lock/unlock the USB control plane\n"
        "                                    (--session: this power session only,\n"
        "                                     re-locks when power is lost)\n"
        "  encryption on|off                 AES-256 encrypt new recordings\n"
        "  encryption create                 generate the key; SHOWN ONCE, keep it\n"
        "  encryption delete                 show the key and how to destroy it\n"
        "  encryption delete --confirm <id> [--yes]\n"
        "                                    destroy it (recordings become unreadable);\n"
        "                                    --yes is required without a terminal\n"
        "  key show [--out <file>]           print the key, or write it to a 0600 file\n"
        "  factory-reset [--yes]             defaults + unlock; DESTROYS the encryption key\n"
        "  sync-time                         set the device clock from the host\n"
        "  time                              read the device wall clock\n"
        "  location                          read the device's current location\n"
        "  location set <lat> <lon> [alt]    persist the device location (all recordings)\n"
        "  reboot                            reboot the device\n");
}

// True when the device gates this link and we failed to authenticate: the point
// where a human should be told, since the library stays silent by design.
bool di_locked_unauth(const Device& dev) {
    if (dev.is_authenticated()) return false;
    const DeviceInformation& i = dev.get_device_information();
    // A session-unlocked device answers gated verbs without a password, so the
    // "only info/state/storage" warning would be false there.
    return (i.usb_locked && !i.session_unlocked) ||
           i.input_type == INPUT_TYPE::STREAM;
}

// `detail` is whoever refused saying why (the device, or the update service): the
// ERROR_CODE is a category, and the specific reason is usually the actionable part.
int fail(ERROR_CODE ec, const char* what, const std::string& detail = "") {
    std::fprintf(stderr, "%s failed: %s\n", what, to_string(ec));
    if (!detail.empty()) std::fprintf(stderr, "  reason: %s\n", detail.c_str());
    // A permissions failure means the device is on the bus but not openable, almost
    // always a missing udev rule. Point the user at the fix.
    if (ec == ERROR_CODE::INSUFFICIENT_PERMISSIONS)
        std::fprintf(stderr,
            "  Device is connected but not accessible (USB permissions).\n"
            "  Fix: run './build.sh --udev' once (installs a udev rule, needs sudo),\n"
            "  then replug the device. Alternatively, run this command with sudo.\n");
    return 1;
}

// Renders `ef-cli update` as one line per phase: the percent redraws in place while
// a phase runs, then the line is committed when the next phase begins. Uses '\r' on a
// TTY; on a redirected stream it prints one plain line per phase. Errors go on their
// own line. Verification is collapsed to a single line just before "ready".
struct UpdatePrinter {
    bool        tty  = isatty(fileno(stdout));
    bool        have = false;                 // a phase line is currently open
    std::string cur;                          // current display label
    std::string last_msg;                     // last device message (for error text)
    std::chrono::steady_clock::time_point phase_start;

    // Consumer-facing label. The wire push and the device's network fetch both read as
    // "downloading" (the update coming down to the box) and merge into one line; the
    // slot write reads as "installing".
    static std::string display(const UpdateStatus& u) {
        switch (u.state) {
            case UPDATE_STATE::CHECKING:       return "checking";
            case UPDATE_STATE::UPLOADING:      return "downloading";
            case UPDATE_STATE::DOWNLOADING:
                return u.message.find("Writing") != std::string::npos ? "installing"
                                                                      : "downloading";
            case UPDATE_STATE::VERIFYING:      return "verifying";
            case UPDATE_STATE::READY_TO_APPLY: return "ready";
            case UPDATE_STATE::APPLYING:       return "applying";
            case UPDATE_STATE::REBOOTING:      return "rebooting";
            case UPDATE_STATE::RECONNECTING:   return "reconnect";
        }
        return "update";
    }

    long elapsed() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::steady_clock::now() - phase_start).count();
    }

    // Finalize the current phase as "[label] status". A message is shown only for a
    // failure (the reason); success lines are just the label + "done".
    void commit(const char* status, const std::string& msg = "") {
        const char* sep = msg.empty() ? "" : "  ";
        if (tty) std::printf("\r[%-11s] %s%s%s\033[K\n", cur.c_str(), status, sep, msg.c_str());
        else     std::printf("[%-11s] %s%s%s\n",         cur.c_str(), status, sep, msg.c_str());
        std::fflush(stdout);
    }

    // Redraw the in-progress line (TTY only): label + percent, or "...." + ticking
    // elapsed when there is no real percentage (indeterminate, or stuck at 0 = a device
    // that never reports size, e.g. v0.3). No device message: the label says enough.
    void live(int progress) {
        if (!tty) return;
        long secs = elapsed();
        if (progress >= 0 && progress <= 100 && !(progress == 0 && secs >= 3))
            std::printf("\r[%-11s] %3d%%\033[K", cur.c_str(), progress);
        else
            std::printf("\r[%-11s] .... (%lds)\033[K", cur.c_str(), secs);
        std::fflush(stdout);
    }

    void enter(const std::string& disp) {
        cur = disp; have = true;
        phase_start = std::chrono::steady_clock::now();
    }

    void on(const UpdateStatus& u) {
        if (!u.message.empty()) last_msg = u.message;   // capture reason (incl. terminal)
        // A terminal error: close the open phase as FAIL with the reason.
        if (u.last_error != ERROR_CODE::SUCCESS &&
            u.last_error != ERROR_CODE::DEVICE_UP_TO_DATE) {
            if (have) { commit("FAIL", last_msg); have = false; }
            return;
        }
        if (!u.active) return;
        std::string disp = display(u);

        // "checking" and "verifying" are internal detail with no line of their own; keep
        // the current line ticking so a slow hash/verify reads as activity, not a stall.
        // A single "verifying" line is emitted when the device reaches "ready" (the user
        // doesn't need "ready" called out separately). Suppressing "checking" also merges
        // the host push and the device's fetch into one "downloading" line.
        if (disp == "checking" || disp == "verifying") { if (have) live(-1); return; }
        if (disp == "ready") {
            if (have) { commit("done"); have = false; }
            cur = "verifying"; commit("done");
            return;
        }
        if (!have || disp != cur) {
            if (have) commit("done");   // finalize the previous phase
            enter(disp);
        }
        // "installing" (slot write + hash) reports no real percentage; show a ticking
        // spinner so it never freezes at 100%.
        live(disp == "installing" ? -1 : u.progress);
    }

    // After a successful update, close the last open phase.
    void done() { if (have) commit("done"); have = false; }
};

// Classify a generic FAILED_TO_UPDATE by the device's message. Match BUSY before
// NETWORK: the lock also fails in the download phase.
enum class UpdateFailure { GENERIC, NETWORK, BUSY };
UpdateFailure classify_update_failure(ERROR_CODE ec, const std::string& msg) {
    if (ec != ERROR_CODE::FAILED_TO_UPDATE) return UpdateFailure::GENERIC;
    auto has = [&](const char* s) { return msg.find(s) != std::string::npos; };
    if (has("already running") || has("Download refused")) return UpdateFailure::BUSY;
    if (has("connectivity") || has("Failed to download") ||
        has("interrupted (network"))
        return UpdateFailure::NETWORK;
    return UpdateFailure::GENERIC;
}

void print_info(const DeviceInformation& i) {
    std::printf("serial           : %s\n", i.serial.c_str());
    // serial_number is the numeric form of `serial`, 0 for non-decimal serials
    // (e.g. "M1BENCH002"). Show it only when set, so it reads as one identity.
    if (i.serial_number != 0)
        std::printf("serial_number    : %u\n", i.serial_number);
    std::printf("model            : %s\n", to_string(i.model));
    // Prefer the device's human-readable version string ("vXX.XX.XX"); fall back
    // to the monotonic int for older devices that don't report a usable string.
    if (!i.firmware_version_str.empty() && i.firmware_version_str != "unknown")
        std::printf("firmware_version : v%s\n", i.firmware_version_str.c_str());
    else
        std::printf("firmware_version : %u\n", i.firmware_version);
    std::printf("input_type       : %s\n", to_string(i.input_type));
    const CameraConfiguration& c = i.camera_configuration;
    std::printf("camera           : %dx%d @ %d fps, %s\n",
                c.resolution.width, c.resolution.height, c.fps,
                to_string(c.compression));
    // fx<=0 is the factory-default (uncalibrated) sentinel.
    const CalibrationParameters& cal = c.calibration;
    std::printf("camera calibration : %s (%s)\n", to_string(cal.model),
                cal.fx > 0.0 ? "calibrated" : "factory-default, uncalibrated");
    std::printf("  %-10s = fx=%.2f fy=%.2f cx=%.2f cy=%.2f xi=%.4f alpha=%.4f\n",
                "intrinsics", cal.fx, cal.fy, cal.cx, cal.cy, cal.xi, cal.alpha);
    std::printf("  %-10s = %dx%d\n", "size", cal.width, cal.height);
    std::printf("  %-10s = %s   fov-scale = %.2f\n",
                "rectify", cal.rectify ? "on" : "off", cal.fov_scale);
    const SensorsConfiguration& s = i.sensors_configuration;
    std::printf("accelerometer    : %s\n", to_string(s.accelerometer.state));
    std::printf("gyroscope        : %s\n", to_string(s.gyroscope.state));
    // Full IMU field calibration: a* = M*S*(x - b) (gyro is bias-only). Factory
    // default is zero bias + identity scale-misalignment; "field" iff any bias != 0
    // or the accel scale-misalignment differs from identity.
    {
        bool nz = s.accel_bias[0] || s.accel_bias[1] || s.accel_bias[2] ||
                  s.gyro_bias[0]  || s.gyro_bias[1]  || s.gyro_bias[2];
        for (int k = 0; k < 9 && !nz; ++k)
            if (s.accel_scale_misalign[k] != (k % 4 == 0 ? 1.0 : 0.0)) nz = true;
        const auto& a = s.accel_scale_misalign;
        const auto& g = s.gyro_scale_misalign;
        const auto& T = s.camera_imu_transform.m;
        std::printf("imu calibration  : %s\n", nz ? "field" : "factory-default (uncalibrated)");
        std::printf("  accel bias  [m/s^2]  = [% .6f % .6f % .6f]\n",
                    s.accel_bias[0], s.accel_bias[1], s.accel_bias[2]);
        std::printf("  gyro  bias  [rad/s]  = [% .6f % .6f % .6f]\n",
                    s.gyro_bias[0], s.gyro_bias[1], s.gyro_bias[2]);
        std::printf("  accel M*S (row-major)= [% .6f % .6f % .6f]\n"
                    "                         [% .6f % .6f % .6f]\n"
                    "                         [% .6f % .6f % .6f]\n",
                    a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]);
        std::printf("  gyro  M*S (row-major)= [% .6f % .6f % .6f] [% .6f % .6f % .6f] [% .6f % .6f % .6f]\n",
                    g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7], g[8]);
        std::printf("  accel noise=%.6f  bias_rw=%.6f  tau=%.4f\n",
                    s.accelerometer.noise_density, s.accelerometer.bias_random_walk,
                    s.accelerometer.tau);
        std::printf("  gyro  noise=%.6f  bias_rw=%.6f  tau=%.4f\n",
                    s.gyroscope.noise_density, s.gyroscope.bias_random_walk,
                    s.gyroscope.tau);
        std::printf("  cam->imu (row-major) = [% .5f % .5f % .5f % .5f]\n"
                    "                         [% .5f % .5f % .5f % .5f]\n"
                    "                         [% .5f % .5f % .5f % .5f]\n"
                    "                         [% .5f % .5f % .5f % .5f]\n",
                    T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7],
                    T[8], T[9], T[10], T[11], T[12], T[13], T[14], T[15]);
        std::printf("  time offset [ns]     = %lld\n", s.time_offset_ns);
    }
    const WirelessConfiguration& w = i.wireless;
    std::printf("wifi mac         : %s\n",
                w.wifi_mac_address.empty() ? "(unprovisioned)" : w.wifi_mac_address.c_str());
    std::printf("bt mac           : %s%s\n",
                w.bt_mac_address.empty() ? "(unprovisioned)" : w.bt_mac_address.c_str(),
                w.bt_paired ? " (paired)" : "");
    if (w.wifi_connected)
        std::printf("wifi             : connected \"%s\" (%s, rssi %d)\n",
                    w.wifi_ssid.c_str(), w.wifi_ip_address.c_str(), w.wifi_rssi);
    else
        std::printf("wifi             : not connected\n");
    for (const auto& n : w.saved_networks)
        std::printf("saved network    : %s\n", n.c_str());
    // Access + at-rest state. Both come from the ungated GetDeviceInformation, so
    // `info` answers even on a locked device you cannot otherwise talk to; that is
    // how an operator discovers WHY everything else is refusing them.
    // Three states, not two: reporting a session-unlocked device as "LOCKED" would
    // send an operator hunting for a password it is not currently asking for.
    std::printf("usb access       : %s\n",
                !i.usb_locked      ? "unlocked"
                : i.session_unlocked ? "LOCKED, open for this session "
                                       "(re-locks when power is lost)"
                                     : "LOCKED (password required)");
    std::printf("encryption       : %s\n",
                i.encryption_enabled ? "ON (new recordings encrypted)" : "off");
    // The key_id, never the key: it is how an operator matches a device against
    // the copy they saved, and how they tell which key opens a given recording.
    // The id is a gated field: present is readable pre-auth, the id is not.
    if (!i.encryption_key_present)
        std::printf("encryption key   : none (run 'encryption create')\n");
    else if (i.encryption_key_id.empty())
        std::printf("encryption key   : present (password required for id)\n");
    else
        std::printf("encryption key   : %s (AES-256-GCM)\n", i.encryption_key_id.c_str());
}

// Parse "LAT,LON[,ALT]" into a Location (exception-free). Returns false on any
// malformed token or fewer than two components.
bool parse_location(const std::string& s, Location& loc) {
    double v[3] = {0, 0, 0};
    int n = 0;
    const char* p = s.c_str();
    while (n < 3 && *p) {
        char* end = nullptr;
        v[n] = std::strtod(p, &end);
        if (end == p) return false;            // no number consumed
        n++;
        p = end;
        if (*p == ',') p++;
        else if (*p != '\0') return false;     // trailing junk
    }
    if (n < 2) return false;                    // need at least lat,lon
    loc.latitude  = v[0];
    loc.longitude = v[1];
    loc.altitude  = (n >= 3) ? v[2] : 0.0;
    return true;
}

// show_target: print the DEVICE_LOCAL/HOST_FILE column. Meaningful for `record
// status`; pointless for `record list` (entries are always DEVICE_LOCAL).
void print_recording(const RecordingStatus& r, bool show_target = true) {
    std::printf("%-24s ", r.name.c_str());
    if (show_target) std::printf("%-12s ", to_string(r.target));
    // The stop reason qualifies "complete": e.g. "complete (disk full)".
    // UNSPECIFIED (in progress, or firmware/recordings predating the field)
    // prints nothing extra.
    const char* why = "";
    switch (r.stopped_reason) {
        case STOP_REASON::DISK_FULL:   why = " (disk full)"; break;
        case STOP_REASON::WRITE_ERROR: why = " (write error)"; break;
        // r.partial: an unrepaired torso served as-is (decrypts with a
        // truncated-tail warning) — say so rather than implying a finalized file.
        case STOP_REASON::INTERRUPTED:
            why = r.partial ? " (partial: recovered after power loss)"
                            : " (recovered after power loss)";
            break;
        default: break;
    }
    std::printf("%s%s  %.1f MB, %llu frames, %llu ms",
                r.recording ? "RECORDING" : "complete",
                r.recording ? "" : why,
                r.bytes / (1024.0 * 1024.0), (unsigned long long)r.frames,
                (unsigned long long)r.duration_ms);
    std::printf("  %s", r.encrypted ? "[encrypted]" : "[unencrypted]");
    if (r.upload == UPLOAD_STATE::RUNNING)
        std::printf("  [upload %.1f/%.1f MB]",
                    r.upload_bytes_sent  / (1024.0 * 1024.0),
                    r.upload_bytes_total / (1024.0 * 1024.0));
    if (r.last_error != ERROR_CODE::SUCCESS)
        std::printf("  [last_error %s]", to_string(r.last_error));
    std::printf("\n");
    if (r.storage_total_bytes)
        std::printf("storage: %llu / %llu MiB free\n",
                    (unsigned long long)(r.storage_free_bytes >> 20),
                    (unsigned long long)(r.storage_total_bytes >> 20));
}

// Read a secret from the terminal, echoing '*' per keystroke (backspace edits,
// Ctrl-C/Ctrl-D cancels -> false). Piped stdin falls back to a plain line read.
// The prompt and echo go to stderr so stdout stays clean for scripting.
bool prompt_secret(const char* prompt, std::string& out) {
    out.clear();
    std::fprintf(stderr, "%s", prompt);
    if (!isatty(STDIN_FILENO)) {
        int c;
        while ((c = std::fgetc(stdin)) != EOF && c != '\n') out.push_back((char)c);
        return !out.empty() || c == '\n';
    }
    termios saved{};
    if (tcgetattr(STDIN_FILENO, &saved) != 0) return false;
    termios raw = saved;
    // ISIG off too: ^C is read as a byte and handled as cancel, so the terminal
    // is always restored (a signal mid-prompt would leave echo disabled).
    raw.c_lflag &= ~(ECHO | ICANON | ISIG);
    raw.c_cc[VMIN] = 1;
    raw.c_cc[VTIME] = 0;
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw);
    bool ok = true;
    for (;;) {
        char c;
        if (read(STDIN_FILENO, &c, 1) != 1) { ok = false; break; }
        if (c == '\n' || c == '\r') break;
        if (c == 3 || c == 4) { ok = false; break; }   // ^C / ^D
        if (c == 0x7f || c == '\b') {
            if (!out.empty()) { out.pop_back(); std::fputs("\b \b", stderr); }
        } else if ((unsigned char)c >= 0x20) {
            out.push_back(c);
            std::fputc('*', stderr);
        }
    }
    tcsetattr(STDIN_FILENO, TCSAFLUSH, &saved);
    std::fputc('\n', stderr);
    if (!ok) out.clear();
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    InitParameters init;
    std::vector<std::string> args;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "--ble") && i + 1 < argc) {
            init.input_type  = INPUT_TYPE::STREAM;
            init.ble_address = argv[++i];
        } else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) {
            init.device_id = std::atoi(argv[++i]);
        } else if (!std::strcmp(argv[i], "--password") && i + 1 < argc) {
            init.ble_password = argv[++i];
        } else if (!std::strcmp(argv[i], "--udp") && i + 1 < argc) {
            std::string hp = argv[++i];
            std::string::size_type colon = hp.rfind(':');
            if (colon != std::string::npos) {
                init.udp_port = (uint16_t)std::atoi(hp.c_str() + colon + 1);
                hp.resize(colon);
            }
            init.udp_host = hp;
        } else if (!std::strcmp(argv[i], "--verbose")) {
            init.verbose = 1;
        } else {
            args.push_back(argv[i]);
        }
    }
    if (args.empty()) { usage(); return 2; }
    const std::string& cmd = args[0];

    // ---- discovery runs without open() ---------------------------------------
    if (cmd == "list") {
        bool ble = args.size() > 1 && args[1] == "--scan-ble";
        auto devs = Device::get_device_list(ble);
        if (devs.empty()) { std::puts("no devices found"); return 1; }
        for (const auto& d : devs) {
            if (d.input_type == INPUT_TYPE::USB) {
                std::printf("USB  device_id=%d  serial=%s", d.device_id, d.serial.c_str());
                // Discovery can't read vendor storage, so briefly open the device
                // to surface its WiFi/BT MAC (useful when connected over the wire).
                Device probe;
                InitParameters pi;
                pi.input_type = INPUT_TYPE::USB;
                pi.device_id  = d.device_id;
                if (probe.open(pi) == ERROR_CODE::SUCCESS) {
                    const WirelessConfiguration& w =
                        probe.get_device_information().wireless;
                    if (!w.wifi_mac_address.empty())
                        std::printf("  wifi_mac=%s", w.wifi_mac_address.c_str());
                    if (!w.bt_mac_address.empty())
                        std::printf("  bt_mac=%s", w.bt_mac_address.c_str());
                }
                std::printf("\n");
            } else {
                std::printf("BLE  %s  (%s)\n", d.ble_address.c_str(), d.ble_name.c_str());
            }
        }
        return 0;
    }
    if (cmd == "help" || cmd == "--help" || cmd == "-h") { usage(); return 0; }

    // ---- everything else talks to one device ---------------------------------
    Device dev;
    ERROR_CODE ec = dev.open(init);
    if (ec != ERROR_CODE::SUCCESS)
        // One control client per cable: a second USB open is refused, not queued.
        return fail(ec, "open",
                    ec == ERROR_CODE::DEVICE_BUSY
                        ? "device is in use by another process; close the other "
                          "SDK app / ef-cli, or reach it over BLE (--ble)"
                        : "");

    // The library keeps quiet about a refused password (no stderr from a lib);
    // the CLI is where a human is, so warn here instead.
    if (di_locked_unauth(dev))
        std::fprintf(stderr,
            "ef-cli: control password not accepted; only info, state, storage"
            " and factory-reset will answer\n");
    if (cmd == "info") {
        print_info(dev.get_device_information());
        return 0;
    }
    if (cmd == "config" && args.size() > 1 && args[1] == "set") {
        if (args.size() < 6) {
            std::fprintf(stderr,
                "usage: ef-cli config set <width> <height> <fps> <codec>\n"
                "  codec: raw | h264 | h264hq | h265 | h265hq\n");
            return 2;
        }
        int w   = std::atoi(args[2].c_str());
        int h   = std::atoi(args[3].c_str());
        int fps = std::atoi(args[4].c_str());
        const std::string& cs = args[5];
        COMPRESSION_MODE codec;
        if      (cs == "raw")    codec = COMPRESSION_MODE::RAW;
        else if (cs == "h264")   codec = COMPRESSION_MODE::H264;
        else if (cs == "h264hq") codec = COMPRESSION_MODE::H264_HQ;
        else if (cs == "h265")   codec = COMPRESSION_MODE::H265;
        else if (cs == "h265hq") codec = COMPRESSION_MODE::H265_HQ;
        else {
            std::fprintf(stderr, "unknown codec '%s' (raw|h264|h264hq|h265|h265hq)\n",
                         cs.c_str());
            return 2;
        }
        ec = dev.set_configuration(w, h, fps, codec);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "config set");
        std::printf("configured %dx%d @ %d fps, %s\n", w, h, fps, to_string(codec));
        return 0;
    }
    if (cmd == "config") {
        const DeviceInformation& di = dev.get_device_information();
        const CameraConfiguration& c = di.camera_configuration;
        std::printf("current          : %dx%d @ %d fps, %s\n",
                    c.resolution.width, c.resolution.height, c.fps,
                    to_string(c.compression));
        // Both change the bytes on disk, and neither shows in the mode list.
        std::printf("encryption       : %s\n",
                    di.encryption_enabled ? "on (AES-256)" : "off");
        std::printf("rectify          : %s   fov-scale = %.2f\n",
                    c.calibration.rectify ? "on" : "off", c.calibration.fov_scale);
        const Capabilities& caps = di.capabilities;
        if (caps.modes.empty()) {
            std::puts("supported modes  : (none advertised)");
        } else {
            std::puts("supported modes  :");
            for (const auto& m : caps.modes)
                std::printf("  %dx%d @ %d fps\n",
                            m.resolution.width, m.resolution.height, m.fps);
        }
        if (!caps.codecs.empty()) {
            std::printf("supported codecs : ");
            for (size_t k = 0; k < caps.codecs.size(); k++)
                std::printf("%s%s", caps.codecs[k].c_str(),
                            k + 1 < caps.codecs.size() ? ", " : "");
            std::printf("\n");
        }
        return 0;
    }
    if (cmd == "calibration") {
        bool want_camera = false, want_imu = false;
        bool do_get = false, do_set = false, do_reset = false;
        bool set_rectify = false;   // on-device FEC rectification
        double set_fov_scale = 1.0;    // rectified-output FOV scale
        bool rectify_given = false; // flag explicitly passed
        bool fov_scale_given  = false; // flag explicitly passed
        std::string imu_mode;       // --mode <raw|calibrated|both>
        std::vector<std::string> vals;
        for (size_t i = 1; i < args.size(); i++) {
            const std::string& a = args[i];
            if      (a == "--get")        do_get   = true;
            else if (a == "--set")        do_set   = true;
            else if (a == "--reset")      do_reset = true;
            else if (a == "--camera")     want_camera = true;
            else if (a == "--imu")        want_imu    = true;
            else if (a == "--rectify") {
                // Explicit on|off (default off if the flag is omitted entirely).
                if (i + 1 < args.size() &&
                    (args[i + 1] == "on" || args[i + 1] == "off")) {
                    set_rectify = (args[++i] == "on");
                    rectify_given = true;
                } else {
                    std::fprintf(stderr, "--rectify wants 'on' or 'off'\n");
                    return 2;
                }
            }
            else if (a == "--fov-scale") {
                if (i + 1 >= args.size()) { std::fprintf(stderr, "--fov-scale wants a value\n"); return 2; }
                char* end = nullptr;
                set_fov_scale = std::strtod(args[i + 1].c_str(), &end);
                // Reject unparseable/non-positive: 0 is the "unset" sentinel, not a value.
                if (end == args[i + 1].c_str() || *end != '\0' || set_fov_scale <= 0.0) {
                    std::fprintf(stderr, "--fov-scale wants a positive number\n"); return 2;
                }
                ++i; fov_scale_given = true;
            }
            else if (a == "--mode" && i + 1 < args.size()) imu_mode = args[++i];
            else                          vals.push_back(a);
        }
        // Rectify flags with stray positional args (e.g. mimicking --set) is a
        // mistake; don't silently drop the flag. --set takes 8 vals; toggle takes none.
        if ((rectify_given || fov_scale_given) && !do_set && !vals.empty()) {
            std::fprintf(stderr, "unexpected arguments; pass --rectify/--fov-scale alone "
                         "to toggle, or use --set for full intrinsics\n");
            return 2;
        }
        // Toggle rectify without re-typing intrinsics: read the current
        // calibration, change only the named flag(s), resend.
        if ((rectify_given || fov_scale_given) && vals.empty() &&
            !do_set && !do_reset && !do_get) {
            if (want_imu) {
                std::fprintf(stderr, "--rectify/--fov-scale apply to --camera only\n");
                return 2;
            }
            const DeviceInformation& di = dev.get_device_information();
            CalibrationParameters c = di.camera_configuration.calibration;
            // Keep the resolution the intrinsics were calibrated at (full sensor); the
            // session resolution can differ (1080p crop) and must not overwrite it.
            int w = c.width, h = c.height;
            const CameraConfiguration& cam = di.camera_configuration;
            if (w <= 0 || h <= 0) { w = cam.resolution.width; h = cam.resolution.height; }
            if (c.fx <= 0.0)
                std::fprintf(stderr, "warning: device is uncalibrated (fx<=0); the flag is "
                             "persisted but rectification has no intrinsics to use\n");
            c.model = LENS_DISTORTION_MODEL::DS;
            if (rectify_given) c.rectify = set_rectify;
            if (fov_scale_given)  c.fov_scale  = set_fov_scale;
            ec = dev.set_camera_calibration(c, w, h);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "calibration set");
            std::printf("rectify: %s (on-device FEC); fov-scale: %.2f (applies next capture session)\n",
                        c.rectify ? "on" : "off", c.fov_scale);
            if (c.rectify)
                std::printf("note: device dewarps frames to rectilinear before encoding "
                            "(recordings ship rectified from the next session).\n");
            return 0;
        }
        // IMU capture-mode select: standalone (re-sends current geometry + the mode).
        if (want_imu && !imu_mode.empty() && !do_set && !do_reset) {
            IMU_DATA m;
            if      (imu_mode == "raw")        m = IMU_DATA::RAW;
            else if (imu_mode == "calibrated") m = IMU_DATA::CALIBRATED;
            else if (imu_mode == "both")       m = IMU_DATA::BOTH;
            else { std::fprintf(stderr, "unknown imu mode '%s' (raw|calibrated|both)\n",
                                imu_mode.c_str()); return 2; }
            const DeviceInformation& di = dev.get_device_information();
            const CameraConfiguration& c = di.camera_configuration;
            ec = dev.set_configuration(c.resolution.width, c.resolution.height,
                                       c.fps, c.compression, m);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "imu mode set");
            std::printf("imu capture mode = %s (applies next session)\n", to_string(m));
            return 0;
        }
        if (do_get || (!do_set && !do_reset)) {
            // --get (also the default with no verb): show camera + IMU calibration.
            const DeviceInformation& di = dev.get_device_information();
            const CalibrationParameters& cc = di.camera_configuration.calibration;
            const CameraConfiguration&   cam = di.camera_configuration;
            // fx<=0 (camera) / all-zero (IMU) are the factory-default sentinels.
            std::printf("camera calibration: %s\n",
                        cc.fx > 0.0 ? "calibrated" : "factory-default (uncalibrated)");
            std::printf("  model  : %s\n", to_string(cc.model));
            std::printf("  rectify: %s (on-device FEC)\n", cc.rectify ? "on" : "off");
            std::printf("  fov-scale : %.2f\n", cc.fov_scale);
            std::printf("  size   : %dx%d\n", cam.resolution.width, cam.resolution.height);
            std::printf("  fx=%.4f fy=%.4f cx=%.4f cy=%.4f xi=%.6f alpha=%.6f\n",
                        cc.fx, cc.fy, cc.cx, cc.cy, cc.xi, cc.alpha);
            const SensorsConfiguration& s = di.sensors_configuration;
            bool imu_cal = s.accel_bias[0] || s.accel_bias[1] || s.accel_bias[2] ||
                           s.gyro_bias[0] || s.gyro_bias[1] || s.gyro_bias[2] ||
                           s.accelerometer.noise_density > 0.f;
            std::printf("imu calibration: %s\n",
                        imu_cal ? "field" : "factory-default (uncalibrated)");
            std::printf("  accel bias =[% .5f % .5f % .5f] m/s^2\n",
                        s.accel_bias[0], s.accel_bias[1], s.accel_bias[2]);
            std::printf("  gyro  bias =[% .6f % .6f % .6f] rad/s\n",
                        s.gyro_bias[0], s.gyro_bias[1], s.gyro_bias[2]);
            std::printf("  accel noise=%.6f  gyro noise=%.6f\n",
                        s.accelerometer.noise_density, s.gyroscope.noise_density);
            return 0;
        }
        if (do_set) {
            if (want_imu) {
                std::fprintf(stderr,
                    "IMU field calibration is run via the calibrate_imu tutorial, not this flag:\n"
                    "  tutorials/calibrate_imu - captures the still + tumble procedure,\n"
                    "  solves, and writes it via set_imu_calibration.\n");
                return 2;
            }
            if (!want_camera) { std::fprintf(stderr, "specify --camera\n"); return 2; }
            if (vals.size() != 8) {
                std::fprintf(stderr,
                    "usage: ef-cli calibration --camera --set <fx> <fy> <cx> <cy> "
                    "<xi> <alpha> <width> <height>\n");
                return 2;
            }
            CalibrationParameters c;
            c.model = LENS_DISTORTION_MODEL::DS;
            c.fx = std::strtod(vals[0].c_str(), nullptr);
            c.fy = std::strtod(vals[1].c_str(), nullptr);
            c.cx = std::strtod(vals[2].c_str(), nullptr);
            c.cy = std::strtod(vals[3].c_str(), nullptr);
            c.xi = std::strtod(vals[4].c_str(), nullptr);
            c.alpha = std::strtod(vals[5].c_str(), nullptr);
            c.rectify = set_rectify;   // on-device FEC rectification; default off
            c.fov_scale  = set_fov_scale;    // rectified-output FOV scale; default 1.0
            int w = std::atoi(vals[6].c_str());
            int h = std::atoi(vals[7].c_str());
            ec = dev.set_camera_calibration(c, w, h);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "calibration set");
            std::printf("camera calibration set; intrinsics published as recording "
                        "metadata (applies next capture session)\n");
            if (c.rectify)
                std::printf("note: on-device rectification enabled; the device dewarps frames "
                            "to rectilinear before encoding (from the next capture session).\n");
            return 0;
        }
        // do_reset: default to both sensors when neither is named.
        bool rc = want_camera || (!want_camera && !want_imu);
        bool ri = want_imu    || (!want_camera && !want_imu);
        ec = dev.reset_calibration(rc, ri);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "calibration reset");
        std::printf("calibration reset to factory default%s%s\n",
                    rc ? " [camera]" : "", ri ? " [imu]" : "");
        return 0;
    }
    if (cmd == "storage") {
        uint64_t free_b = 0, total_b = 0;
        ec = dev.get_storage(free_b, total_b);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "storage");
        uint64_t used_b = total_b > free_b ? total_b - free_b : 0;
        double pct = total_b ? 100.0 * (double)used_b / (double)total_b : 0.0;
        std::printf("userdata: %llu / %llu MiB free (%.1f%% used)\n",
                    (unsigned long long)(free_b  >> 20),
                    (unsigned long long)(total_b >> 20), pct);
        return 0;
    }
    if (cmd == "state") {
        std::printf("%s\n", to_string(dev.get_state()));
        return 0;
    }
    if (cmd == "health") {
        bool deep = args.size() > 1 && args[1] == "--deep";
        std::printf("running %s health sweep...\n", deep ? "deep" : "shallow");
        HealthStatus h;
        ec = dev.health_check(h, deep);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "health_check");
        std::printf("overall=%s camera=%s imu=%s checks=%zu\n",
                    h.passed ? "PASS" : "FAIL", to_string(h.camera),
                    to_string(h.imu), h.checks.size());
        for (const auto& c : h.checks)
            std::printf("  [%s] %s%s%s\n", c.passed ? "PASS" : "FAIL", c.name.c_str(),
                        c.detail.empty() ? "" : ": ", c.detail.c_str());
        return h.passed ? 0 : 1;
    }
    if (cmd == "record" && args.size() > 1) {
        const std::string& sub = args[1];
        if (sub == "start") {
            RecordingParameters rp;
            rp.target = RECORDING_TARGET::DEVICE_LOCAL;
            for (size_t i = 2; i < args.size(); i++) {
                if (args[i] == "--location") {
                    if (i + 1 >= args.size() ||
                        !parse_location(args[i + 1], rp.location)) {
                        std::fprintf(stderr,
                            "record start: --location wants LAT,LON[,ALT]\n");
                        return 2;
                    }
                    ++i;
                    rp.has_location = true;
                } else {
                    rp.name = args[i];
                }
            }
            ec = dev.enable_recording(rp);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "record start");
            std::puts("recording");
            return 0;
        }
        if (sub == "stop") {
            ec = dev.disable_recording();
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "record stop");
            std::puts("recording stopped");
            return 0;
        }
        if (sub == "status") {
            const bool named = args.size() > 2;
            RecordingStatus r;
            ec = dev.get_recording_status(r, named ? args[2] : "");
            // With no name the query is "is a session in progress?", so no session
            // (or no recordings at all) is a normal answer, not an error.
            if (ec == ERROR_CODE::RECORDING_NOT_FOUND && !named) {
                std::puts("no active recording");
                return 0;
            }
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "record status");
            print_recording(r);
            return 0;
        }
        if (sub == "list") {
            std::vector<RecordingStatus> rs;
            ec = dev.list_recordings(rs);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "record list");
            if (rs.empty()) std::puts("no recordings (0)");
            else {
                std::printf("%zu recording%s:\n", rs.size(), rs.size() == 1 ? "" : "s");
                for (const auto& r : rs) print_recording(r, /*show_target=*/false);
            }
            // Always show remaining space on the recording store.
            uint64_t free_b = 0, total_b = 0;
            if (dev.get_storage(free_b, total_b) == ERROR_CODE::SUCCESS)
                std::printf("storage: %llu / %llu MiB free\n",
                            (unsigned long long)(free_b  >> 20),
                            (unsigned long long)(total_b >> 20));
            return 0;
        }
        if (sub == "delete" && args.size() > 2) {
            ec = dev.delete_recording(args[2]);
            // Surface the device's own refusal text ("recording is live; stop
            // it first") instead of a bare code.
            if (ec != ERROR_CODE::SUCCESS)
                return fail(ec, "record delete", dev.last_error_message());
            std::printf("deleted '%s'\n", args[2].c_str());
            return 0;
        }
    }
    if (cmd == "download" && args.size() > 1) {
        std::string dest = args.size() > 2 ? args[2] : args[1] + ".mcap";
        ec = dev.download_recording(args[1], dest);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "download");
        std::printf("saved %s\n", dest.c_str());
        return 0;
    }
    if (cmd == "upload" && args.size() > 2) {
        std::string lu = args[2];
        for (char& c : lu) c = (char)std::tolower((unsigned char)c);
        if (lu.rfind("http://", 0) != 0 && lu.rfind("https://", 0) != 0) {
            std::fprintf(stderr, "upload: URL must be an http(s):// address\n");
            return 2;
        }
        ec = dev.upload_recording(args[1], args[2]);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "upload");
        std::printf("upload started: '%s'\n", args[1].c_str());
        return 0;
    }
    if (cmd == "stop-upload" && args.size() > 1) {
        ec = dev.stop_upload(args[1]);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "stop-upload");
        std::printf("upload stopped: '%s'\n", args[1].c_str());
        return 0;
    }
    if (cmd == "check-update") {
        UpdateAvailability a;
        ec = dev.check_update(a);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "check-update", a.service_error);
        const DeviceInformation di = dev.get_device_information();
        if (!a.available) {
            std::printf("up to date (running v%s)\n", di.firmware_version_str.c_str());
        } else {
            std::printf("update available: v%s (running v%s)\n",
                        a.target_version_str.empty() ? std::to_string(a.target_version).c_str()
                                                     : a.target_version_str.c_str(),
                        di.firmware_version_str.c_str());
            std::printf("  from %s\n", a.url.c_str());
        }
        if (!a.notes.empty()) std::printf("  note: %s\n", a.notes.c_str());
        // The service can answer "nothing for you" (404) for a misconfiguration, not just
        // because the device is current. Say which, or a stalled fleet looks up to date.
        if (!a.available && !a.service_error.empty())
            std::printf("  service: %s\n", a.service_error.c_str());
        return 0;
    }
    if (cmd == "update") {
        // Source is explicit: --url downloads that URL (no service call), --file
        // sideloads a local bundle over the wire, neither asks the service. A bare
        // positional is still accepted for compatibility, where a path that exists on
        // disk means sideload and anything else is a URL.
        std::string src = args.size() > 1 && args[1].rfind("--", 0) != 0 ? args[1] : "";
        for (size_t i = 1; i < args.size(); i++) {
            if (args[i] != "--url" && args[i] != "--file") continue;
            // A flag with no value must not fall back to "ask the service": the user
            // named a source, and quietly updating from a different one is worse than
            // refusing. Same for naming two sources.
            if (i + 1 >= args.size()) {
                std::fprintf(stderr, "update: %s needs a value\n", args[i].c_str());
                return 1;
            }
            if (!src.empty()) {
                std::fprintf(stderr, "update: give only one of --url / --file / a path\n");
                return 1;
            }
            const bool want_file = (args[i] == "--file");
            src = args[++i];
            // Downstream still infers sideload-vs-download by stat()ing the string, so a
            // mistyped --file path would be fetched as a URL. Check the flag here instead.
            struct stat sb{};
            const bool is_file = ::stat(src.c_str(), &sb) == 0 && S_ISREG(sb.st_mode);
            if (want_file && !is_file) {
                std::fprintf(stderr, "update: --file '%s' is not a readable file\n", src.c_str());
                return 1;
            }
            // Pushing ~400 MB in 7 KB chunks over BLE takes hours and cannot resume, so
            // the SDK refuses it. Say so here rather than returning a bare error code.
            if (want_file && init.input_type == INPUT_TYPE::STREAM) {
                std::fprintf(stderr,
                    "update: --file needs USB; a local bundle cannot be pushed over BLE\n"
                    "  Connect over USB, or use --url so the device downloads it itself.\n");
                return 1;
            }
            if (!want_file && src.rfind("http://", 0) != 0 && src.rfind("https://", 0) != 0) {
                std::fprintf(stderr,
                    "update: --url '%s' is not an http(s) URL%s\n", src.c_str(),
                    is_file ? " (use --file for a local bundle)" : "");
                return 1;
            }
        }
        UpdatePrinter pr;
        ec = dev.update(src, [&pr](const UpdateStatus& u) { pr.on(u); });
        if (ec == ERROR_CODE::DEVICE_UP_TO_DATE) {
            pr.done();   // close the open line before the summary
            std::puts("already up to date");
            return 0;
        }
        if (ec == ERROR_CODE::WIFI_NOT_CONNECTED) {
            std::fprintf(stderr,
                "update failed: the device is not on WiFi, so it cannot fetch the update\n"
                "  Put it on a network:  ef-cli wifi add <ssid> <psk>\n"
                "  Or update over USB:   ef-cli update --file ./update.eff\n");
            return 1;
        }
        if (ec != ERROR_CODE::SUCCESS) {
            if (pr.have) pr.commit("FAIL", pr.last_msg);
            const std::string& url = src;
            switch (classify_update_failure(ec, pr.last_msg)) {
                case UpdateFailure::BUSY:
                    std::fprintf(stderr,
                        "update failed: a download is already in progress on the device\n"
                        "  Wait for it to finish (re-running attaches to it), or run\n"
                        "  'ef-cli abort-update' to cancel it, then retry.\n");
                    return 1;
                case UpdateFailure::NETWORK: {
                    const std::string retry =
                        url.empty() ? "ef-cli update" : "ef-cli update " + url;
                    std::fprintf(stderr,
                        "update failed: network interruption during download\n"
                        "  The device lost connectivity while fetching the update.\n"
                        "  Re-run '%s' to try again.\n", retry.c_str());
                    return 1;
                }
                case UpdateFailure::GENERIC:
                default:
                    // crashed / rolled back / never came back: lead with the reason.
                    if (!pr.last_msg.empty())
                        std::fprintf(stderr, "error: %s\n", pr.last_msg.c_str());
                    return fail(ec, "update", dev.last_error_message());
            }
        }
        pr.done();
        DeviceInformation di = dev.get_device_information();
        if (!di.firmware_version_str.empty() && di.firmware_version_str != "unknown")
            std::printf("[%-11s] done  running firmware v%s\n", "updated", di.firmware_version_str.c_str());
        else
            std::printf("[%-11s] done  running firmware %u\n", "updated", di.firmware_version);
        return 0;
    }
    if (cmd == "abort-update") {
        ec = dev.abort_update();
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "abort-update");
        std::puts("update aborted");
        return 0;
    }
    if (cmd == "wifi" && args.size() > 1) {
        const std::string& sub = args[1];
        if (sub == "add" && args.size() > 2) {
            // A word count that can't be <ssid> <psk> [country] is almost always
            // an unquoted SSID with spaces shifting the password into the wrong
            // slot; name the fix rather than mis-provisioning the device.
            const char* quote_hint =
                "  (an SSID with spaces must be quoted: "
                "ef-cli wifi add \"My Network\" <password>)\n";
            if (args.size() > 5) {
                std::fprintf(stderr, "wifi add: too many arguments\n%s", quote_hint);
                return 2;
            }
            if (args.size() == 5) {
                const std::string& c = args[4];
                bool cc = c.size() >= 2 && c.size() <= 3;
                for (char ch : c)
                    if (!std::isalnum((unsigned char)ch)) cc = false;
                if (!cc) {
                    std::fprintf(stderr,
                                 "wifi add: '%s' is not a country code (2-3 letters, e.g. US)\n%s",
                                 c.c_str(), quote_hint);
                    return 2;
                }
            }
            std::string psk;
            if (args.size() == 3) {   // ssid only: ask, echoing '*' per keystroke
                if (!prompt_secret("WiFi password: ", psk)) {
                    std::fprintf(stderr, "wifi add: cancelled\n");
                    return 1;
                }
            } else {
                psk = args[3];
            }
            // WPA-PSK passphrases are 8-63 chars; the device would accept the
            // store write and then fail the association with a bare timeout.
            if (psk.size() < 8 || psk.size() > 63) {
                std::fprintf(stderr, "wifi add: password must be 8-63 characters\n");
                return 2;
            }
            ec = dev.wifi_add(args[2], psk, args.size() > 4 ? args[4] : "");
        }
        else if (sub == "remove" && args.size() > 2) ec = dev.wifi_remove(args[2]);
        else if (sub == "select" && args.size() > 2) ec = dev.wifi_select(args[2]);
        else if (sub == "status") {
            const WirelessConfiguration& w = dev.get_device_information().wireless;
            if (w.wifi_state == "connecting") {
                if (!w.wifi_ssid.empty())
                    std::printf("connecting to \"%s\"...\n", w.wifi_ssid.c_str());
                else
                    std::puts("connecting...");
            } else if (w.wifi_state == "auth_failed") {
                // Device rejected the credentials (wrong password / bad auth);
                // name the network so the user knows which one to re-add.
                if (!w.wifi_ssid.empty())
                    std::printf("authentication failed for \"%s\"; check the password\n",
                                w.wifi_ssid.c_str());
                else
                    std::puts("authentication failed; check the password");
            } else if (w.wifi_state == "connected" || w.wifi_connected) {
                // Append only the detail the device reported; older firmware
                // leaves security/freq/link_speed/rssi at ""/0, which drop out.
                std::string x;
                if (!w.wifi_security.empty()) x += ", " + w.wifi_security;
                if (w.wifi_freq_mhz > 0)
                    x += w.wifi_freq_mhz < 3000 ? ", 2.4 GHz" : ", 5 GHz";
                if (w.wifi_link_speed > 0)
                    x += ", " + std::to_string(w.wifi_link_speed) + " Mbps";
                if (w.wifi_rssi != 0)
                    x += ", signal " + std::to_string(w.wifi_rssi) + " dBm";
                std::printf("connected to \"%s\" (%s%s)\n",
                            w.wifi_ssid.c_str(), w.wifi_ip_address.c_str(), x.c_str());
            } else {
                std::puts("not connected");
            }
            return 0;
        } else if (sub == "list") {
            const WirelessConfiguration& w = dev.get_device_information().wireless;
            if (w.saved_networks.empty()) {
                std::puts("no saved networks");
            } else {
                const std::string& cur = w.wifi_connected ? w.wifi_ssid : std::string();
                for (const auto& n : w.saved_networks)
                    std::printf("%s%s\n", n.c_str(), n == cur ? "  (connected)" : "");
            }
            return 0;
        } else if (sub == "scan") {
            std::vector<WifiNetwork> nets;
            ec = dev.scan_wifi_networks(nets);
            // BUSY is the expected answer while the radio is in use (recording,
            // livestream, or an update), not an error; tell the user to retry.
            if (ec == ERROR_CODE::DEVICE_BUSY) {
                std::puts("device busy (recording/livestream/update); try again after it stops");
                return 1;
            }
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "wifi scan");
            if (nets.empty()) {
                std::puts("no networks found");
            } else {
                // nets is sorted strongest-first by the SDK; show the top 10.
                size_t show = std::min<size_t>(nets.size(), 10);
                for (size_t i = 0; i < show; i++)
                    std::printf("%-32s  %4d dBm  %s\n", nets[i].ssid.c_str(),
                                nets[i].rssi, nets[i].secured ? "secured" : "open");
                if (nets.size() > show)
                    std::printf("(%zu more; showing strongest %zu)\n",
                                nets.size() - show, show);
            }
            return 0;
        } else { usage(); return 2; }
        // A remove of a never-saved ssid comes back INVALID_FUNCTION_CALL (device
        // NOT_FOUND): say so plainly rather than the misleading "forgotten" below.
        if (sub == "remove" && ec == ERROR_CODE::INVALID_FUNCTION_CALL) {
            std::fprintf(stderr, "wifi remove: '%s' is not a saved network\n", args[2].c_str());
            return 1;
        }
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, std::string("wifi " + sub).c_str());
        if (sub == "add")
            std::printf("wifi add accepted, connecting to '%s' (poll `ef-cli wifi status`)\n",
                        args[2].c_str());
        else if (sub == "remove")
            std::printf("wifi remove accepted, '%s' forgotten, disconnected\n", args[2].c_str());
        else if (sub == "select")
            std::printf("wifi select accepted, '%s' (poll `ef-cli wifi status`)\n", args[2].c_str());
        return 0;
    }
    if (cmd == "set-password" && args.size() > 1) {
        // USB: set-password <new> (old not required, physical access resets).
        // BLE: set-password <old> <new>.
        std::string old_pw = args.size() > 2 ? args[1] : "";
        std::string new_pw = args.size() > 2 ? args[2] : args[1];
        ec = dev.set_ble_password(old_pw, new_pw);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "set-password");
        std::puts("control password updated (BLE and USB)");
        return 0;
    }
    if (cmd == "lock" && args.size() > 1) {
        bool want = (args[1] == "on" || args[1] == "true" || args[1] == "1");
        if (!want && args[1] != "off" && args[1] != "false" && args[1] != "0") {
            std::fprintf(stderr, "lock: expected 'on' or 'off'\n");
            return 2;
        }
        // --session leaves the stored policy alone, so losing power closes the
        // device again with nothing to undo.
        bool session = std::find(args.begin(), args.end(), "--session") != args.end();
        ec = dev.set_usb_lock(want, session);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "lock", dev.last_error_message());
        if (session)
            std::printf("USB %s\n", want
                ? "session unlock ended: password required again"
                : "open for this session: no password needed until you re-lock it "
                  "or the device loses power (still LOCKED after a reboot)");
        else
            std::printf("USB %s\n", want
                ? "LOCKED: every command except info/state/storage/factory-reset now needs --password"
                : "unlocked: full privileges over USB");
        return 0;
    }
    if (cmd == "encryption" && args.size() > 1 && args[1] == "create") {
        EncryptionKey k;
        ec = dev.create_encryption_key(k);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "encryption create", dev.last_error_message());
        std::fprintf(stderr,
            "This is the ONLY time this key is shown. Save it now.\n"
            "Without it, recordings made under it cannot be decrypted by anyone,\n"
            "including us. A factory reset destroys the device's copy.\n\n");
        std::printf("key_id %s\n", k.key_id.c_str());
        print_key_hex(k.key);
        wipe(k.key);
        return 0;
    }
    if (cmd == "encryption" && args.size() > 1 && args[1] == "delete") {
        // Two steps on purpose. The first shows the key and destroys nothing, so
        // an operator who still needs to read old recordings can save it before
        // the key is gone; only --confirm with the matching id destroys, and the
        // device enforces that match too.
        std::string confirm_id;
        for (size_t a = 2; a + 1 < args.size(); a++)
            if (args[a] == "--confirm") confirm_id = args[a + 1];

        if (confirm_id.empty()) {
            EncryptionKey k;
            ec = dev.get_encryption_key(k);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "encryption delete", dev.last_error_message());
            if (!k.present) { std::puts("no encryption key to delete"); return 0; }
            std::fprintf(stderr,
                "Key %s is about to be destroyed. Save it FIRST if you still need to\n"
                "read recordings made under it, including ones already uploaded, which\n"
                "become permanently undecryptable.\n\n", k.key_id.c_str());
            print_key_hex(k.key);
            wipe(k.key);
            std::fprintf(stderr,
                "\nNothing has been destroyed yet. To go ahead:\n"
                "  ef-cli encryption delete --confirm %s\n", k.key_id.c_str());
            return 0;
        }

        bool assume_yes = std::find(args.begin(), args.end(), "--yes") != args.end();
        if (!assume_yes) {
            // No tty REFUSES rather than falling through: ssh box 'ef-cli ...', a
            // cron job, or a redirected stdin is exactly where an unprompted
            // destroy goes unnoticed. --yes is the only way to mean it.
            if (!isatty(fileno(stdin))) {
                std::fprintf(stderr, "encryption delete: refusing to destroy a key "
                                     "without a terminal to confirm on; pass --yes "
                                     "to mean it\n");
                return 1;
            }
            std::fprintf(stderr, "Destroy key %s? Type 'destroy' to confirm: ",
                         confirm_id.c_str());
            char line[32] = {0};
            if (!std::fgets(line, sizeof line, stdin) ||
                std::strncmp(line, "destroy", 7) != 0) {
                std::fprintf(stderr, "aborted\n");
                return 1;
            }
        }
        EncryptionKey gone;
        ec = dev.delete_encryption_key(confirm_id, gone);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "encryption delete", dev.last_error_message());
        std::fprintf(stderr, "key %s destroyed. Encryption is now off; "
                             "'encryption create' makes a new one.\n", gone.key_id.c_str());
        wipe(gone.key);
        return 0;
    }
    if (cmd == "encryption" && args.size() > 1) {
        bool on = (args[1] == "on" || args[1] == "true" || args[1] == "1");
        if (!on && args[1] != "off" && args[1] != "false" && args[1] != "0") {
            std::fprintf(stderr, "encryption: expected 'on', 'off', 'create' or 'delete'\n");
            return 2;
        }
        ec = dev.set_encryption(on);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "encryption", dev.last_error_message());
        std::printf("encryption %s for new recordings\n", on ? "ON" : "OFF");
        return 0;
    }
    if (cmd == "key" && args.size() > 1 && args[1] == "show") {
        std::string out_path;
        for (size_t a = 2; a + 1 < args.size(); a++)
            if (args[a] == "--out") out_path = args[a + 1];

        EncryptionKey k;
        ec = dev.get_encryption_key(k);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "key show", dev.last_error_message());
        if (!k.present) { std::puts("no encryption key"); return 0; }
        if (out_path.empty()) { print_key_hex(k.key); wipe(k.key); return 0; }
        int rc = write_key_file(out_path, k.key);
        wipe(k.key);
        return rc;
    }
    if (cmd == "factory-reset") {
        // Ungated on the device by design (the escape when the password is lost),
        // so the CLI is the only place a typo can be caught. Confirm on a TTY;
        // --yes keeps it scriptable.
        bool assume_yes = std::find(args.begin(), args.end(), "--yes") != args.end();
        if (!assume_yes) {
            if (!isatty(fileno(stdin))) {
                std::fprintf(stderr, "factory-reset: refusing to destroy the encryption "
                                     "key without a terminal to confirm on; pass --yes "
                                     "to mean it\n");
                return 1;
            }
            std::fprintf(stderr,
                "This erases recordings, wifi credentials, calibration, capture config\n"
                "and the control password.\n"
                "It also DESTROYS the encryption key. Every recording ever made under\n"
                "it becomes permanently undecryptable, including copies already\n"
                "uploaded elsewhere. Save the key first ('ef-cli key show --out <file>')\n"
                "if you will ever need to read those recordings.\n"
                "Type 'reset' to confirm: ");
            char line[32] = {0};
            if (!std::fgets(line, sizeof line, stdin) ||
                std::strncmp(line, "reset", 5) != 0) {
                std::fprintf(stderr, "aborted\n");
                return 1;
            }
        }
        ec = dev.factory_reset();
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "factory-reset");
        std::puts("factory settings restored "
                  "(password default, USB unlocked, encryption off; "
                  "encryption key DESTROYED)");
        return 0;
    }
    if (cmd == "sync-time") {
        ec = dev.sync_time();
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "sync-time");
        std::puts("time synced (device clock set from host)");
        return 0;
    }
    if (cmd == "time") {
        Timestamp t;
        ec = dev.get_device_time(t);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "time");
        time_t secs = (time_t)t.seconds();
        struct tm tmv;
        char buf[64] = "";
        if (gmtime_r(&secs, &tmv)) strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S UTC", &tmv);
        std::printf("device time      : %s  (epoch %llu ms)\n",
                    buf, (unsigned long long)t.milliseconds());
        return 0;
    }
    if (cmd == "location") {
        if (args.size() > 1 && args[1] == "set") {
            if (args.size() < 4) {
                std::fprintf(stderr, "location set: need <lat> <lon> [alt]\n");
                return 2;
            }
            double lat = std::strtod(args[2].c_str(), nullptr);
            double lon = std::strtod(args[3].c_str(), nullptr);
            double alt = args.size() > 4 ? std::strtod(args[4].c_str(), nullptr) : 0.0;
            ec = dev.set_location(lat, lon, alt);
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "location set");
            std::printf("location set: %.6f, %.6f  (alt %.1f m), persisted\n",
                        lat, lon, alt);
            return 0;
        }
        Location loc;
        ec = dev.get_location(loc);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "location");
        std::printf("location         : %.6f, %.6f  (alt %.1f m)\n",
                    loc.latitude, loc.longitude, loc.altitude);
        return 0;
    }
    if (cmd == "reboot") {
        ec = dev.reboot();
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "reboot");
        std::puts("reboot initiated, device is going down (reconnect in ~30-60s)");
        return 0;
    }

    usage();
    return 2;
}
