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

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

#include <ef/Device.hpp>

using namespace ef;

namespace {

void usage() {
    std::puts(
        "ef: Efference M1 control tool\n"
        "\n"
        "  ef [--ble <MAC>] [--device <id>] [--password <pw>] [--udp <host[:port]>]\n"
        "     [--verbose] <command> [args]\n"
        "\n"
        "flags:\n"
        "  --ble <MAC>                       connect over Bluetooth instead of USB\n"
        "  --device <id>                     pick one of several USB devices\n"
        "  --password <pw>                   BLE control password (default 123456)\n"
        "  --udp <host[:port]>               with --ble: device streams video+IMU to\n"
        "                                    this host over WiFi/UDP (default port 5005)\n"
        "\n"
        "commands:\n"
        "  list [--scan-ble]                 discover devices (USB, and BLE with --scan-ble)\n"
        "  info                              device information snapshot\n"
        "  config                            list enabled capture modes + codecs\n"
        "  config set <W> <H> <fps> <codec>  set capture config (idle only; codec:\n"
        "                                    raw|h264|h264hq|h265|h265hq)\n"
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
        "  check-update                      is newer firmware available?\n"
        "  update [url|file.eff]             download + apply firmware (local .eff sideloads)\n"
        "  abort-update                      cancel an update in progress\n"
        "  wifi add <ssid> <psk> [country]   provision a WiFi network (\"US\" unlocks 5 GHz)\n"
        "  wifi remove <ssid> | select <ssid>\n"
        "  wifi list                         saved networks (marks the connected one)\n"
        "  wifi status                       current association\n"
        "  set-password <new>                rekey the BLE password (over USB)\n"
        "  set-password <old> <new>          rekey the BLE password (over BLE)\n"
        "  sync-time                         set the device clock from the host\n"
        "  time                              read the device wall clock\n"
        "  location                          read the device's current location\n"
        "  location set <lat> <lon> [alt]    persist the device location (all recordings)\n"
        "  reboot                            reboot the device\n");
}

int fail(ERROR_CODE ec, const char* what) {
    std::fprintf(stderr, "%s failed: %s\n", what, to_string(ec));
    return 1;
}

void print_info(const DeviceInformation& i) {
    std::printf("serial           : %s\n", i.serial.c_str());
    // serial_number is only the numeric convenience form of `serial`; it is 0
    // for any non-decimal serial (e.g. "M1BENCH002"). Show it only when it
    // actually carries information, so it isn't mistaken for a second identity.
    if (i.serial_number != 0)
        std::printf("serial_number    : %u\n", i.serial_number);
    std::printf("model            : %s\n", to_string(i.model));
    std::printf("firmware_version : %u\n", i.firmware_version);
    std::printf("input_type       : %s\n", to_string(i.input_type));
    const CameraConfiguration& c = i.camera_configuration;
    std::printf("camera           : %dx%d @ %d fps, %s\n",
                c.resolution.width, c.resolution.height, c.fps,
                to_string(c.compression));
    std::printf("calibration      : fx=%.2f fy=%.2f cx=%.2f cy=%.2f xi=%.4f alpha=%.4f (%s)\n",
                c.calibration.fx, c.calibration.fy, c.calibration.cx,
                c.calibration.cy, c.calibration.xi, c.calibration.alpha,
                to_string(c.calibration.model));
    const SensorsConfiguration& s = i.sensors_configuration;
    std::printf("accelerometer    : %s (noise %.6f)\n",
                to_string(s.accelerometer.state), s.accelerometer.noise_density);
    std::printf("gyroscope        : %s (noise %.6f)\n",
                to_string(s.gyroscope.state), s.gyroscope.noise_density);
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

// show_target: print the DEVICE_LOCAL/HOST_FILE column. Meaningful for
// `record status` (a live host recording reads HOST_FILE); pointless for
// `record list`, whose entries are always DEVICE_LOCAL.
void print_recording(const RecordingStatus& r, bool show_target = true) {
    std::printf("%-24s ", r.name.c_str());
    if (show_target) std::printf("%-12s ", to_string(r.target));
    std::printf("%s  %.1f MB, %llu frames, %llu ms",
                r.recording ? "RECORDING" : "complete",
                r.bytes / (1024.0 * 1024.0), (unsigned long long)r.frames,
                (unsigned long long)r.duration_ms);
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
    if (ec != ERROR_CODE::SUCCESS) return fail(ec, "open");

    if (cmd == "info") {
        print_info(dev.get_device_information());
        return 0;
    }
    if (cmd == "config" && args.size() > 1 && args[1] == "set") {
        if (args.size() < 6) {
            std::fprintf(stderr,
                "usage: ef config set <width> <height> <fps> <codec>\n"
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
            if (ec != ERROR_CODE::SUCCESS) return fail(ec, "record delete");
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
        bool available = false;
        ec = dev.check_update(available);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "check-update");
        std::puts(available ? "update available" : "up to date");
        return 0;
    }
    if (cmd == "update") {
        ec = dev.update(args.size() > 1 ? args[1] : "",
                        [](const UpdateStatus& u) {
                            std::printf("\r%-14s %3d%%  %s",
                                        u.active ? to_string(u.state) : "…",
                                        u.progress, u.message.c_str());
                            std::fflush(stdout);
                        });
        std::printf("\n");
        if (ec == ERROR_CODE::DEVICE_UP_TO_DATE) { std::puts("already up to date"); return 0; }
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "update");
        std::printf("updated, running firmware %u\n",
                    dev.get_device_information().firmware_version);
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
        if (sub == "add" && args.size() > 3)
            ec = dev.wifi_add(args[2], args[3], args.size() > 4 ? args[4] : "");
        else if (sub == "remove" && args.size() > 2) ec = dev.wifi_remove(args[2]);
        else if (sub == "select" && args.size() > 2) ec = dev.wifi_select(args[2]);
        else if (sub == "status") {
            const WirelessConfiguration& w = dev.get_device_information().wireless;
            if (w.wifi_state == "connecting") {
                if (!w.wifi_ssid.empty())
                    std::printf("connecting to \"%s\"...\n", w.wifi_ssid.c_str());
                else
                    std::puts("connecting...");
            } else if (w.wifi_state == "connected" || w.wifi_connected) {
                // Append only the detail the device actually reported, older
                // firmware leaves security/freq/link_speed/rssi at ""/0, and those
                // just drop out of the line (no wire-format dependency).
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
        } else { usage(); return 2; }
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, std::string("wifi " + sub).c_str());
        if (sub == "add")
            std::printf("wifi add accepted, connecting to '%s' (poll `ef wifi status`)\n",
                        args[2].c_str());
        else if (sub == "remove")
            std::printf("wifi remove accepted, '%s' forgotten, disconnected\n", args[2].c_str());
        else if (sub == "select")
            std::printf("wifi select accepted, '%s' (poll `ef wifi status`)\n", args[2].c_str());
        return 0;
    }
    if (cmd == "set-password" && args.size() > 1) {
        // USB: set-password <new> (old not required, physical access resets).
        // BLE: set-password <old> <new>.
        std::string old_pw = args.size() > 2 ? args[1] : "";
        std::string new_pw = args.size() > 2 ? args[2] : args[1];
        ec = dev.set_ble_password(old_pw, new_pw);
        if (ec != ERROR_CODE::SUCCESS) return fail(ec, "set-password");
        std::puts("BLE password updated");
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
