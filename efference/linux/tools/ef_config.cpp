// ef-config — list the device's advertised capabilities, or change the session
// configuration (resolution / framerate / codec / container / recording knobs).
//
//   ef-config [--list] [device_index]
//       Print the advertised caps (every usable mode) + the current selection.
//
//   ef-config --set KEY=VALUE [KEY=VALUE ...] [device_index]
//       Apply a (partial) configuration. Recognised keys:
//         width, height, fps, segment_seconds   (integers)
//         codec        (h264 | h265)
//         container    (mcap)
//         capture_mode (usb_sdk | wifi_collect | livestream | uvc)
//       Configuration is IDLE-only (ADR-0009): stop any recording first. The
//       camera tuple is validated on the device against the advertised modes; an
//       unusable/unsupported combination is rejected and nothing changes.
//
//   Examples:
//       ef-config --list
//       ef-config --set width=1280 height=720 fps=30 codec=h264
//       ef-config --set fps=15            # change only the framerate
#include <ef/device.hpp>

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

using namespace ef;

namespace {

bool is_uint(const std::string& s) {
    if (s.empty()) return false;
    for (char ch : s) if (!std::isdigit((unsigned char)ch)) return false;
    return true;
}

void usage(const char* argv0) {
    std::cerr <<
        "usage:\n"
        "  " << argv0 << " [--list] [device_index]\n"
        "  " << argv0 << " --set KEY=VALUE [KEY=VALUE ...] [device_index]\n"
        "\n"
        "keys: width height fps segment_seconds (int), codec (h264|h265),\n"
        "      container (mcap), capture_mode (usb_sdk|wifi_collect|livestream|uvc)\n";
}

int do_list(Device& dev) {
    DeviceInformation i = dev.get_device_information();
    const auto& cap = i.caps;
    const auto& c   = i.camera;
    const auto& e   = i.encoding;

    std::cout << "\n=== Device capabilities (advertised) ===\n";

    std::cout << "  codecs        : ";
    for (auto& s : cap.codecs)        std::cout << s << " ";
    std::cout << "\n  pixel_formats : ";
    for (auto& s : cap.pixel_formats) std::cout << s << " ";
    std::cout << "\n  containers    : ";
    for (auto& s : cap.containers)    std::cout << s << " ";
    std::cout << "\n";

    std::cout << "\n  modes (selectable unless noted; codecs = the list above):\n";
    std::cout << "    " << "geometry" << "\t" << "fps" << "\tbinning\tstatus\n";
    std::cout << "    --------------------------------------------\n";
    for (auto& m : cap.modes)
        std::cout << "    " << m.width << "x" << m.height
                  << "\t" << m.fps
                  << "\t" << (m.binning.empty() ? "-" : m.binning)
                  << "\t" << (m.usable ? "usable" : "NOT usable") << "\n";

    std::cout << "\n=== Current selection ===\n";
    std::cout << "  resolution/fps : " << c.width << "x" << c.height << "@" << c.fps << "\n";
    std::cout << "  codec          : " << to_string(e.codec) << "\n";
    std::cout << "  container      : " << i.packer.format << "\n";
    std::cout << "\n  (orchestrator_reachable=" << (i.orchestrator_reachable ? "true" : "false")
              << ", capture_reachable=" << (i.capture_reachable ? "true" : "false") << ")\n";
    return 0;
}

// Parse KEY=VALUE tokens into a Configuration. Returns false on a bad token.
bool parse_set(int argc, char** argv, int start, Configuration& cfg) {
    bool any = false;
    for (int k = start; k < argc; k++) {
        std::string tok = argv[k];
        auto eq = tok.find('=');
        if (eq == std::string::npos) {
            std::cerr << "bad token (expected KEY=VALUE): " << tok << "\n";
            return false;
        }
        std::string key = tok.substr(0, eq);
        std::string val = tok.substr(eq + 1);
        if      (key == "width")           cfg.width           = std::atoi(val.c_str());
        else if (key == "height")          cfg.height          = std::atoi(val.c_str());
        else if (key == "fps")             cfg.fps             = std::atoi(val.c_str());
        else if (key == "segment_seconds") cfg.segment_seconds = std::atoi(val.c_str());
        else if (key == "codec")           cfg.codec           = val;
        else if (key == "container")       cfg.container       = val;
        else if (key == "capture_mode")    cfg.capture_mode    = val;
        else { std::cerr << "unknown key: " << key << "\n"; return false; }
        any = true;
    }
    if (!any) std::cerr << "--set needs at least one KEY=VALUE\n";
    return any;
}

}  // namespace

int main(int argc, char** argv) {
    bool          set_mode = false;
    int           set_start = 0;
    int           device_id = 0;

    // Scan for --set / --list and a trailing numeric device index. In --set mode
    // every non-flag token after --set is a KEY=VALUE (the device index, if any,
    // is the last bare integer, but KEY=VALUE never looks like one).
    for (int a = 1; a < argc; a++) {
        std::string arg = argv[a];
        if (arg == "--set") { set_mode = true; set_start = a + 1; }
        else if (arg == "--list") { /* default action */ }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else if (!set_mode && is_uint(arg)) {
            device_id = std::atoi(arg.c_str());  // bare index in list mode
        }
    }

    Configuration cfg;
    if (set_mode) {
        // Allow a trailing bare integer device index after the KEY=VALUEs.
        int end = argc;
        if (end > set_start && is_uint(argv[end - 1])) {
            device_id = std::atoi(argv[end - 1]);
            end--;
        }
        if (!parse_set(end, argv, set_start, cfg)) { usage(argv[0]); return 2; }
    }

    Device dev;
    InitParameters init;
    init.verbose   = 1;
    init.device_id = device_id;

    ERROR_CODE st = dev.open(init);
    if (st != ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open device: " << to_string(st) << "\n";
        return 1;
    }

    int rc = 0;
    try {
        if (set_mode) {
            ConfigureResult r = dev.configure(cfg);
            if (r.applied) {
                std::cout << "configured: applied (config_version=" << r.config_version << ")\n";
            } else {
                std::cerr << "configure REJECTED: " << r.reason << "\n";
                if (r.reason == "not_idle")
                    std::cerr << "  (the device is recording — stop the session first)\n";
                else if (r.reason.rfind("camera:", 0) == 0)
                    std::cerr << "  (run `" << argv[0] << " --list` to see usable modes)\n";
                rc = 1;
            }
        } else {
            rc = do_list(dev);
        }
    } catch (const std::exception& ex) {
        std::cerr << (set_mode ? "configure failed: " : "get_device_information failed: ")
                  << ex.what() << "\n";
        rc = 1;
    }

    dev.close();
    return rc;
}
