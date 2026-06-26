// ef-stream — capture the device's MCAP stream to a .mcap file in one call.
// The Checkpoint-4 deliverable for the data plane (mcap-streaming.md M3):
// open the device, stream ep3 to disk for N seconds (or until Ctrl-C / a
// terminal end), then report how it ended.
//
//   ef-stream <out.mcap> [seconds] [device_index]
//
// seconds <= 0 streams until Ctrl-C or a terminal end (drop / producer restart).
// The written .mcap is complete (footer captured) on a clean stop, and a valid
// truncated prefix on a terminal end. A live viewer should read <out.mcap>
// independently (e.g. tail it into Foxglove) — the reader here persists at line
// rate and is never coupled to a viewer.
#include <ef/device.hpp>

#include <atomic>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace ef;

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop.store(true); }

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <out.mcap> [seconds] [device_index]\n";
        return 2;
    }
    const std::string out = argv[1];
    const double seconds   = (argc > 2) ? std::atof(argv[2]) : 0.0;
    const int    dev_index = (argc > 3) ? std::atoi(argv[3]) : 0;

    std::signal(SIGINT, on_sigint);

    Device device;
    InitParameters init;
    init.verbose   = 1;
    init.device_id = dev_index;

    ERROR_CODE st = device.open(init);
    if (st != ERROR_CODE::SUCCESS) {
        std::cerr << "Failed to open device: " << to_string(st) << "\n";
        return 1;
    }

    std::cerr << "streaming -> " << out << " ("
              << (seconds > 0 ? (std::to_string(seconds) + "s") : std::string("until Ctrl-C / terminal end"))
              << "); Ctrl-C to stop\n";

    // Notified once at the end — before any resume decision (the SDK never
    // auto-restarts; that is the user's call).
    auto on_end = [](const StreamResult& r) {
        std::cerr << "[on_end] " << to_string(r.end) << " (device last_end=\"" << r.detail << "\")\n";
    };

    StreamResult r = (seconds > 0)
        ? device.stream_mcap(out, seconds, on_end)
        : device.stream_mcap(out, [] { return g_stop.load(); }, on_end);

    std::cout << "\n=== stream ended ===\n";
    std::cout << "  end      : " << to_string(r.end) << "\n";
    std::cout << "  bytes    : " << r.bytes << "\n";
    std::cout << "  dropped  : " << r.dropped << "\n";
    std::cout << "  restarts : " << r.restarts << "\n";
    std::cout << "  detail   : " << r.detail << "\n";
    std::cout << "  output   : " << out << "\n";

    device.close();

    // Clean stop => a complete, Foxglove-openable .mcap; terminal => prefix.
    return (r.end == STREAM_END::STOPPED) ? 0 : 1;
}
