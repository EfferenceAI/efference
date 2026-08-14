////////////////////////////////////////////////////////////////////////////////
//
// File:      tools/progress.hpp
// Purpose:   In-place progress line for `ef-cli download`.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef EF_TOOLS_PROGRESS_HPP
#define EF_TOOLS_PROGRESS_HPP

#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>

#include <ef/Core.hpp>

namespace ef {

// Renders `ef-cli download` as one line that redraws in place. TTY only: on a
// redirected stream nothing is printed, so a scripted caller still sees exactly the
// one "saved <path>" line it parses today. The bar is erased before that line.
struct DownloadPrinter {
    // A small recording lands in well under a second, where a bar reports nothing
    // and only flickers. Nothing is drawn until the transfer proves it is slow
    // enough to be worth watching.
    // constexpr, not const: these are odr-used by std::chrono::milliseconds below.
    static constexpr long kQuietMs  = 300;
    static constexpr long kRedrawMs = 100;
    // The quiet window doubles as the first redraw gate, which only works while it
    // is the longer of the two.
    static_assert(kQuietMs >= kRedrawMs, "the quiet window must outlast the redraw interval");

    bool tty     = isatty(fileno(stdout));
    bool drawn   = false;
    bool running = false;        // the clock starts when bytes do, not at construction
    std::chrono::steady_clock::time_point start, last;

    // Mebibytes, matching what `record list` divides by and labels "MB", so the same
    // recording reads the same in both places.
    static double mib(uint64_t b) { return (double)b / (1024.0 * 1024.0); }

    void on(const DownloadProgress& p) {
        if (!tty) return;
        auto now = std::chrono::steady_clock::now();
        // The first call lands before any bytes are written, so it starts the clock
        // rather than diluting the rate with the round trip that preceded it.
        if (!running) { start = last = now; running = true; return; }
        if (now - start < std::chrono::milliseconds(kQuietMs))  return;
        if (now - last  < std::chrono::milliseconds(kRedrawMs)) return;
        last  = now;
        drawn = true;

        double secs = std::chrono::duration<double>(now - start).count();
        double rate = mib(p.received - p.resumed_from) / secs;
        // Below 1 MB/s, report KB/s: "%.1f MB/s" would render as "0.0 MB/s".
        char rbuf[24];
        if (rate < 1.0) std::snprintf(rbuf, sizeof rbuf, "%.0f KB/s", rate * 1024.0);
        else            std::snprintf(rbuf, sizeof rbuf, "%.1f MB/s", rate);

        if (p.total) {
            int pct = p.received >= p.total ? 100 : (int)(p.received * 100 / p.total);
            std::printf("\r[%-11s] %3d%%  %.1f/%.1f MB  %s\033[K",
                        "download", pct, mib(p.received), mib(p.total), rbuf);
        } else {
            // Firmware that never fills the size field: report what has arrived
            // rather than a percentage that would be invented.
            std::printf("\r[%-11s] ....  %.1f MB  %s\033[K",
                        "download", mib(p.received), rbuf);
        }
        std::fflush(stdout);
    }

    // Erase the bar so the result line that follows starts on a clean row.
    void done() {
        if (drawn) { std::printf("\r\033[K"); std::fflush(stdout); }
        drawn = false;
    }
};

}  // namespace ef

#endif  // EF_TOOLS_PROGRESS_HPP
