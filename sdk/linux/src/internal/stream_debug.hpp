////////////////////////////////////////////////////////////////////////////////
//
// File:      internal/stream_debug.hpp
// Purpose:   Opt-in stream/frame diagnostics helpers (internal).
// Author:    Calvin Nguyen, Gianluca Bencomo
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
//
// Diagnostics for isolating the stream-startup pause (a few frames, then a
// ~1.9 s stall skipping ~one GOP of frame IDs, then normal 30 fps). NOT a
// logging framework: two inline helpers plus a shared line prefix, gated behind
// the existing InitParameters::verbose knob so production stays byte-for-byte
// unchanged. Every line goes to stderr as `[ef.diag] ...`, one std::fprintf per
// line (libc per-FILE locking keeps lines from interleaving across the
// transport and consumer threads, matching the existing `[ef]` logging).
//
////////////////////////////////////////////////////////////////////////////////

#ifndef EF_STREAM_DEBUG_HPP
#define EF_STREAM_DEBUG_HPP

#include <chrono>
#include <cstdint>

namespace ef {
namespace internal {

// The verbose level at which the high-volume per-frame / per-packet stream
// diagnostics turn on. verbose 0 (production) and 1 (connection trace) stay
// quiet; 2 opts into everything in this header.
inline constexpr int kDiagLevel = 2;

// Single opt-in switch. Reuses InitParameters::verbose (0 quiet, 1 trace, >=2
// stream diagnostics) rather than adding a second flag or a new framework.
inline bool diag_on(int verbose) { return verbose >= kDiagLevel; }

// Monotonic host clock in nanoseconds (steady_clock, never wall time), so the
// timestamps are safe to subtract across a session even if the system clock is
// stepped by NTP. Matches udp_stream_reader.cpp's mono_ns().
inline uint64_t diag_mono_ns() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Milliseconds between two monotonic samples, as a double for readable logs.
inline double diag_ms_since(uint64_t then_ns, uint64_t now_ns) {
    return now_ns >= then_ns ? (double)(now_ns - then_ns) / 1e6 : 0.0;
}

}  // namespace internal
}  // namespace ef

#endif  // EF_STREAM_DEBUG_HPP
