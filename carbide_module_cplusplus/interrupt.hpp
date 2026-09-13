// interrupt.hpp — SIGINT handling so Ctrl+C during a watch loop returns to
// the menu (matching Python's `except KeyboardInterrupt`) instead of
// killing the process, which is C++'s default SIGINT behavior.
#pragma once
#include <atomic>
#include <csignal>

namespace carbide {

inline std::atomic<bool> g_interrupted{false};

inline void sigint_handler(int) { g_interrupted = true; }

inline void install_sigint_handler() {
    std::signal(SIGINT, sigint_handler);
}

// call after any loop that checked g_interrupted, so the next watch works too
inline void clear_interrupt() { g_interrupted = false; }

} // namespace carbide
