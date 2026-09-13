// state.hpp — mirrors carbide_modules/state.py. training_active is a real
// std::atomic<bool> (Python's plain bool worked "by accident" under the
// GIL; C++ needs an actual atomic for that same read-without-a-lock pattern
// to be well-defined). step/loss_history/current_loss stay mutex-guarded at
// exactly the call sites the Python version guards them at.
#pragma once
#include <vector>
#include <mutex>
#include <atomic>
#include <thread>
#include <chrono>
#include <utility>

namespace carbide {

struct TrainingState {
    long long step = 0;
    std::vector<std::pair<long long, double>> loss_history;
    double current_loss = 0.0;
    std::chrono::steady_clock::time_point start_time;
    std::atomic<bool> training_active{false};
    long long training_target_steps = 0;
    std::mutex training_lock;
    std::thread training_thread;
    bool thread_started = false;

    TrainingState() { reset(); }

    void reset() {
        step = 0;
        loss_history.clear();
        current_loss = 0.0;
        start_time = std::chrono::steady_clock::now();
        training_active = false;
        training_target_steps = 0;
        // training_lock and training_thread are left as-is, same as the
        // Python version reusing the existing Lock object on reset.
    }

    double get_elapsed() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    }

    double get_speed() const {
        double elapsed = get_elapsed();
        if (elapsed > 0 && step > 0) return (double)step / elapsed;
        return 0.0;
    }
};

inline TrainingState train_state;

} // namespace carbide
