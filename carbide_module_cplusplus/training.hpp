// training.hpp — mirrors carbide_modules/training.py. `model`/`opt` live
// only here; other modules must go through carbide::model / carbide::opt
// (this namespace's globals) rather than capturing them by value, so a
// reinit or checkpoint load is visible everywhere.
#pragma once
#include "model.hpp"
#include "optim.hpp"
#include "state.hpp"
#include "dataset.hpp"
#include "checkpoint.hpp"
#include "monitor.hpp"
#include <memory>
#include <random>
#include <cstdio>
#include <algorithm>

namespace carbide {

inline std::unique_ptr<Carbide> model;
inline std::unique_ptr<AdamW> opt;

inline void init_model() {
    std::mt19937 r(std::random_device{}());
    model = std::make_unique<Carbide>(config.d_model, config.n_layers, config.d_state, r);
    opt = std::make_unique<AdamW>(model->parameters(), config.learning_rate);
    std::cout << "\xE2\x9C\x93 Model initialized: " << model->num_params() << " parameters\n";
}

inline double train_step() {
    if (!model) init_model();
    auto batch = get_batch();
    auto logits = model->forward(batch.xs, batch.B, batch.T, Mode::Full);
    auto loss = ag::cross_entropy(logits, batch.ys);
    opt->zero_grad();
    ag::backward_scalar(loss);
    auto params = model->parameters();
    clip_grad_norm(params, 1.0);
    opt->step();
    return loss->data[0];
}

inline void save_checkpoint(const std::string& suffix = "") {
    std::string filename = config.checkpoint_dir + "/carbide_ckpt" + suffix + ".bin";
    save_checkpoint_file(filename, *model, *opt);
    std::cout << "  \xE2\x9C\x93 Checkpoint saved: " << filename << "\n";
}

inline bool load_checkpoint(const std::string& suffix = "") {
    std::string filename = config.checkpoint_dir + "/carbide_ckpt" + suffix + ".bin";
    LoadedCheckpoint lc;
    if (!load_checkpoint_file(filename, lc)) {
        std::cout << "  \xE2\x9C\x97 Checkpoint not found: " << filename << "\n";
        return false;
    }
    model = std::move(lc.model);
    opt = std::move(lc.opt);
    std::cout << "  \xE2\x9C\x93 Checkpoint loaded: " << filename << " (step " << train_state.step << ")\n";
    return true;
}

inline void background_training_thread(long long n) {
    if (!model) init_model();
    train_state.training_active = true;
    long long start_step = train_state.step;
    for (long long step = start_step + 1; step <= start_step + n; ++step) {
        if (!train_state.training_active) break;
        double loss = train_step();
        {
            std::lock_guard<std::mutex> lk(train_state.training_lock);
            train_state.loss_history.push_back({step, loss});
            train_state.step = step;
            train_state.current_loss = loss;
        }
    }
    train_state.training_active = false;
}

inline void start_background_training(long long n, const std::string& display_mode = "none") {
    // a std::thread object must be joined before it's reassigned or it
    // calls std::terminate(); by the time the user starts a NEW run, any
    // previous one has already stopped (explicitly or by finishing).
    if (train_state.training_thread.joinable()) train_state.training_thread.join();
    train_state.training_thread = std::thread(background_training_thread, n);

    if (display_mode == "log") show_training_log();
    else if (display_mode == "monitor") show_live_monitor();
    else if (display_mode == "both") show_both_monitors();
}

inline void stop_background_training() {
    train_state.training_active = false;
    std::cout << "  \xE2\x9C\x93 Stopping background training...\n";
    if (train_state.training_thread.joinable()) train_state.training_thread.join();
    std::cout << "  \xE2\x9C\x93 Stopped at step " << train_state.step << "\n";
}

inline void train_n_steps(long long n) {
    if (!model) init_model();
    long long start_step = train_state.step;
    for (long long step = start_step + 1; step <= start_step + n; ++step) {
        double loss = train_step();
        train_state.loss_history.push_back({step, loss});
        train_state.step = step;
        train_state.current_loss = loss;

        if (step % std::max(1LL, n / 10) == 0 || step == start_step + 1) {
            double ppl = (loss < 10) ? std::exp(loss) : INFINITY;
            printf("  step %5lld | loss %.4f | perplexity %.1f\n", step, loss, ppl);
        }
    }
    std::cout << "  \xE2\x9C\x93 Trained " << n << " steps (total: " << train_state.step << ")\n";
}

} // namespace carbide
