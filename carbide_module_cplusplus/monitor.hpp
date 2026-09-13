// monitor.hpp — mirrors carbide_modules/monitor.py
#pragma once
#include "config.hpp"
#include "state.hpp"
#include "dataset.hpp"
#include "interrupt.hpp"
#include <cstdio>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>
#include <iostream>

namespace carbide {

inline std::string draw_ascii_graph(const std::vector<double>& losses, int max_width = 60, int max_height = 8) {
    if (losses.empty()) return "No data yet";
    int start = (int)losses.size() > max_width ? (int)losses.size() - max_width : 0;
    std::vector<double> recent(losses.begin() + start, losses.end());
    if (recent.empty()) return "No data";

    double mn = *std::min_element(recent.begin(), recent.end());
    double mx = *std::max_element(recent.begin(), recent.end());
    double range = (mx > mn) ? (mx - mn) : 1.0;

    std::vector<int> scaled;
    for (double l : recent) {
        int sv = (int)((l - mn) / range * (max_height - 1));
        scaled.push_back(std::max(0, std::min(max_height - 1, sv)));
    }

    std::ostringstream out;
    for (int h = max_height - 1; h >= 0; --h) {
        out << "\xE2\x94\x82 ";
        for (int v : scaled) {
            if (v == h) out << "\xE2\x96\x88";
            else if (v > h) out << "\xE2\x94\x82";
            else out << " ";
        }
        out << " \xE2\x94\x82\n";
    }
    out << "\xE2\x94\x94" << std::string(scaled.size() + 1, '-') << "\xE2\x94\x98";
    return out.str();
}

inline void print_live_status(long long total_steps) {
    double elapsed = train_state.get_elapsed();
    double speed = train_state.get_speed();

    int eta_min = 0, eta_sec = 0;
    if (train_state.step > 0 && speed > 0) {
        double eta = (double)(total_steps - train_state.step) / speed;
        eta_min = (int)(eta / 60);
        eta_sec = (int)eta % 60;
    }

    double ppl = (train_state.current_loss < 10) ? std::exp(train_state.current_loss) : INFINITY;

    if (std::system("clear") != 0) { /* non-fatal: terminal just won't clear */ }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "CARBIDE LIVE MONITOR\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Dataset: " << config.data_file << " (" << data.size() << " bytes)\n\n";

    double progress = total_steps > 0 ? ((double)train_state.step / total_steps * 50) : 0;
    std::string bar(std::max(0, (int)progress), '\xDB');
    std::cout << "Progress: [" << std::string((int)progress, '#') << std::string(50 - (int)progress, '.')
              << "] " << train_state.step << "/" << total_steps << "\n\n";

    printf("Loss:       %8.4f\n", train_state.current_loss);
    printf("Perplexity: %8.1f\n", ppl);
    printf("Speed:      %8.2f steps/sec\n", speed);
    printf("Elapsed:    %3d:%02d\n", (int)elapsed / 60, (int)elapsed % 60);
    printf("ETA:        %3d:%02d\n\n", eta_min, eta_sec);

    if (!train_state.loss_history.empty()) {
        std::cout << "Loss Curve (last 60 steps):\n";
        std::vector<double> losses;
        for (auto& [s, l] : train_state.loss_history) losses.push_back(l);
        std::cout << draw_ascii_graph(losses, 60, 7) << "\n";
    } else {
        std::cout << "Loss Curve: (waiting for first step)\n";
    }
    std::cout << "\n" << std::string(70, '=') << "\n";
}

inline void show_training_log() {
    long long last_printed = 0;
    std::cout << "\n" << std::string(70, '=') << "\nTRAINING LOG (Option 1)\n" << std::string(70, '=') << "\n";
    std::cout << "Press Ctrl+C to return to menu (training continues in background)\n\n";

    while (train_state.training_active) {
        if (g_interrupted) {
            clear_interrupt();
            std::cout << "\n\xE2\x9C\x93 Returned to menu (training continues in background)\n";
            return;
        }
        long long cur_step; double cur_loss;
        { std::lock_guard<std::mutex> lk(train_state.training_lock);
          cur_step = train_state.step; cur_loss = train_state.current_loss; }
        if (cur_step > last_printed) {
            if (cur_step % 10 == 0 || cur_step == 1) {
                double ppl = (cur_loss < 10) ? std::exp(cur_loss) : INFINITY;
                printf("step %5lld | loss %.4f | perplexity %.1f\n", cur_step, cur_loss, ppl);
            }
            last_printed = cur_step;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "\n\xE2\x9C\x93 Training complete at step " << train_state.step << "\n";
}

inline void show_live_monitor() {
    std::cout << "\nStarting live monitor...\n";
    std::cout << "Press Ctrl+C to return to menu (training continues in background)\n\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    while (train_state.training_active) {
        if (g_interrupted) {
            clear_interrupt();
            std::cout << "\n\xE2\x9C\x93 Returned to menu (training continues in background)\n";
            return;
        }
        { std::lock_guard<std::mutex> lk(train_state.training_lock);
          if (!train_state.loss_history.empty()) print_live_status(train_state.training_target_steps); }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "\n\xE2\x9C\x93 Training complete at step " << train_state.step << "\n";
}

inline void show_both_monitors() {
    long long last_printed = 0;
    std::cout << "\n" << std::string(70, '=') << "\nTRAINING LOG + LIVE MONITOR (Option 3)\n" << std::string(70, '=') << "\n";
    std::cout << "Press Ctrl+C to return to menu (training continues in background)\n\n";

    long long update_counter = 0;
    while (train_state.training_active) {
        if (g_interrupted) {
            clear_interrupt();
            std::cout << "\n\xE2\x9C\x93 Returned to menu (training continues in background)\n";
            return;
        }
        long long cur_step; double cur_loss;
        { std::lock_guard<std::mutex> lk(train_state.training_lock);
          cur_step = train_state.step; cur_loss = train_state.current_loss; }

        if (cur_step > last_printed && cur_step % 10 == 0) {
            double ppl = (cur_loss < 10) ? std::exp(cur_loss) : INFINITY;
            printf("step %5lld | loss %.4f | perplexity %.1f\n", cur_step, cur_loss, ppl);
            last_printed = cur_step;
        }
        if (update_counter % 20 == 0) {
            std::lock_guard<std::mutex> lk(train_state.training_lock);
            if (!train_state.loss_history.empty()) print_live_status(train_state.training_target_steps);
        }
        update_counter++;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::cout << "\n\xE2\x9C\x93 Training complete at step " << train_state.step << "\n";
}

} // namespace carbide
