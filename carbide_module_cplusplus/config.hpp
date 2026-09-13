// config.hpp — mirrors carbide_modules/config.py
#pragma once
#include <string>
#include <filesystem>

namespace carbide {

struct Config {
    int d_model = 64;
    int n_layers = 2;
    int d_state = 16;
    int batch_size = 1;
    int seq_len = 128;
    double learning_rate = 3e-3;
    int steps_total = 3000;
    std::string checkpoint_dir = "carbide_checkpoints";
    std::string data_file = "";

    Config() {
        std::filesystem::create_directories(checkpoint_dir);
    }
};

inline Config config;

} // namespace carbide
