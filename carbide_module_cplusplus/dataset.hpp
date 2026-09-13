// dataset.hpp — mirrors carbide_modules/dataset.py
#pragma once
#include "config.hpp"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <filesystem>
#include <iostream>

namespace carbide {

inline const std::string FALLBACK =
    "the little cat saw the sun. the sun was big and warm. "
    "the cat ran to the hill. the hill was green. ";

// unsigned bytes, 0-255 — same representation as Python's list(text.encode("utf-8"))
inline std::vector<int> data;

inline std::mt19937& rng() {
    static std::mt19937 r(std::random_device{}());
    return r;
}

inline std::vector<int> bytes_of(const std::string& text) {
    std::vector<int> out;
    out.reserve(text.size());
    for (unsigned char c : text) out.push_back((int)c);
    return out;
}

inline bool read_file_text(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Load dataset from file or use default. Returns true on success.
inline bool load_dataset(const std::string& filepath = "") {
    if (!filepath.empty()) {
        if (!std::filesystem::exists(filepath)) {
            std::cout << "  \xE2\x9C\x97 File not found: " << filepath << "\n";
            std::cout << "  Current directory: " << std::filesystem::current_path().string() << "\n";
            std::string avail;
            for (auto& e : std::filesystem::directory_iterator(".")) {
                if (e.path().extension() == ".txt") { if (!avail.empty()) avail += ", "; avail += e.path().filename().string(); }
            }
            std::cout << "  Available .txt files: " << (avail.empty() ? "(none)" : avail) << "\n";
            return false;
        }
        if (!std::filesystem::is_regular_file(filepath)) {
            std::cout << "  \xE2\x9C\x97 Not a file: " << filepath << "\n";
            return false;
        }
        std::string text;
        if (!read_file_text(filepath, text)) {
            std::cout << "  \xE2\x9C\x97 Error loading file\n";
            return false;
        }
        data = bytes_of(text);
        config.data_file = filepath;
        std::cout << "  \xE2\x9C\x93 Loaded " << filepath << "\n";
        std::cout << "  \xE2\x9C\x93 " << text.size() << " chars, " << data.size() << " bytes\n";
        return true;
    } else {
        std::string text;
        if (std::filesystem::exists("train.txt") && read_file_text("train.txt", text)) {
            config.data_file = "train.txt";
            std::cout << "\xE2\x9C\x93 loaded train.txt (" << text.size() << " chars)\n";
        } else {
            text.clear();
            for (int i = 0; i < 500; ++i) text += FALLBACK;
            config.data_file = "[fallback]";
            std::cout << "\xE2\x9C\x93 using built-in fallback sample\n";
        }
        data = bytes_of(text);
        std::cout << "\xE2\x9C\x93 " << data.size() << " bytes total\n\n";
        return true;
    }
}

struct Batch { std::vector<int> xs, ys; int B, T; };

inline Batch get_batch(int batch = -1, int seq_len = -1) {
    if (batch < 0) batch = config.batch_size;
    if (seq_len < 0) seq_len = config.seq_len;
    std::uniform_int_distribution<int> pick(0, (int)data.size() - seq_len - 1 - 1);
    Batch b; b.B = batch; b.T = seq_len;
    b.xs.resize((size_t)batch * seq_len);
    b.ys.resize((size_t)batch * seq_len);
    for (int n = 0; n < batch; ++n) {
        int i = pick(rng());
        for (int t = 0; t < seq_len; ++t) {
            b.xs[(size_t)n * seq_len + t] = data[i + t];
            b.ys[(size_t)n * seq_len + t] = data[i + t + 1];
        }
    }
    return b;
}

} // namespace carbide
