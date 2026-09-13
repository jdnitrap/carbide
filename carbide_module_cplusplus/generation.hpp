// generation.hpp — mirrors carbide_modules/generation.py
#pragma once
#include "training.hpp"
#include <random>
#include <numeric>
#include <algorithm>

namespace carbide {

// Mirrors Python's bytes.decode("utf-8", errors="ignore"): walk the byte
// string, keep well-formed UTF-8 sequences, drop invalid lead/continuation
// bytes one at a time.
inline std::string utf8_filter_ignore(const std::vector<int>& bytes) {
    std::string out;
    size_t i = 0, n = bytes.size();
    auto cont = [&](size_t j) { return j < n && (bytes[j] & 0xC0) == 0x80; };
    while (i < n) {
        unsigned char b0 = (unsigned char)bytes[i];
        int len = 0;
        if (b0 < 0x80) len = 1;
        else if ((b0 & 0xE0) == 0xC0) len = 2;
        else if ((b0 & 0xF0) == 0xE0) len = 3;
        else if ((b0 & 0xF8) == 0xF0) len = 4;
        else { i++; continue; } // invalid lead byte, drop it

        bool ok = true;
        for (int k = 1; k < len; ++k) if (!cont(i + k)) { ok = false; break; }
        if (!ok) { i++; continue; } // malformed sequence, drop the lead byte only

        for (int k = 0; k < len; ++k) out += (char)(unsigned char)bytes[i + k];
        i += len;
    }
    return out;
}

inline std::string generate(const std::string& prompt, int n_bytes = 100,
                             double temperature = 0.8, int top_k = 20) {
    if (!model) return "[Model not trained yet]";

    std::vector<int> out;
    for (unsigned char c : prompt) out.push_back((int)c);

    std::mt19937 r(std::random_device{}());

    for (int step = 0; step < n_bytes; ++step) {
        int T = std::min((int)out.size(), config.seq_len);
        std::vector<int> ctx(out.end() - T, out.end());
        auto logits_t = model->forward(ctx, 1, T, Mode::Full); // (T,256)

        std::vector<double> last(VOCAB);
        for (int c = 0; c < VOCAB; ++c) last[c] = logits_t->data[(T - 1) * VOCAB + c] / temperature;

        int k = std::min(top_k, VOCAB);
        std::vector<int> idx(VOCAB);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                           [&](int a, int b) { return last[a] > last[b]; });
        double kth = last[idx[k - 1]];
        for (int c = 0; c < VOCAB; ++c) if (last[c] < kth) last[c] = -1e300;

        auto probs = ag::softmax_raw(last);
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        out.push_back(dist(r));
    }

    return utf8_filter_ignore(out);
}

} // namespace carbide
