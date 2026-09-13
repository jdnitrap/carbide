// model_smoke_test.cpp — trains the real Carbide architecture for a few
// steps on synthetic data and checks: (1) param count matches the Python
// version's default config exactly, (2) loss actually decreases.
#include "model.hpp"
#include "optim.hpp"
#include <iostream>
#include <random>

using namespace carbide;

int main() {
    std::mt19937 rng(42);

    // default config, same as carbide_modules/config.py
    int d_model = 64, n_layers = 2, d_state = 16;
    Carbide model(d_model, n_layers, d_state, rng);

    long long n_params = model.num_params();
    std::cout << "param count: " << n_params << " (Python default: 52608)\n";
    if (n_params != 52608) {
        std::cout << "FAIL: param count mismatch\n";
        return 1;
    }
    std::cout << "PASS: param count matches Python exactly\n";

    AdamW opt(model.parameters(), 3e-3);

    // synthetic byte data: same repeated pattern as carbide's FALLBACK text,
    // batch=1, seq_len=32 (small, for a fast smoke test)
    std::string text = "the little cat saw the sun. the sun was big and warm. ";
    std::vector<int> corpus;
    for (int rep = 0; rep < 50; ++rep)
        for (unsigned char c : text) corpus.push_back((int)c);

    int B = 1, T = 32;
    std::uniform_int_distribution<int> pick(0, (int)corpus.size() - T - 2);

    std::vector<double> losses;
    for (int step = 0; step < 60; ++step) {
        int start = pick(rng);
        std::vector<int> xb(corpus.begin() + start, corpus.begin() + start + T);
        std::vector<int> yb(corpus.begin() + start + 1, corpus.begin() + start + T + 1);

        auto logits = model.forward(xb, B, T, Mode::Full);
        auto loss = ag::cross_entropy(logits, yb);

        opt.zero_grad();
        ag::backward_scalar(loss);
        auto params = model.parameters();
        clip_grad_norm(params, 1.0);
        opt.step();

        losses.push_back(loss->data[0]);
        if (step % 10 == 0 || step == 59)
            std::cout << "step " << step << " loss " << loss->data[0] << "\n";
    }

    double first_avg = 0, last_avg = 0;
    for (int i = 0; i < 5; ++i) first_avg += losses[i];
    for (int i = 0; i < 5; ++i) last_avg += losses[losses.size() - 1 - i];
    first_avg /= 5; last_avg /= 5;
    std::cout << "avg loss first 5 steps: " << first_avg << "\n";
    std::cout << "avg loss last 5 steps:  " << last_avg << "\n";

    if (last_avg >= first_avg) {
        std::cout << "FAIL: loss did not decrease\n";
        return 1;
    }
    std::cout << "PASS: loss decreased over training\n";
    return 0;
}
