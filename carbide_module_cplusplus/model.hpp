// model.hpp — MDBE embedding + SelectiveSSM + Block + Carbide, built on the
// from-scratch autodiff engine in tensor.hpp. Mirrors carbide_modules/mdbe.py
// formula-for-formula.
#pragma once
#include "tensor.hpp"
#include <random>

namespace carbide {

using ag::Tensor;

constexpr int NUM_CONSTRAINTS = 6;
constexpr int VOCAB = 256;

enum class Mode { Full, NoConstraints, PlainEmbedding };

// 6 hand-coded, deterministic per-byte flags — never learned, no gradient.
inline Tensor mdbe_constraints(const std::vector<int>& bytes_flat, bool live) {
    long long N = (long long)bytes_flat.size();
    auto out = ag::make_tensor({(int)N, NUM_CONSTRAINTS}, false);
    if (!live) return out; // all zeros, matches Python's torch.zeros(...) when live=False
    for (long long n = 0; n < N; ++n) {
        int b = bytes_flat[n];
        double is_alpha = ((b >= 65 && b <= 90) || (b >= 97 && b <= 122)) ? 1.0 : 0.0;
        double is_digit = (b >= 48 && b <= 57) ? 1.0 : 0.0;
        double is_upper = (b >= 65 && b <= 90) ? 1.0 : 0.0;
        double is_punct = ((b >= 33 && b <= 47) || (b >= 58 && b <= 64) ||
                            (b >= 91 && b <= 96) || (b >= 123 && b <= 126)) ? 1.0 : 0.0;
        double is_space = (b == 32 || b == 9 || b == 10 || b == 13) ? 1.0 : 0.0;
        double utf8_lead = ((b >= 0 && b < 0x80) || (b >= 0xC0 && b <= 0xFF)) ? 1.0 : 0.0;
        out->data[n * NUM_CONSTRAINTS + 0] = is_alpha;
        out->data[n * NUM_CONSTRAINTS + 1] = is_digit;
        out->data[n * NUM_CONSTRAINTS + 2] = is_upper;
        out->data[n * NUM_CONSTRAINTS + 3] = is_punct;
        out->data[n * NUM_CONSTRAINTS + 4] = is_space;
        out->data[n * NUM_CONSTRAINTS + 5] = utf8_lead;
    }
    return out;
}

inline void init_linear(int out_f, int in_f, std::mt19937& rng, Tensor& W, Tensor& b) {
    W = ag::make_tensor({out_f, in_f}, true);
    b = ag::make_tensor({out_f}, true);
    double bound = 1.0 / std::sqrt((double)in_f); // matches torch.nn.Linear default init
    std::uniform_real_distribution<double> u(-bound, bound);
    for (auto& v : W->data) v = u(rng);
    for (auto& v : b->data) v = u(rng);
}

struct MDBE {
    int d_model;
    Tensor base;             // (256, d_model) — learned embedding table
    Tensor proj_W, proj_b;   // Linear(d_model+6, d_model)

    MDBE(int d_model_, std::mt19937& rng) : d_model(d_model_) {
        base = ag::make_tensor({VOCAB, d_model}, true);
        std::normal_distribution<double> emb_init(0.0, 1.0); // matches torch.nn.Embedding default init
        for (auto& v : base->data) v = emb_init(rng);
        init_linear(d_model, d_model + NUM_CONSTRAINTS, rng, proj_W, proj_b);
    }

    Tensor forward(const std::vector<int>& bytes_flat, int B, int T, bool live) {
        Tensor learned = ag::embedding(bytes_flat, base);            // (B*T, d_model)
        Tensor cols = mdbe_constraints(bytes_flat, live);             // (B*T, 6)
        Tensor cat = ag::concat_last_dim(learned, cols);              // (B*T, d_model+6)
        Tensor out = ag::linear(cat, proj_W, proj_b);                 // (B*T, d_model)
        return ag::reshape(out, {B, T, d_model});
    }

    std::vector<Tensor> parameters() { return {base, proj_W, proj_b}; }
};

struct SelectiveSSM {
    int d_model, d_state;
    Tensor A_log;                     // (d_model, d_state)
    Tensor Wd_W, Wd_b;                // Linear(d_model, d_model)
    Tensor WB_W, WB_b;                // Linear(d_model, d_state)
    Tensor WC_W, WC_b;                // Linear(d_model, d_state)
    Tensor D;                         // (d_model,)

    SelectiveSSM(int d_model_, int d_state_, std::mt19937& rng)
        : d_model(d_model_), d_state(d_state_) {
        A_log = ag::make_tensor({d_model, d_state}, true);
        // Python: torch.log(torch.exp(uniform(1,16))) — algebraically just uniform(1,16)
        std::uniform_real_distribution<double> au(1.0, 16.0);
        for (auto& v : A_log->data) v = au(rng);

        init_linear(d_model, d_model, rng, Wd_W, Wd_b);
        init_linear(d_state, d_model, rng, WB_W, WB_b);
        init_linear(d_state, d_model, rng, WC_W, WC_b);

        D = ag::make_tensor({d_model}, true);
        for (auto& v : D->data) v = 1.0;
    }

    // x: (B,T,d_model) -> (B,T,d_model)
    //
    // Sequential scan (Option 1's fast-path add/mul + the hoisted A_r reshape
    // still apply here — this is the verified-fastest version measured on
    // this machine: 21.2 steps/sec at default config vs. 7.7 unoptimized).
    // A parallel/associative-scan rewrite was tried and reverted — it was
    // 4.3 steps/sec, slower than even the unoptimized loop, because each of
    // its O(log T) rounds re-materializes full (B,T,d_model,d_state)-sized
    // arrays via slice+concat rather than updating in place, so it does more
    // total work and heap traffic than this loop despite fewer op calls.
    // The primitives it needed (slice_range_axis, concat_axis in tensor.hpp)
    // are left in place, gradient-checked, just unused by this function.
    Tensor forward(const Tensor& x, int B, int T) {
        auto x2d = ag::reshape(x, {B * T, d_model});
        auto delta = ag::reshape(ag::softplus(ag::linear(x2d, Wd_W, Wd_b)), {B, T, d_model});
        auto A = ag::neg(ag::exp_op(A_log));                           // (d_model, d_state)
        auto Bx = ag::reshape(ag::linear(x2d, WB_W, WB_b), {B, T, d_state});
        auto Cx = ag::reshape(ag::linear(x2d, WC_W, WC_b), {B, T, d_state});

        auto A_r = ag::reshape(A, {1, d_model, d_state}); // invariant across t — computed once, not per step

        Tensor h;
        std::vector<Tensor> ys;
        for (int t = 0; t < T; ++t) {
            auto delta_t = ag::reshape(ag::slice_axis(delta, 1, t), {B, d_model, 1});
            auto a_bar_t = ag::exp_op(ag::mul(delta_t, A_r));           // (B,d_model,d_state)
            auto Bx_t = ag::reshape(ag::slice_axis(Bx, 1, t), {B, 1, d_state});
            auto b_bar_t = ag::mul(delta_t, Bx_t);                      // (B,d_model,d_state)
            if (t == 0) h = b_bar_t;
            else h = ag::add(ag::mul(a_bar_t, h), b_bar_t);
            auto Cx_t = ag::reshape(ag::slice_axis(Cx, 1, t), {B, 1, d_state});
            auto y_t = ag::reshape(ag::sum_axis(ag::mul(Cx_t, h), 2), {B, d_model});
            ys.push_back(y_t);
        }
        auto y = ag::stack_new_axis(ys, 1);                             // (B,T,d_model)
        return ag::add(y, ag::mul(ag::reshape(D, {1, 1, d_model}), x));
    }

    std::vector<Tensor> parameters() { return {A_log, Wd_W, Wd_b, WB_W, WB_b, WC_W, WC_b, D}; }
};

struct Block {
    int d_model;
    Tensor ln_gamma, ln_beta;
    SelectiveSSM ssm;

    Block(int d_model_, int d_state, std::mt19937& rng)
        : d_model(d_model_), ssm(d_model_, d_state, rng) {
        ln_gamma = ag::make_tensor({d_model}, true);
        ln_beta = ag::make_tensor({d_model}, true);
        for (auto& v : ln_gamma->data) v = 1.0;
        for (auto& v : ln_beta->data) v = 0.0;
    }

    Tensor forward(const Tensor& x, int B, int T) {
        auto normed = ag::layer_norm(x, ln_gamma, ln_beta);
        auto ssm_out = ssm.forward(normed, B, T);
        return ag::add(x, ssm_out);
    }

    std::vector<Tensor> parameters() {
        auto p = ssm.parameters();
        p.push_back(ln_gamma);
        p.push_back(ln_beta);
        return p;
    }
};

struct Carbide {
    int d_model, n_layers;
    MDBE mdbe;
    std::vector<Block> blocks;
    Tensor head_norm_gamma, head_norm_beta;
    Tensor head_W, head_b; // Linear(d_model, 256)

    Carbide(int d_model_, int n_layers_, int d_state, std::mt19937& rng)
        : d_model(d_model_), n_layers(n_layers_), mdbe(d_model_, rng) {
        for (int i = 0; i < n_layers; ++i) blocks.emplace_back(d_model_, d_state, rng);
        head_norm_gamma = ag::make_tensor({d_model}, true);
        head_norm_beta = ag::make_tensor({d_model}, true);
        for (auto& v : head_norm_gamma->data) v = 1.0;
        for (auto& v : head_norm_beta->data) v = 0.0;
        init_linear(VOCAB, d_model, rng, head_W, head_b);
    }

    // bytes_flat has B*T entries in [0,255]; returns logits (B*T, 256)
    Tensor forward(const std::vector<int>& bytes_flat, int B, int T, Mode mode = Mode::Full) {
        Tensor x;
        if (mode == Mode::PlainEmbedding) {
            x = ag::reshape(ag::embedding(bytes_flat, mdbe.base), {B, T, d_model});
        } else {
            x = mdbe.forward(bytes_flat, B, T, mode == Mode::Full);
        }
        for (auto& blk : blocks) x = blk.forward(x, B, T);
        auto normed = ag::layer_norm(x, head_norm_gamma, head_norm_beta);
        auto normed2d = ag::reshape(normed, {B * T, d_model});
        return ag::linear(normed2d, head_W, head_b);
    }

    std::vector<Tensor> parameters() {
        std::vector<Tensor> p = mdbe.parameters();
        for (auto& b : blocks) { auto bp = b.parameters(); p.insert(p.end(), bp.begin(), bp.end()); }
        p.push_back(head_norm_gamma);
        p.push_back(head_norm_beta);
        p.push_back(head_W);
        p.push_back(head_b);
        return p;
    }

    long long num_params() {
        long long n = 0;
        for (auto& p : parameters()) n += ag::total_size(p->shape);
        return n;
    }
};

} // namespace carbide
