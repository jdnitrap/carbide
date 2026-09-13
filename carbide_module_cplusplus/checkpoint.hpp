// checkpoint.hpp — custom binary checkpoint format (no torch.save
// available in a from-scratch build). Mirrors what carbide_modules/
// training.py's save_checkpoint/load_checkpoint persist: model weights,
// optimizer moments, train_state step/loss_history, and the shape config.
#pragma once
#include "model.hpp"
#include "optim.hpp"
#include "state.hpp"
#include <fstream>
#include <cstdint>
#include <stdexcept>

namespace carbide {

namespace ckpt_detail {
    inline void w_i32(std::ofstream& f, int32_t v) { f.write((const char*)&v, sizeof(v)); }
    inline void w_i64(std::ofstream& f, int64_t v) { f.write((const char*)&v, sizeof(v)); }
    inline void w_f64(std::ofstream& f, double v) { f.write((const char*)&v, sizeof(v)); }
    inline void w_vec(std::ofstream& f, const std::vector<double>& v) {
        w_i64(f, (int64_t)v.size());
        if (!v.empty()) f.write((const char*)v.data(), (std::streamsize)(v.size() * sizeof(double)));
    }
    inline int32_t r_i32(std::ifstream& f) { int32_t v; f.read((char*)&v, sizeof(v)); return v; }
    inline int64_t r_i64(std::ifstream& f) { int64_t v; f.read((char*)&v, sizeof(v)); return v; }
    inline double r_f64(std::ifstream& f) { double v; f.read((char*)&v, sizeof(v)); return v; }
    inline std::vector<double> r_vec(std::ifstream& f) {
        int64_t n = r_i64(f);
        std::vector<double> v(n);
        if (n > 0) f.read((char*)v.data(), (std::streamsize)(n * sizeof(double)));
        return v;
    }
}

constexpr int32_t CKPT_MAGIC = 0x43424231; // "CBB1"

inline void save_checkpoint_file(const std::string& filename, Carbide& model, AdamW& opt) {
    using namespace ckpt_detail;
    std::ofstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open checkpoint for write: " + filename);

    w_i32(f, CKPT_MAGIC);
    w_i32(f, config.d_model);
    w_i32(f, config.n_layers);
    w_i32(f, config.d_state);
    w_f64(f, config.learning_rate);

    w_i64(f, train_state.step);
    w_i64(f, (int64_t)train_state.loss_history.size());
    for (auto& [s, l] : train_state.loss_history) { w_i64(f, s); w_f64(f, l); }

    auto params = model.parameters();
    w_i64(f, (int64_t)params.size());
    for (auto& p : params) w_vec(f, p->data);

    w_i64(f, opt.t);
    w_i64(f, (int64_t)opt.m.size());
    for (auto& mv : opt.m) w_vec(f, mv);
    for (auto& vv : opt.v) w_vec(f, vv);
}

// Reconstructs config + a fresh model/optimizer from a checkpoint file.
// Caller then swaps these into the live `model`/`opt` unique_ptrs.
struct LoadedCheckpoint {
    std::unique_ptr<Carbide> model;
    std::unique_ptr<AdamW> opt;
};

inline bool load_checkpoint_file(const std::string& filename, LoadedCheckpoint& out) {
    using namespace ckpt_detail;
    std::ifstream f(filename, std::ios::binary);
    if (!f) return false;

    int32_t magic = r_i32(f);
    if (magic != CKPT_MAGIC) throw std::runtime_error("bad checkpoint magic in " + filename);

    config.d_model = r_i32(f);
    config.n_layers = r_i32(f);
    config.d_state = r_i32(f);
    config.learning_rate = r_f64(f);

    train_state.step = r_i64(f);
    int64_t nloss = r_i64(f);
    train_state.loss_history.clear();
    train_state.loss_history.reserve(nloss);
    for (int64_t i = 0; i < nloss; ++i) {
        int64_t s = r_i64(f);
        double l = r_f64(f);
        train_state.loss_history.push_back({s, l});
    }

    std::mt19937 rng(std::random_device{}());
    out.model = std::make_unique<Carbide>(config.d_model, config.n_layers, config.d_state, rng);
    auto params = out.model->parameters();

    int64_t nparams = r_i64(f);
    if (nparams != (int64_t)params.size())
        throw std::runtime_error("checkpoint param count mismatch");
    for (auto& p : params) {
        auto v = r_vec(f);
        if (v.size() != p->data.size()) throw std::runtime_error("checkpoint param size mismatch");
        p->data = v;
    }

    out.opt = std::make_unique<AdamW>(params, config.learning_rate);
    out.opt->t = r_i64(f);
    int64_t nm = r_i64(f);
    if (nm != (int64_t)out.opt->m.size()) throw std::runtime_error("checkpoint optimizer state mismatch");
    for (auto& mv : out.opt->m) mv = r_vec(f);
    for (auto& vv : out.opt->v) vv = r_vec(f);

    return true;
}

} // namespace carbide
