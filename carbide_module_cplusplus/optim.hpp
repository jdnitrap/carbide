// optim.hpp — AdamW optimizer and gradient clipping, from scratch.
// Hyperparameters match torch.optim.AdamW's defaults (betas, eps, weight
// decay), since the Python original only ever passes `lr` explicitly.
#pragma once
#include "tensor.hpp"
#include <cmath>
#include <algorithm>

namespace carbide {

struct AdamW {
    std::vector<ag::Tensor> params;
    double lr;
    double beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 0.01;
    std::vector<std::vector<double>> m, v;
    long long t = 0;

    AdamW(std::vector<ag::Tensor> params_, double lr_) : params(std::move(params_)), lr(lr_) {
        for (auto& p : params) {
            m.emplace_back(p->data.size(), 0.0);
            v.emplace_back(p->data.size(), 0.0);
        }
    }

    void zero_grad() {
        for (auto& p : params) std::fill(p->grad.begin(), p->grad.end(), 0.0);
    }

    void step() {
        t++;
        double bc1 = 1.0 - std::pow(beta1, (double)t);
        double bc2 = 1.0 - std::pow(beta2, (double)t);
        for (size_t pi = 0; pi < params.size(); ++pi) {
            auto& p = params[pi];
            for (size_t i = 0; i < p->data.size(); ++i) {
                double g = p->grad[i];
                p->data[i] -= lr * weight_decay * p->data[i]; // decoupled weight decay
                m[pi][i] = beta1 * m[pi][i] + (1 - beta1) * g;
                v[pi][i] = beta2 * v[pi][i] + (1 - beta2) * g * g;
                double mhat = m[pi][i] / bc1;
                double vhat = v[pi][i] / bc2;
                p->data[i] -= lr * mhat / (std::sqrt(vhat) + eps);
            }
        }
    }
};

inline double grad_global_norm(std::vector<ag::Tensor>& params) {
    double sumsq = 0;
    for (auto& p : params) for (double g : p->grad) sumsq += g * g;
    return std::sqrt(sumsq);
}

inline void clip_grad_norm(std::vector<ag::Tensor>& params, double max_norm) {
    double norm = grad_global_norm(params);
    if (norm > max_norm) {
        double scale = max_norm / (norm + 1e-6);
        for (auto& p : params) for (auto& g : p->grad) g *= scale;
    }
}

} // namespace carbide
