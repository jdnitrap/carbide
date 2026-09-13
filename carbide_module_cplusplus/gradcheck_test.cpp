// gradcheck_test.cpp — finite-difference verification for every primitive
// in tensor.hpp. Nothing downstream gets built on an op until it passes
// here. Run: g++ -std=c++17 -O2 gradcheck_test.cpp -o gradcheck_test && ./gradcheck_test
#include "tensor.hpp"
#include <iostream>
#include <cstdlib>

using namespace ag;

static std::mt19937 rng(1234);

static Tensor randn_tensor(const Shape& s, bool rg) {
    auto t = make_tensor(s, rg);
    std::normal_distribution<double> d(0.0, 1.0);
    for (auto& v : t->data) v = d(rng);
    return t;
}

// rebuild() constructs a *fresh* graph each call (since
// our graph is single-use / not resettable) and returns the scalar loss.
static bool check_fn(const std::string& name, std::function<Tensor()> rebuild,
                      std::vector<Tensor>& leaves, double eps = 1e-4, double tol = 5e-3) {
    Tensor loss = rebuild();
    backward_scalar(loss);
    // stash analytic grads
    std::vector<std::vector<double>> analytic;
    for (auto& lf : leaves) analytic.push_back(lf->grad);

    bool ok = true;
    for (size_t li = 0; li < leaves.size(); ++li) {
        for (size_t i = 0; i < leaves[li]->data.size(); ++i) {
            double orig = leaves[li]->data[i];

            leaves[li]->data[i] = orig + eps;
            Tensor lp = rebuild();
            double fplus = lp->data[0];

            leaves[li]->data[i] = orig - eps;
            Tensor lm = rebuild();
            double fminus = lm->data[0];

            leaves[li]->data[i] = orig;

            double numeric = (fplus - fminus) / (2 * eps);
            double an = analytic[li][i];
            double diff = std::fabs(numeric - an);
            double scale = std::max(1.0, std::fabs(numeric));
            if (diff / scale > tol) {
                std::cout << "  FAIL " << name << " leaf#" << li << " idx " << i
                          << " analytic=" << an << " numeric=" << numeric << " diff=" << diff << "\n";
                ok = false;
            }
        }
    }
    std::cout << (ok ? "PASS " : "FAIL ") << name << "\n";
    return ok;
}

int main() {
    bool all_ok = true;

    // add / sub / mul with broadcasting
    {
        auto a = randn_tensor({2, 3}, true);
        auto b = randn_tensor({3}, true);
        std::vector<Tensor> leaves = {a, b};
        all_ok &= check_fn("add_broadcast", [&]() {
            auto y = add(a, b);
            return sum_axis(sum_axis(square(y), 0), 1);
        }, leaves);
    }
    {
        auto a = randn_tensor({2, 3}, true);
        auto b = randn_tensor({2, 3}, true);
        std::vector<Tensor> leaves = {a, b};
        all_ok &= check_fn("mul_same_shape", [&]() {
            auto y = mul(a, b);
            return sum_axis(sum_axis(square(y), 0), 1);
        }, leaves);
    }
    {
        auto a = randn_tensor({2, 4, 3, 1}, true);
        auto b = randn_tensor({3, 5}, true);
        std::vector<Tensor> leaves = {a, b};
        all_ok &= check_fn("mul_broadcast_4d", [&]() {
            auto y = mul(a, b); // -> (2,4,3,5)
            auto s1 = sum_axis(y, 3);
            auto s2 = sum_axis(s1, 2);
            auto s3 = sum_axis(s2, 1);
            return sum_axis(s3, 0);
        }, leaves);
    }
    // square, sqrt, exp, softplus, reciprocal
    {
        auto a = randn_tensor({5}, true);
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("square", [&]() { return sum_axis(square(a), 0); }, leaves);
    }
    {
        auto a = make_tensor({5}, true);
        for (auto& v : a->data) v = std::fabs(v) + 0.5; // keep sqrt domain positive
        std::normal_distribution<double> d(2.0, 0.3);
        for (auto& v : a->data) v = std::fabs(d(rng)) + 0.5;
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("sqrt", [&]() { return sum_axis(sqrt_op(a), 0); }, leaves);
    }
    {
        auto a = randn_tensor({5}, true);
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("exp", [&]() { return sum_axis(exp_op(a), 0); }, leaves);
    }
    {
        auto a = randn_tensor({6}, true);
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("softplus", [&]() { return sum_axis(softplus(a), 0); }, leaves);
    }
    {
        auto a = make_tensor({5}, true);
        std::normal_distribution<double> d(3.0, 0.4);
        for (auto& v : a->data) v = std::fabs(d(rng)) + 1.0;
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("reciprocal", [&]() { return sum_axis(reciprocal(a), 0); }, leaves);
    }
    // linear
    {
        auto x = randn_tensor({4, 3}, true);
        auto W = randn_tensor({5, 3}, true);
        auto b = randn_tensor({5}, true);
        std::vector<Tensor> leaves = {x, W, b};
        all_ok &= check_fn("linear", [&]() {
            auto y = linear(x, W, b); // (4,5)
            auto s1 = sum_axis(square(y), 1);
            return sum_axis(s1, 0);
        }, leaves);
    }
    // embedding
    {
        auto table = randn_tensor({8, 4}, true);
        std::vector<int> idx = {0, 3, 3, 7};
        std::vector<Tensor> leaves = {table};
        all_ok &= check_fn("embedding", [&]() {
            auto y = embedding(idx, table); // (4,4)
            auto s1 = sum_axis(square(y), 1);
            return sum_axis(s1, 0);
        }, leaves);
    }
    // concat_last_dim
    {
        auto a = randn_tensor({3, 2}, true);
        auto b = randn_tensor({3, 5}, true);
        std::vector<Tensor> leaves = {a, b};
        all_ok &= check_fn("concat_last_dim", [&]() {
            auto y = concat_last_dim(a, b); // (3,7)
            auto s1 = sum_axis(square(y), 1);
            return sum_axis(s1, 0);
        }, leaves);
    }
    // slice_axis + stack_new_axis (round trip)
    {
        auto a = randn_tensor({2, 3, 4}, true);
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("slice_stack_roundtrip", [&]() {
            std::vector<Tensor> slices;
            for (int t = 0; t < 3; ++t) slices.push_back(slice_axis(a, 1, t));
            auto y = stack_new_axis(slices, 1); // back to (2,3,4)
            auto s1 = sum_axis(square(y), 2);
            auto s2 = sum_axis(s1, 1);
            return sum_axis(s2, 0);
        }, leaves);
    }
    // layer_norm
    {
        auto x = randn_tensor({3, 6}, true);
        auto gamma = randn_tensor({6}, true);
        auto beta = randn_tensor({6}, true);
        std::vector<Tensor> leaves = {x, gamma, beta};
        all_ok &= check_fn("layer_norm", [&]() {
            auto y = layer_norm(x, gamma, beta);
            auto s1 = sum_axis(square(y), 1);
            return sum_axis(s1, 0);
        }, leaves);
    }
    // cross_entropy
    {
        auto logits = randn_tensor({4, 6}, true);
        std::vector<int> targets = {0, 5, 2, 2};
        std::vector<Tensor> leaves = {logits};
        all_ok &= check_fn("cross_entropy", [&]() {
            return cross_entropy(logits, targets);
        }, leaves);
    }
    // a scaled-down version of the actual SelectiveSSM recurrence
    {
        int B = 2, T = 4, d = 3, ds = 2;
        auto x = randn_tensor({B, T, d}, true);
        auto Wd_W = randn_tensor({d, d}, true), Wd_b = randn_tensor({d}, true);
        auto WB_W = randn_tensor({ds, d}, true), WB_b = randn_tensor({ds}, true);
        auto WC_W = randn_tensor({ds, d}, true), WC_b = randn_tensor({ds}, true);
        auto A_log = randn_tensor({d, ds}, true);
        auto D = randn_tensor({d}, true);
        std::vector<Tensor> leaves = {x, Wd_W, Wd_b, WB_W, WB_b, WC_W, WC_b, A_log, D};
        all_ok &= check_fn("selective_ssm_recurrence", [&]() {
            auto x2d = reshape(x, {B * T, d});
            auto delta2d = softplus(linear(x2d, Wd_W, Wd_b));
            auto delta = reshape(delta2d, {B, T, d});
            auto A = neg(exp_op(A_log)); // (d,ds)
            auto Bx2d = linear(x2d, WB_W, WB_b);
            auto Bx = reshape(Bx2d, {B, T, ds});
            auto Cx2d = linear(x2d, WC_W, WC_b);
            auto Cx = reshape(Cx2d, {B, T, ds});

            Tensor h; // (B,d,ds)
            std::vector<Tensor> ys;
            for (int t = 0; t < T; ++t) {
                auto delta_t = slice_axis(delta, 1, t);          // (B,d)
                auto delta_t_r = reshape(delta_t, {B, d, 1});
                auto A_r = reshape(A, {1, d, ds});
                auto a_bar_t = exp_op(mul(delta_t_r, A_r));       // (B,d,ds)
                auto Bx_t = slice_axis(Bx, 1, t);                  // (B,ds)
                auto Bx_t_r = reshape(Bx_t, {B, 1, ds});
                auto b_bar_t = mul(delta_t_r, Bx_t_r);             // (B,d,ds)
                if (t == 0) h = b_bar_t;
                else h = add(mul(a_bar_t, h), b_bar_t);
                auto Cx_t = slice_axis(Cx, 1, t);                  // (B,ds)
                auto Cx_t_r = reshape(Cx_t, {B, 1, ds});
                auto y_t = sum_axis(mul(Cx_t_r, h), 2);            // (B,d,1)
                ys.push_back(reshape(y_t, {B, d}));
            }
            auto y = stack_new_axis(ys, 1); // (B,T,d)
            auto out = add(y, mul(reshape(D, {1, 1, d}), x));
            auto s1 = sum_axis(square(out), 2);
            auto s2 = sum_axis(s1, 1);
            return sum_axis(s2, 0);
        }, leaves, 1e-4, 1e-2);
    }

    // slice_range_axis + concat_axis (the two ops the parallel scan needs)
    {
        auto a = randn_tensor({2, 5, 3}, true);
        std::vector<Tensor> leaves = {a};
        all_ok &= check_fn("slice_range_axis", [&]() {
            auto y = slice_range_axis(a, 1, 1, 4); // (2,3,3)
            auto s1 = sum_axis(square(y), 2);
            return sum_axis(sum_axis(s1, 1), 0);
        }, leaves);
    }
    {
        auto a = randn_tensor({2, 3, 4}, true);
        auto b = randn_tensor({2, 5, 4}, true);
        std::vector<Tensor> leaves = {a, b};
        all_ok &= check_fn("concat_axis", [&]() {
            auto y = concat_axis(a, b, 1); // (2,8,4)
            auto s1 = sum_axis(square(y), 2);
            return sum_axis(sum_axis(s1, 1), 0);
        }, leaves);
    }

    // Parallel-scan SelectiveSSM vs the sequential formulation it replaced,
    // on the SAME random inputs — not just "does the scan gradient-check in
    // isolation" but "does it compute the same function as the loop did."
    {
        int B = 2, T = 7, d = 3, ds = 2; // T=7 deliberately not a power of 2
        std::mt19937 rng2(99);
        std::normal_distribution<double> nd(0.0, 1.0);
        auto mk = [&](std::vector<int> shp) {
            Shape s(shp.begin(), shp.end());
            auto t = make_tensor(s, true);
            for (auto& v : t->data) v = nd(rng2);
            return t;
        };
        auto x = mk({B, T, d});
        auto Wd_W = mk({d, d}), Wd_b = mk({d});
        auto WB_W = mk({ds, d}), WB_b = mk({ds});
        auto WC_W = mk({ds, d}), WC_b = mk({ds});
        auto A_log = mk({d, ds});
        auto D = mk({d});

        auto sequential = [&]() {
            auto x2d = reshape(x, {B * T, d});
            auto delta = reshape(softplus(linear(x2d, Wd_W, Wd_b)), {B, T, d});
            auto A = neg(exp_op(A_log));
            auto Bx = reshape(linear(x2d, WB_W, WB_b), {B, T, ds});
            auto Cx = reshape(linear(x2d, WC_W, WC_b), {B, T, ds});
            Tensor h;
            std::vector<Tensor> ys;
            for (int t = 0; t < T; ++t) {
                auto delta_t = reshape(slice_axis(delta, 1, t), {B, d, 1});
                auto A_r = reshape(A, {1, d, ds});
                auto a_bar_t = exp_op(mul(delta_t, A_r));
                auto Bx_t = reshape(slice_axis(Bx, 1, t), {B, 1, ds});
                auto b_bar_t = mul(delta_t, Bx_t);
                if (t == 0) h = b_bar_t; else h = add(mul(a_bar_t, h), b_bar_t);
                auto Cx_t = reshape(slice_axis(Cx, 1, t), {B, 1, ds});
                ys.push_back(reshape(sum_axis(mul(Cx_t, h), 2), {B, d}));
            }
            auto y = stack_new_axis(ys, 1);
            return add(y, mul(reshape(D, {1, 1, d}), x));
        };

        auto scan = [&]() {
            auto x2d = reshape(x, {B * T, d});
            auto delta = reshape(softplus(linear(x2d, Wd_W, Wd_b)), {B, T, d});
            auto A = neg(exp_op(A_log));
            auto Bx = reshape(linear(x2d, WB_W, WB_b), {B, T, ds});
            auto Cx = reshape(linear(x2d, WC_W, WC_b), {B, T, ds});
            auto delta4 = reshape(delta, {B, T, d, 1});
            auto A4 = reshape(A, {1, 1, d, ds});
            auto a_bar = exp_op(mul(delta4, A4));
            auto Bx4 = reshape(Bx, {B, T, 1, ds});
            auto b_bar = mul(delta4, Bx4);
            Tensor A_scan = a_bar, B_scan = b_bar;
            for (int s = 1; s < T; s *= 2) {
                auto A_left = slice_range_axis(A_scan, 1, 0, T - s);
                auto A_right = slice_range_axis(A_scan, 1, s, T);
                auto B_left = slice_range_axis(B_scan, 1, 0, T - s);
                auto B_right = slice_range_axis(B_scan, 1, s, T);
                auto newA_tail = mul(A_left, A_right);
                auto newB_tail = add(mul(A_right, B_left), B_right);
                auto A_head = slice_range_axis(A_scan, 1, 0, s);
                auto B_head = slice_range_axis(B_scan, 1, 0, s);
                A_scan = concat_axis(A_head, newA_tail, 1);
                B_scan = concat_axis(B_head, newB_tail, 1);
            }
            auto Cx4 = reshape(Cx, {B, T, 1, ds});
            auto y4 = mul(Cx4, B_scan);
            auto y = reshape(sum_axis(y4, 3), {B, T, d});
            return add(y, mul(reshape(D, {1, 1, d}), x));
        };

        std::vector<Tensor> leaves = {x, Wd_W, Wd_b, WB_W, WB_b, WC_W, WC_b, A_log, D};

        auto out_seq = sequential();
        auto out_scan = scan();
        double maxdiff = 0;
        for (size_t i = 0; i < out_seq->data.size(); ++i)
            maxdiff = std::max(maxdiff, std::fabs(out_seq->data[i] - out_scan->data[i]));
        std::cout << (maxdiff < 1e-9 ? "PASS " : "FAIL ") << "scan_matches_sequential (forward, maxdiff="
                  << maxdiff << ")\n";
        all_ok &= (maxdiff < 1e-9);

        // and the scan formulation gradient-checks correctly on its own
        all_ok &= check_fn("scan_gradcheck", [&]() {
            auto out = scan();
            auto s1 = sum_axis(square(out), 2);
            auto s2 = sum_axis(s1, 1);
            return sum_axis(s2, 0);
        }, leaves, 1e-4, 1e-2);

        // gradients also match the sequential formulation's, not just the
        // scan's own finite-difference check
        auto loss_seq = [&]() {
            auto out = sequential();
            auto s1 = sum_axis(square(out), 2);
            auto s2 = sum_axis(s1, 1);
            return sum_axis(s2, 0);
        };
        auto loss_scan = [&]() {
            auto out = scan();
            auto s1 = sum_axis(square(out), 2);
            auto s2 = sum_axis(s1, 1);
            return sum_axis(s2, 0);
        };
        auto l1 = loss_seq(); backward_scalar(l1);
        std::vector<std::vector<double>> grads_seq;
        for (auto& lf : leaves) grads_seq.push_back(lf->grad);
        auto l2 = loss_scan(); backward_scalar(l2);
        double gmax = 0;
        for (size_t li = 0; li < leaves.size(); ++li)
            for (size_t i = 0; i < leaves[li]->grad.size(); ++i)
                gmax = std::max(gmax, std::fabs(grads_seq[li][i] - leaves[li]->grad[i]));
        std::cout << (gmax < 1e-8 ? "PASS " : "FAIL ") << "scan_matches_sequential (gradients, maxdiff="
                  << gmax << ")\n";
        all_ok &= (gmax < 1e-8);
    }

    std::cout << (all_ok ? "\nALL GRADIENT CHECKS PASSED\n" : "\nSOME GRADIENT CHECKS FAILED\n");
    return all_ok ? 0 : 1;
}
