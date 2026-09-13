// tensor.hpp — a small reverse-mode autodiff tensor engine, written from
// scratch (no PyTorch/libtorch/Eigen — only the C++ standard library).
//
// Design: each Tensor is a shared_ptr<TensorImpl> node in a DAG. Every op
// (add, mul, linear, ...) builds a new node whose backward_fn knows how to
// push gradient into its parents. backward() does a DFS topological sort
// from the root and replays backward_fn in reverse order (classic
// reverse-mode autodiff, same idea as micrograd/tinygrad).
//
// Every primitive here is covered by finite-difference gradient checks in
// gradcheck_test.cpp — do not trust a new op until it passes there too.
#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>
#include <stdexcept>
#include <cmath>
#include <random>
#include <numeric>
#include <sstream>
#include <algorithm>

namespace ag {

using Shape = std::vector<int>;

inline long long total_size(const Shape& s) {
    long long n = 1;
    for (int d : s) n *= d;
    return n;
}

inline std::vector<long long> row_major_strides(const Shape& s) {
    std::vector<long long> strides(s.size());
    long long acc = 1;
    for (int i = (int)s.size() - 1; i >= 0; --i) {
        strides[i] = acc;
        acc *= s[i];
    }
    return strides;
}

inline void unravel_into(long long flat, const Shape& shape, std::vector<int>& idx_out) {
    idx_out.resize(shape.size());
    for (int i = (int)shape.size() - 1; i >= 0; --i) {
        idx_out[i] = (int)(flat % shape[i]);
        flat /= shape[i];
    }
}

inline std::string shape_str(const Shape& s) {
    std::ostringstream o;
    o << "(";
    for (size_t i = 0; i < s.size(); ++i) { o << s[i]; if (i + 1 < s.size()) o << ","; }
    o << ")";
    return o.str();
}

// ---- NumPy-style right-aligned broadcasting ----

inline Shape broadcast_shape(const Shape& a, const Shape& b) {
    size_t rank = std::max(a.size(), b.size());
    Shape out(rank);
    for (size_t i = 0; i < rank; ++i) {
        int ai = (i < rank - a.size()) ? 1 : a[i - (rank - a.size())];
        int bi = (i < rank - b.size()) ? 1 : b[i - (rank - b.size())];
        if (ai != bi && ai != 1 && bi != 1)
            throw std::runtime_error("broadcast shape mismatch: " + shape_str(a) + " vs " + shape_str(b));
        out[i] = std::max(ai, bi);
    }
    return out;
}

// Per-dim stride of `ishape` expressed in `oshape`'s rank — 0 on any axis
// that's broadcast (missing, or size-1 against a larger oshape dim).
// Computed ONCE per op call, not per element, so the hot per-element loop
// in add()/mul() below is pure integer arithmetic with no heap traffic.
inline std::vector<long long> broadcast_strides(const Shape& ishape, const Shape& oshape) {
    auto istrides = row_major_strides(ishape);
    int rankdiff = (int)oshape.size() - (int)ishape.size();
    std::vector<long long> bs(oshape.size(), 0);
    for (size_t d = 0; d < ishape.size(); ++d)
        if (ishape[d] != 1) bs[d + rankdiff] = istrides[d];
    return bs;
}

struct TensorImpl {
    Shape shape;
    std::vector<double> data;
    std::vector<double> grad;
    bool requires_grad = false;
    std::vector<std::shared_ptr<TensorImpl>> parents;
    std::function<void()> backward_fn; // captures `this` raw ptr (safe: DAG shared_ptrs keep it alive)

    explicit TensorImpl(Shape s) : shape(std::move(s)) {
        data.assign(total_size(shape), 0.0);
        grad.assign(total_size(shape), 0.0);
    }
};

using Tensor = std::shared_ptr<TensorImpl>;

inline Tensor make_tensor(const Shape& shape, bool requires_grad = false) {
    auto t = std::make_shared<TensorImpl>(shape);
    t->requires_grad = requires_grad;
    return t;
}

inline Tensor make_tensor(const Shape& shape, const std::vector<double>& data, bool requires_grad = false) {
    auto t = make_tensor(shape, requires_grad);
    if ((long long)data.size() != total_size(shape))
        throw std::runtime_error("make_tensor: data size mismatch");
    t->data = data;
    return t;
}

inline Tensor zeros(const Shape& shape) { return make_tensor(shape, false); }

// ---- backward pass ----

inline void backward(const Tensor& root, const std::vector<double>& seed) {
    std::vector<TensorImpl*> topo;
    std::unordered_set<TensorImpl*> visited;
    std::function<void(TensorImpl*)> build = [&](TensorImpl* n) {
        if (visited.count(n)) return;
        visited.insert(n);
        for (auto& p : n->parents) build(p.get());
        topo.push_back(n);
    };
    build(root.get());
    for (auto* n : topo) std::fill(n->grad.begin(), n->grad.end(), 0.0);
    root->grad = seed;
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

inline void backward_scalar(const Tensor& root) {
    if (total_size(root->shape) != 1) throw std::runtime_error("backward_scalar: root is not scalar");
    backward(root, std::vector<double>{1.0});
}

// ============================== primitive ops ==============================

inline Tensor add(const Tensor& a, const Tensor& b) {
    if (a->shape == b->shape) { // fast path: no broadcasting at all
        auto out = make_tensor(a->shape, a->requires_grad || b->requires_grad);
        for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = a->data[i] + b->data[i];
        out->parents = {a, b};
        TensorImpl* self = out.get();
        auto ap = a; auto bp = b;
        out->backward_fn = [self, ap, bp]() {
            if (ap->requires_grad) for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i];
            if (bp->requires_grad) for (size_t i = 0; i < bp->grad.size(); ++i) bp->grad[i] += self->grad[i];
        };
        return out;
    }
    Shape oshape = broadcast_shape(a->shape, b->shape);
    auto ostrides = row_major_strides(oshape);
    auto abs_ = broadcast_strides(a->shape, oshape);
    auto bbs = broadcast_strides(b->shape, oshape);
    int rank = (int)oshape.size();
    long long N = total_size(oshape);
    auto out = make_tensor(oshape, a->requires_grad || b->requires_grad);
    for (long long flat = 0; flat < N; ++flat) {
        long long aoff = 0, boff = 0, rem = flat;
        for (int d = 0; d < rank; ++d) {
            long long idx = rem / ostrides[d]; rem %= ostrides[d];
            aoff += idx * abs_[d]; boff += idx * bbs[d];
        }
        out->data[flat] = a->data[aoff] + b->data[boff];
    }
    out->parents = {a, b};
    TensorImpl* self = out.get();
    auto ap = a; auto bp = b;
    out->backward_fn = [self, ap, bp, oshape, ostrides, abs_, bbs, rank]() {
        long long N = total_size(oshape);
        if (ap->requires_grad) {
            for (long long flat = 0; flat < N; ++flat) {
                long long aoff = 0, rem = flat;
                for (int d = 0; d < rank; ++d) { long long idx = rem / ostrides[d]; rem %= ostrides[d]; aoff += idx * abs_[d]; }
                ap->grad[aoff] += self->grad[flat];
            }
        }
        if (bp->requires_grad) {
            for (long long flat = 0; flat < N; ++flat) {
                long long boff = 0, rem = flat;
                for (int d = 0; d < rank; ++d) { long long idx = rem / ostrides[d]; rem %= ostrides[d]; boff += idx * bbs[d]; }
                bp->grad[boff] += self->grad[flat];
            }
        }
    };
    return out;
}

inline Tensor neg(const Tensor& a) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = -a->data[i];
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += -self->grad[i];
    };
    return out;
}

inline Tensor sub(const Tensor& a, const Tensor& b) { return add(a, neg(b)); }

inline Tensor mul(const Tensor& a, const Tensor& b) {
    if (a->shape == b->shape) { // fast path: no broadcasting at all
        auto out = make_tensor(a->shape, a->requires_grad || b->requires_grad);
        for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = a->data[i] * b->data[i];
        out->parents = {a, b};
        TensorImpl* self = out.get();
        auto ap = a; auto bp = b;
        out->backward_fn = [self, ap, bp]() {
            if (ap->requires_grad) for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i] * bp->data[i];
            if (bp->requires_grad) for (size_t i = 0; i < bp->grad.size(); ++i) bp->grad[i] += self->grad[i] * ap->data[i];
        };
        return out;
    }
    Shape oshape = broadcast_shape(a->shape, b->shape);
    auto ostrides = row_major_strides(oshape);
    auto abs_ = broadcast_strides(a->shape, oshape);
    auto bbs = broadcast_strides(b->shape, oshape);
    int rank = (int)oshape.size();
    long long N = total_size(oshape);
    auto out = make_tensor(oshape, a->requires_grad || b->requires_grad);
    for (long long flat = 0; flat < N; ++flat) {
        long long aoff = 0, boff = 0, rem = flat;
        for (int d = 0; d < rank; ++d) {
            long long idx = rem / ostrides[d]; rem %= ostrides[d];
            aoff += idx * abs_[d]; boff += idx * bbs[d];
        }
        out->data[flat] = a->data[aoff] * b->data[boff];
    }
    out->parents = {a, b};
    TensorImpl* self = out.get();
    auto ap = a; auto bp = b;
    out->backward_fn = [self, ap, bp, oshape, ostrides, abs_, bbs, rank]() {
        long long N = total_size(oshape);
        bool ag = ap->requires_grad, bg = bp->requires_grad;
        if (!ag && !bg) return;
        for (long long flat = 0; flat < N; ++flat) {
            long long aoff = 0, boff = 0, rem = flat;
            for (int d = 0; d < rank; ++d) { long long idx = rem / ostrides[d]; rem %= ostrides[d]; aoff += idx * abs_[d]; boff += idx * bbs[d]; }
            if (ag) ap->grad[aoff] += self->grad[flat] * bp->data[boff];
            if (bg) bp->grad[boff] += self->grad[flat] * ap->data[aoff];
        }
    };
    return out;
}

inline Tensor add_scalar(const Tensor& a, double c) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = a->data[i] + c;
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i];
    };
    return out;
}

inline Tensor mul_scalar(const Tensor& a, double c) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = a->data[i] * c;
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap, c]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i] * c;
    };
    return out;
}

inline Tensor square(const Tensor& a) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = a->data[i] * a->data[i];
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i] * 2.0 * ap->data[i];
    };
    return out;
}

inline Tensor sqrt_op(const Tensor& a) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = std::sqrt(a->data[i]);
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i)
                ap->grad[i] += self->grad[i] * 0.5 / self->data[i];
    };
    return out;
}

inline Tensor exp_op(const Tensor& a) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = std::exp(a->data[i]);
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i] * self->data[i];
    };
    return out;
}

// numerically stable softplus: log(1+exp(x)); backward = sigmoid(x)
inline Tensor softplus(const Tensor& a) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) {
        double x = a->data[i];
        out->data[i] = (x > 20.0) ? x : std::log1p(std::exp(x));
    }
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) {
                double sig = 1.0 / (1.0 + std::exp(-ap->data[i]));
                ap->grad[i] += self->grad[i] * sig;
            }
    };
    return out;
}

inline Tensor reshape(const Tensor& a, const Shape& newshape) {
    if (total_size(newshape) != total_size(a->shape))
        throw std::runtime_error("reshape: element count mismatch " + shape_str(a->shape) + " -> " + shape_str(newshape));
    auto out = make_tensor(newshape, a->requires_grad);
    out->data = a->data;
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i) ap->grad[i] += self->grad[i];
    };
    return out;
}

// reduce-sum over one axis; keeps the axis with size 1 (keepdim semantics)
inline Tensor sum_axis(const Tensor& a, int axis) {
    if (axis < 0) axis += (int)a->shape.size();
    Shape oshape = a->shape;
    oshape[axis] = 1;
    auto out = make_tensor(oshape, a->requires_grad);
    auto istrides = row_major_strides(a->shape);
    long long N = total_size(a->shape);
    std::vector<int> idx;
    for (long long i = 0; i < N; ++i) {
        unravel_into(i, a->shape, idx);
        std::vector<int> oidx = idx;
        oidx[axis] = 0;
        long long ooff = 0; auto ostrides = row_major_strides(oshape);
        for (size_t d = 0; d < oshape.size(); ++d) ooff += (long long)oidx[d] * ostrides[d];
        out->data[ooff] += a->data[i];
    }
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    Shape ashape = a->shape;
    out->backward_fn = [self, ap, ashape, axis]() {
        if (!ap->requires_grad) return;
        long long N = total_size(ashape);
        auto ostrides = row_major_strides(self->shape);
        std::vector<int> idx;
        for (long long i = 0; i < N; ++i) {
            unravel_into(i, ashape, idx);
            std::vector<int> oidx = idx;
            oidx[axis] = 0;
            long long ooff = 0;
            for (size_t d = 0; d < self->shape.size(); ++d) ooff += (long long)oidx[d] * ostrides[d];
            ap->grad[i] += self->grad[ooff];
        }
    };
    return out;
}

// extract one index along an axis, reducing rank by 1
inline Tensor slice_axis(const Tensor& a, int axis, int index) {
    if (axis < 0) axis += (int)a->shape.size();
    Shape oshape;
    for (size_t d = 0; d < a->shape.size(); ++d) if ((int)d != axis) oshape.push_back(a->shape[d]);
    auto out = make_tensor(oshape, a->requires_grad);
    auto astrides = row_major_strides(a->shape);
    long long N = total_size(oshape);
    std::vector<int> oidx;
    for (long long i = 0; i < N; ++i) {
        unravel_into(i, oshape, oidx);
        std::vector<int> aidx; int k = 0;
        for (size_t d = 0; d < a->shape.size(); ++d) {
            if ((int)d == axis) aidx.push_back(index);
            else aidx.push_back(oidx[k++]);
        }
        long long aoff = 0;
        for (size_t d = 0; d < a->shape.size(); ++d) aoff += (long long)aidx[d] * astrides[d];
        out->data[i] = a->data[aoff];
    }
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    Shape ashape = a->shape;
    out->backward_fn = [self, ap, ashape, axis, index]() {
        if (!ap->requires_grad) return;
        auto astrides = row_major_strides(ashape);
        long long N = total_size(self->shape);
        std::vector<int> oidx;
        for (long long i = 0; i < N; ++i) {
            unravel_into(i, self->shape, oidx);
            std::vector<int> aidx; int k = 0;
            for (size_t d = 0; d < ashape.size(); ++d) {
                if ((int)d == axis) aidx.push_back(index);
                else aidx.push_back(oidx[k++]);
            }
            long long aoff = 0;
            for (size_t d = 0; d < ashape.size(); ++d) aoff += (long long)aidx[d] * astrides[d];
            ap->grad[aoff] += self->grad[i];
        }
    };
    return out;
}

// extract a contiguous [start:end) range along one axis (rank unchanged).
// Used by the parallel-scan SSM formulation to grab shifted, overlapping
// windows along the time axis without an allocation per element.
inline Tensor slice_range_axis(const Tensor& a, int axis, int start, int end) {
    if (axis < 0) axis += (int)a->shape.size();
    Shape oshape = a->shape;
    oshape[axis] = end - start;
    auto out = make_tensor(oshape, a->requires_grad);
    auto astrides = row_major_strides(a->shape);
    auto ostrides = row_major_strides(oshape);
    int rank = (int)oshape.size();
    if (rank > 8) throw std::runtime_error("slice_range_axis: rank too high for fixed buffer");
    long long N = total_size(oshape);
    for (long long flat = 0; flat < N; ++flat) {
        long long idxbuf[8] = {}, r = flat, aoff = 0;
        for (int d = 0; d < rank; ++d) { idxbuf[d] = r / ostrides[d]; r %= ostrides[d]; }
        for (int d = 0; d < rank; ++d) aoff += (d == axis ? idxbuf[d] + start : idxbuf[d]) * astrides[d];
        out->data[flat] = a->data[aoff];
    }
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap, astrides, ostrides, axis, start, rank]() {
        if (!ap->requires_grad) return;
        long long N = total_size(self->shape);
        for (long long flat = 0; flat < N; ++flat) {
            long long idxbuf[8] = {}, r = flat, aoff = 0;
            for (int d = 0; d < rank; ++d) { idxbuf[d] = r / ostrides[d]; r %= ostrides[d]; }
            for (int d = 0; d < rank; ++d) aoff += (d == axis ? idxbuf[d] + start : idxbuf[d]) * astrides[d];
            ap->grad[aoff] += self->grad[flat];
        }
    };
    return out;
}

// concatenate along an arbitrary axis (concat_last_dim below stays a
// specialized fast path for its own case; this one is the general form
// the scan needs for axis=1 / the time axis).
inline Tensor concat_axis(const Tensor& a, const Tensor& b, int axis) {
    if (axis < 0) axis += (int)a->shape.size();
    if (a->shape.size() != b->shape.size()) throw std::runtime_error("concat_axis: rank mismatch");
    Shape oshape = a->shape;
    int aw = a->shape[axis];
    oshape[axis] = aw + b->shape[axis];
    auto out = make_tensor(oshape, a->requires_grad || b->requires_grad);
    auto ostrides = row_major_strides(oshape);
    auto astrides = row_major_strides(a->shape);
    auto bstrides = row_major_strides(b->shape);
    int rank = (int)oshape.size();
    if (rank > 8) throw std::runtime_error("concat_axis: rank too high for fixed buffer");
    long long N = total_size(oshape);
    for (long long flat = 0; flat < N; ++flat) {
        long long idxbuf[8] = {}, r = flat;
        for (int d = 0; d < rank; ++d) { idxbuf[d] = r / ostrides[d]; r %= ostrides[d]; }
        bool fromA = idxbuf[axis] < aw;
        long long off = 0;
        for (int d = 0; d < rank; ++d) {
            long long v = (d == axis) ? (fromA ? idxbuf[d] : idxbuf[d] - aw) : idxbuf[d];
            off += v * (fromA ? astrides[d] : bstrides[d]);
        }
        out->data[flat] = fromA ? a->data[off] : b->data[off];
    }
    out->parents = {a, b};
    TensorImpl* self = out.get();
    auto ap = a; auto bp = b;
    out->backward_fn = [self, ap, bp, ostrides, astrides, bstrides, axis, aw, rank]() {
        long long N = total_size(self->shape);
        for (long long flat = 0; flat < N; ++flat) {
            long long idxbuf[8] = {}, r = flat;
            for (int d = 0; d < rank; ++d) { idxbuf[d] = r / ostrides[d]; r %= ostrides[d]; }
            bool fromA = idxbuf[axis] < aw;
            if ((fromA && !ap->requires_grad) || (!fromA && !bp->requires_grad)) continue;
            long long off = 0;
            for (int d = 0; d < rank; ++d) {
                long long v = (d == axis) ? (fromA ? idxbuf[d] : idxbuf[d] - aw) : idxbuf[d];
                off += v * (fromA ? astrides[d] : bstrides[d]);
            }
            if (fromA) ap->grad[off] += self->grad[flat];
            else bp->grad[off] += self->grad[flat];
        }
    };
    return out;
}

// stack a list of same-shaped tensors along a NEW axis (inserted at `axis`)
inline Tensor stack_new_axis(const std::vector<Tensor>& xs, int axis) {
    Shape ishape = xs[0]->shape;
    if (axis < 0) axis += (int)ishape.size() + 1;
    Shape oshape = ishape;
    oshape.insert(oshape.begin() + axis, (int)xs.size());
    bool rg = false;
    for (auto& x : xs) rg = rg || x->requires_grad;
    auto out = make_tensor(oshape, rg);
    auto ostrides = row_major_strides(oshape);
    for (size_t k = 0; k < xs.size(); ++k) {
        long long M = total_size(ishape);
        std::vector<int> iidx;
        for (long long i = 0; i < M; ++i) {
            unravel_into(i, ishape, iidx);
            std::vector<int> oidx; int d2 = 0;
            for (size_t d = 0; d < oshape.size(); ++d) {
                if ((int)d == axis) oidx.push_back((int)k);
                else oidx.push_back(iidx[d2++]);
            }
            long long ooff = 0;
            for (size_t d = 0; d < oshape.size(); ++d) ooff += (long long)oidx[d] * ostrides[d];
            out->data[ooff] = xs[k]->data[i];
        }
    }
    out->parents = xs;
    TensorImpl* self = out.get();
    std::vector<Tensor> xsCopy = xs;
    out->backward_fn = [self, xsCopy, ishape, axis]() {
        auto ostrides = row_major_strides(self->shape);
        for (size_t k = 0; k < xsCopy.size(); ++k) {
            if (!xsCopy[k]->requires_grad) continue;
            long long M = total_size(ishape);
            std::vector<int> iidx;
            for (long long i = 0; i < M; ++i) {
                unravel_into(i, ishape, iidx);
                std::vector<int> oidx; int d2 = 0;
                for (size_t d = 0; d < self->shape.size(); ++d) {
                    if ((int)d == axis) oidx.push_back((int)k);
                    else oidx.push_back(iidx[d2++]);
                }
                long long ooff = 0;
                for (size_t d = 0; d < self->shape.size(); ++d) ooff += (long long)oidx[d] * ostrides[d];
                xsCopy[k]->grad[i] += self->grad[ooff];
            }
        }
    };
    return out;
}

// concatenate two tensors along the last axis
inline Tensor concat_last_dim(const Tensor& a, const Tensor& b) {
    if (a->shape.size() != b->shape.size()) throw std::runtime_error("concat_last_dim: rank mismatch");
    Shape oshape = a->shape;
    int last = (int)oshape.size() - 1;
    oshape[last] = a->shape[last] + b->shape[last];
    auto out = make_tensor(oshape, a->requires_grad || b->requires_grad);
    long long rows = total_size(a->shape) / a->shape[last];
    int aw = a->shape[last], bw = b->shape[last];
    for (long long r = 0; r < rows; ++r) {
        for (int c = 0; c < aw; ++c) out->data[r * (aw + bw) + c] = a->data[r * aw + c];
        for (int c = 0; c < bw; ++c) out->data[r * (aw + bw) + aw + c] = b->data[r * bw + c];
    }
    out->parents = {a, b};
    TensorImpl* self = out.get();
    auto ap = a; auto bp = b;
    out->backward_fn = [self, ap, bp, aw, bw, rows]() {
        for (long long r = 0; r < rows; ++r) {
            if (ap->requires_grad)
                for (int c = 0; c < aw; ++c) ap->grad[r * aw + c] += self->grad[r * (aw + bw) + c];
            if (bp->requires_grad)
                for (int c = 0; c < bw; ++c) bp->grad[r * bw + c] += self->grad[r * (aw + bw) + aw + c];
        }
    };
    return out;
}

// x: (..., in) flattened to (N,in); W: (out,in); b: (out) -> (..., out)
inline Tensor linear(const Tensor& x, const Tensor& W, const Tensor& b) {
    Shape xshape = x->shape;
    int in = xshape.back();
    long long N = total_size(xshape) / in;
    int out_f = W->shape[0];
    Shape oshape = xshape; oshape.back() = out_f;
    auto out = make_tensor(oshape, x->requires_grad || W->requires_grad || b->requires_grad);
    for (long long n = 0; n < N; ++n) {
        for (int o = 0; o < out_f; ++o) {
            double acc = b->data[o];
            for (int i = 0; i < in; ++i) acc += x->data[n * in + i] * W->data[o * in + i];
            out->data[n * out_f + o] = acc;
        }
    }
    out->parents = {x, W, b};
    TensorImpl* self = out.get();
    auto xp = x; auto Wp = W; auto bp = b;
    out->backward_fn = [self, xp, Wp, bp, N, in, out_f]() {
        if (xp->requires_grad) {
            for (long long n = 0; n < N; ++n)
                for (int i = 0; i < in; ++i) {
                    double acc = 0;
                    for (int o = 0; o < out_f; ++o) acc += self->grad[n * out_f + o] * Wp->data[o * in + i];
                    xp->grad[n * in + i] += acc;
                }
        }
        if (Wp->requires_grad) {
            for (int o = 0; o < out_f; ++o)
                for (int i = 0; i < in; ++i) {
                    double acc = 0;
                    for (long long n = 0; n < N; ++n) acc += self->grad[n * out_f + o] * xp->data[n * in + i];
                    Wp->grad[o * in + i] += acc;
                }
        }
        if (bp->requires_grad) {
            for (int o = 0; o < out_f; ++o) {
                double acc = 0;
                for (long long n = 0; n < N; ++n) acc += self->grad[n * out_f + o];
                bp->grad[o] += acc;
            }
        }
    };
    return out;
}

// gather rows of `table` (V,D) at `indices` (flat, length N) -> (N,D)
inline Tensor embedding(const std::vector<int>& indices, const Tensor& table) {
    int D = table->shape[1];
    long long N = (long long)indices.size();
    auto out = make_tensor({(int)N, D}, table->requires_grad);
    for (long long n = 0; n < N; ++n)
        for (int d = 0; d < D; ++d)
            out->data[n * D + d] = table->data[(long long)indices[n] * D + d];
    out->parents = {table};
    TensorImpl* self = out.get();
    auto tp = table;
    out->backward_fn = [self, tp, indices, D, N]() {
        if (!tp->requires_grad) return;
        for (long long n = 0; n < N; ++n)
            for (int d = 0; d < D; ++d)
                tp->grad[(long long)indices[n] * D + d] += self->grad[n * D + d];
    };
    return out;
}

inline Tensor reciprocal(const Tensor& a) {
    auto out = make_tensor(a->shape, a->requires_grad);
    for (size_t i = 0; i < a->data.size(); ++i) out->data[i] = 1.0 / a->data[i];
    out->parents = {a};
    TensorImpl* self = out.get();
    auto ap = a;
    out->backward_fn = [self, ap]() {
        if (ap->requires_grad)
            for (size_t i = 0; i < ap->grad.size(); ++i)
                ap->grad[i] += self->grad[i] * (-1.0 / (ap->data[i] * ap->data[i]));
    };
    return out;
}

// LayerNorm over the last dimension, composed from primitives above.
inline Tensor layer_norm(const Tensor& x, const Tensor& gamma, const Tensor& beta, double eps = 1e-5) {
    int last = (int)x->shape.size() - 1;
    int D = x->shape[last];
    Tensor mean = mul_scalar(sum_axis(x, last), 1.0 / D);   // (...,1)
    Tensor centered = sub(x, mean);                          // broadcasts (...,1) -> (...,D)
    Tensor var = mul_scalar(sum_axis(square(centered), last), 1.0 / D); // (...,1)
    Tensor std_ = sqrt_op(add_scalar(var, eps));              // (...,1)
    Tensor normed = mul(centered, reciprocal(std_));
    return add(mul(normed, gamma), beta);
}

// mean cross-entropy over logits (N,C) with integer targets (length N)
inline Tensor cross_entropy(const Tensor& logits, const std::vector<int>& targets) {
    int N = logits->shape[0], C = logits->shape[1];
    auto out = make_tensor({1}, logits->requires_grad);
    std::vector<double> probs(logits->data.size());
    double total = 0.0;
    for (int n = 0; n < N; ++n) {
        double mx = -1e300;
        for (int c = 0; c < C; ++c) mx = std::max(mx, logits->data[n * C + c]);
        double sum = 0.0;
        for (int c = 0; c < C; ++c) { probs[n * C + c] = std::exp(logits->data[n * C + c] - mx); sum += probs[n * C + c]; }
        for (int c = 0; c < C; ++c) probs[n * C + c] /= sum;
        total += -std::log(std::max(probs[n * C + targets[n]], 1e-300));
    }
    out->data[0] = total / N;
    out->parents = {logits};
    TensorImpl* self = out.get();
    auto lp = logits;
    out->backward_fn = [self, lp, probs, targets, N, C]() {
        if (!lp->requires_grad) return;
        double g = self->grad[0] / N;
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < C; ++c) {
                double p = probs[n * C + c] - (c == targets[n] ? 1.0 : 0.0);
                lp->grad[n * C + c] += g * p;
            }
    };
    return out;
}

// ---- inference-time helpers (no autodiff graph, plain data) ----

inline std::vector<double> softmax_raw(const std::vector<double>& logits) {
    double mx = *std::max_element(logits.begin(), logits.end());
    std::vector<double> out(logits.size());
    double sum = 0;
    for (size_t i = 0; i < logits.size(); ++i) { out[i] = std::exp(logits[i] - mx); sum += out[i]; }
    for (auto& v : out) v /= sum;
    return out;
}

} // namespace ag
