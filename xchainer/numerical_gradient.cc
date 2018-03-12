#include "xchainer/numerical_gradient.h"

#include <algorithm>
#include <functional>
#include <vector>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA

#include "xchainer/array.h"
#include "xchainer/array_repr.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"

namespace xchainer {
namespace numerical_gradient_internal {

Array& Subtract(const Array& lhs, const Array& rhs, Array& out) {
    lhs.device().Synchronize();
    rhs.device().Synchronize();
    out.device().Synchronize();

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); i++) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] - rhs_iarray[indexer];
        }
    });
    return out;
}

Array& Divide(const Array& lhs, const Array& rhs, Array& out) {
    lhs.device().Synchronize();
    rhs.device().Synchronize();
    out.device().Synchronize();

    VisitDtype(lhs.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> lhs_iarray{lhs};
        IndexableArray<const T> rhs_iarray{rhs};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{lhs.shape()};

        for (int64_t i = 0; i < indexer.total_size(); i++) {
            indexer.Set(i);
            out_iarray[indexer] = lhs_iarray[indexer] / rhs_iarray[indexer];
        }
    });
    return out;
}

Array operator-(const Array& lhs, const Array& rhs) {
    Array out = Array::EmptyLike(lhs);
    Subtract(lhs, rhs, out);
    return out;
}

Array operator/(const Array& lhs, const Array& rhs) {
    Array out = Array::EmptyLike(lhs);
    Divide(lhs, rhs, out);
    return out;
}

Scalar Sum(const Array& array) {
    array.device().Synchronize();

    return VisitDtype(array.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> iarray{array};
        Indexer<> indexer{array.shape()};

        T s = 0;
        for (int64_t i = 0; i < indexer.total_size(); i++) {
            indexer.Set(i);
            s += iarray[indexer];
        }
        return Scalar{s};
    });
}

Scalar Norm(const Array& x) {
    Scalar s = Sum(x * x);
    return Scalar(std::sqrt(static_cast<double>(s)), x.dtype());
}

Scalar VectorDot(const Array& x, const Array& y) { return Sum(x * y); }

void Set(Array& out, int64_t flat_index, Scalar value) {
    out.device().Synchronize();

    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<T> iarray{out};
        Indexer<> indexer{out.shape()};
        indexer.Set(flat_index);
        iarray[indexer] = static_cast<T>(value);
    });
}

Scalar Get(const Array& out, int64_t flat_index) {
    out.device().Synchronize();

    return VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> iarray{out};
        Indexer<> indexer{out.shape()};
        indexer.Set(flat_index);
        return Scalar{iarray[indexer]};
    });
}

Arrays CalculateNumericalGradient(
        std::function<Arrays(const Arrays&)> func,
        const Arrays& inputs,
        const Arrays& grad_outputs,
        const Arrays& eps,
        const GraphId& graph_id) {
    // TODO(niboshi): Currently only elementwise functions are supported.
    // TODO(niboshi): Implement arithmetic operations and avoid manual synchronize
    const int nin = inputs.size();
    const int nout = grad_outputs.size();

    if (eps.size() != static_cast<size_t>(nin)) {
        throw XchainerError(
                "Invalid number of eps arrays where number of inputs: " + std::to_string(nin) + ", eps: " + std::to_string(eps.size()));
    }

    for (int i = 0; i < nin; ++i) {
        if (inputs.at(i).shape() != eps.at(i).shape()) {
            throw XchainerError("Invalid eps shape");
        }
        if (inputs.at(i).dtype() != eps.at(i).dtype()) {
            throw XchainerError("Invalid eps dtype");
        }
        // TODO(niboshi): Check: eps must not contain zeros.
    }

    Dtype dtype = inputs[0].dtype();

    auto eval = [&, graph_id](int i_in, int64_t in_flat_index, Scalar eps_scalar, float multiplier) -> Arrays {
        Arrays xs;
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(xs), [graph_id](const Array& x) {
            return x.AsConstant(CopyKind::kCopy).RequireGrad(graph_id);
        });

        Set(xs.at(i_in), in_flat_index, Get(xs.at(i_in), in_flat_index) + Scalar(static_cast<float>(eps_scalar) * multiplier, dtype));
        return func(xs);
    };

    Arrays grads;

    for (int i = 0; i < nin; ++i) {
        Array grad_i = Array::ZerosLike(inputs.at(i));
        int64_t size = grad_i.GetTotalSize();

        for (int64_t in_flat_index = 0; in_flat_index < size; ++in_flat_index) {
            Scalar eps_scalar = Get(eps.at(i), in_flat_index);
            Arrays ys0 = eval(i, in_flat_index, eps_scalar, -1);
            Arrays ys1 = eval(i, in_flat_index, eps_scalar, 1);

            for (int j = 0; j < nout; ++j) {
                Array dy = ys1.at(j) - ys0.at(j);
                Array denom = Array::FullLike(dy, eps_scalar) * Array::FullLike(dy, Scalar(2, dtype));

                Scalar g = VectorDot((ys1.at(j) - ys0.at(j)) / denom, grad_outputs.at(j));
                Scalar g_ij = Get(grad_i, in_flat_index) + g;
                Set(grad_i, in_flat_index, g_ij);
            }
        }
        grads.push_back(grad_i);
    }

    return grads;
}

}  // namespace numerical_gradient_internal
}  // namespace xchainer
