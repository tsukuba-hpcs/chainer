#include "xchainer/routines/connection.h"

#include <cstdint>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backprop_mode.h"
#include "xchainer/backward_builder.h"
#include "xchainer/backward_context.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/graph.h"
#include "xchainer/routines/math.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace internal {

int64_t GetConvOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return (in_dim + pad * 2 - kernel_size + stride - 1) / stride + 1;
    }
    return (in_dim + pad * 2 - kernel_size) / stride + 1;
}

int64_t GetConvTransposeOutDim(int64_t in_dim, int64_t kernel_size, int64_t stride, int64_t pad, bool cover_all) {
    if (cover_all) {
        return stride * (in_dim - 1) + kernel_size - stride + 1 - 2 * pad;
    }
    return stride * (in_dim - 1) + kernel_size - 2 * pad;
}

}  // namespace internal

namespace {

Array ConvGradW(
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    assert(w_shape.ndim() > 2);
    assert(x.ndim() == w_shape.ndim());
    assert(gy.ndim() == w_shape.ndim());
    assert(stride.size() == static_cast<size_t>(w_shape.ndim() - 2));
    assert(pad.size() == static_cast<size_t>(w_shape.ndim() - 2));

    Array out{};
    {
        NoBackpropModeScope scope{};
        out = x.device().ConvGradWeight(w_dtype, w_shape, x, gy, stride, pad, cover_all);
    }

    {
        BackwardBuilder bb{"conv-grad-weight", {x, gy}, out};

        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([ x_shape = x.shape(), gy, stride, pad ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                assert(out_size.size() == stride.size());
                bctx.input_grad() = ConvTranspose(gy, gout, nonstd::nullopt, stride, pad, out_size);
            });
        }

        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([x, stride, pad, cover_all](BackwardContext& bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = Conv(x, gout, nonstd::nullopt, stride, pad, cover_all);
            });
        }
        assert(bb.is_complete());
    }

    return out;
}

void ConvCheckNdim(
        const Array& x, const Array& w, const StackVector<int64_t, kMaxNdim>& stride, const StackVector<int64_t, kMaxNdim>& pad) {
    if (w.ndim() != x.ndim()) {
        throw DimensionError{"Mismatched number of dimensions between input ", x.ndim(), " and weights ", w.ndim(), "."};
    }
    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 0) {
        throw DimensionError{"Number of spatial dimensions must be greater than or equal to 0"};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }
}

}  // namespace

Array Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    ConvCheckNdim(x, w, stride, pad);
    if (w.shape()[1] != x.shape()[1]) {
        throw DimensionError{"Mismatched number of input channels in input ", x.shape(), " and weights ", w.shape(), "."};
    }
    if (b.has_value() && (b->ndim() != 1 || b->shape()[0] != w.shape()[0])) {
        throw DimensionError{"Mismatched bias shape ", b->shape(), " for weights ", w.shape(), "."};
    }

    Array out{};
    {
        NoBackpropModeScope scope{};
        out = x.device().Conv(x, w, b, stride, pad, cover_all);
    }

    {
        // TODO(niboshi): Improve interface of BackwardBuilder for accepting optional input arrays.
        std::vector<ConstArrayRef> inputs{};
        if (b.has_value()) {
            inputs = {x, w, *b};
        } else {
            inputs = {x, w};
        }
        BackwardBuilder bb{"conv", std::move(inputs), out};

        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([ x_shape = x.shape(), w, stride, pad ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                bctx.input_grad() = ConvTranspose(gout, w, nonstd::nullopt, stride, pad, out_size);
            });
        }

        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([ w_dtype = w.dtype(), w_shape = w.shape(), x, stride, pad, cover_all ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = ConvGradW(w_dtype, w_shape, x, gout, stride, pad, cover_all);
            });
        }

        if (b.has_value()) {
            if (BackwardBuilder::Target bt = bb.CreateTarget(2)) {
                bt.Define([](BackwardContext& bctx) {
                    const Array& gout = bctx.output_grad();
                    Axes axis{0};
                    for (int8_t i = 2; i < gout.ndim(); ++i) {
                        axis.emplace_back(int64_t{i});
                    }
                    bctx.input_grad() = Sum(gout, axis, false);
                });
            }
        }
        assert(bb.is_complete());
    }

    return out;
}

Array ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size) {
    ConvCheckNdim(x, w, stride, pad);
    if (x.shape()[1] != w.shape()[0]) {
        throw DimensionError{"Mismatched number of input channels in input ", x.shape(), " and weights ", w.shape(), "."};
    }
    if (b.has_value() && (b->ndim() != 1 || b->shape()[0] != w.shape()[1])) {
        throw DimensionError{"Mismatched bias shape ", b->shape(), " for weights ", w.shape(), "."};
    }
    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    Shape in_dims{x.shape().begin() + 2, x.shape().end()};
    Shape kernel_size{w.shape().begin() + 2, w.shape().end()};

    bool cover_all = false;

    // Compute out_size if not specified
    StackVector<int64_t, kMaxNdim> real_out_size;
    if (out_size.has_value()) {
        real_out_size = *out_size;

        for (int64_t size : real_out_size) {
            if (size < 0) {
                throw DimensionError{"All output sizes must be non-negative."};
            }
        }

        // Detect cover_all from out_size
        for (int8_t i = 0; i < ndim; ++i) {
            if (in_dims[i] != internal::GetConvOutDim(real_out_size[i], kernel_size[i], stride[i], pad[i], false)) {
                cover_all = true;
                break;
            }
        }
    } else {
        // cover_all is assumed to be false.
        for (int8_t i = 0; i < ndim; ++i) {
            int64_t out_dim = internal::GetConvTransposeOutDim(in_dims[i], kernel_size[i], stride[i], pad[i], cover_all);
            if (out_dim < 0) {
                throw DimensionError{"Inconsistent dimensions. Output dimension at axis ", i, " would be negative."};
            }
            real_out_size.emplace_back(out_dim);
        }
    }

    // Check out_size and cover_all are consistent
    for (int8_t i = 0; i < ndim; ++i) {
        if (in_dims[i] != internal::GetConvOutDim(real_out_size[i], kernel_size[i], stride[i], pad[i], cover_all)) {
            throw DimensionError{"Output dims ", Shape{real_out_size.begin(), real_out_size.end()}, " are incosistent."};
        }
    }

    // Compute transposed convolution
    Array out{};
    {
        NoBackpropModeScope scope{};
        out = x.device().ConvTranspose(x, w, b, stride, pad, real_out_size);
    }

    {
        // TODO(niboshi): Improve interface of BackwardBuilder for accepting optional input arrays.
        std::vector<ConstArrayRef> inputs{};
        if (b.has_value()) {
            inputs = {x, w, *b};
        } else {
            inputs = {x, w};
        }
        BackwardBuilder bb{"conv_transpose", std::move(inputs), out};

        if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
            bt.Define([ x_shape = x.shape(), w, stride, pad, cover_all ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                StackVector<int64_t, kMaxNdim> out_size{x_shape.begin() + 2, x_shape.end()};
                bctx.input_grad() = Conv(gout, w, nonstd::nullopt, stride, pad, cover_all);
            });
        }

        if (BackwardBuilder::Target bt = bb.CreateTarget(1)) {
            bt.Define([ w_dtype = w.dtype(), w_shape = w.shape(), x, stride, pad, cover_all ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                bctx.input_grad() = ConvGradW(w_dtype, w_shape, gout, x, stride, pad, cover_all);
            });
        }

        if (b.has_value()) {
            if (BackwardBuilder::Target bt = bb.CreateTarget(2)) {
                bt.Define([](BackwardContext& bctx) {
                    const Array& gout = bctx.output_grad();
                    Axes axis{0};
                    for (int8_t i = 2; i < gout.ndim(); ++i) {
                        axis.emplace_back(int64_t{i});
                    }
                    bctx.input_grad() = Sum(gout, axis, false);
                });
            }
        }
        assert(bb.is_complete());
    }

    return out;
}

}  // namespace xchainer
