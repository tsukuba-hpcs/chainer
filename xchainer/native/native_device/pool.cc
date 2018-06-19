#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
#include "xchainer/macro.h"
#include "xchainer/native/col2im.h"
#include "xchainer/native/elementwise.h"
#include "xchainer/native/im2col.h"
#include "xchainer/native/tensor_dot.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
#include "xchainer/routines/math.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace native {
namespace {

Scalar GetLowestOrInf(Dtype dtype) {
    return VisitDtype(dtype, [](auto pt) {
        using T = typename decltype(pt)::type;
        return Scalar{NumericLimits<T>::LowestOrInf()};
    });
}

// Returns axes that does the following transpose.
// (batch_size, channel, a_1, a_2, ...., a_n, b_1, b_2, ..., b_n) -> (batch_size, channel, b_1, b_2, ...., b_n, a_1, a_2, ..., a_n).
Axes GetSwapSpatialDimensionsAxes(size_t n) {
    Axes axes;
    axes.resize(2 + 2 * n);  // E.g. (batch_size, channel, out_1, out_2, ..., out_n, k_1, k_2, ..., k_n).
    axes[0] = 0;  // Batch dimension kept as is.
    axes[1] = 1;  // Channel dimension kept as is.
    for (size_t i = 2; i < n + 2; ++i) {  // Output and kernel spatial dimensions to be swapped.
        axes[i] = n + i;
        axes[n + i] = i;
    }
    return axes;
}

class NativeMaxPoolForwardBackward : public xchainer::MaxPoolForwardBackward {
public:
    Array Forward(
            const Array& x,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override {
        // Convert to column representation of shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
        col_ = internal::Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all, GetLowestOrInf(x.dtype()));
        axes_.resize(kernel_size.size());
        std::iota(axes_.begin(), axes_.end(), 2);
        return col_.Max(axes_);
    }

    Array Backward(
            const Array& x,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool /*cover_all*/,
            const Array& gout) override {
        indices_ = col_.ArgMax(axes_);
        assert(indices_.shape() == gout.shape());

        // Compute flattened col gradients.
        int64_t kernel_total_size = std::accumulate(kernel_size.begin(), kernel_size.end(), int64_t{1}, std::multiplies<>());
        int64_t out_total_size = indices_.GetTotalSize();
        Shape out_flat{out_total_size};
        Device& device = x.device();
        Array gcol = Zeros({out_total_size * kernel_total_size}, x.dtype(), device);
        offset_ = Arange(0, out_total_size * kernel_total_size, kernel_total_size, indices_.dtype(), device);
        device.AddAt(gcol, indices_.Reshape(out_flat) + offset_, {0}, gout.AsConstant().Reshape(out_flat), gcol);

        // Reshape col gradients to (batch_size, channel, out_1, out_2, ..., out_n, k_1, k_2, ..., k_n).
        Shape out_shape_with_kernel = gout.shape();
        std::copy(kernel_size.begin(), kernel_size.end(), std::back_inserter(out_shape_with_kernel));

        // Transform col gradients to input shape.
        return internal::Col2Im(
                gcol.Reshape(out_shape_with_kernel).Transpose(GetSwapSpatialDimensionsAxes(kernel_size.size())),
                stride,
                pad,
                {x.shape().begin() + 2, x.shape().end()});
    }

    Array DoubleBackward(
            const Array& x,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            const Array& /*gout*/,
            const Array& ggx) override {
        Array col = internal::Im2Col(ggx.AsConstant(), kernel_size, stride, pad, cover_all, GetLowestOrInf(x.dtype()));
        return Take(
                col.Transpose(GetSwapSpatialDimensionsAxes(kernel_size.size())).Reshape({col.GetTotalSize()}),
                indices_ + offset_.Reshape(indices_.shape()),
                0);
    }

private:
    Array col_{};
    Axes axes_{};
    Array indices_{};
    Array offset_{};
};

}  // namespace

std::unique_ptr<MaxPoolForwardBackward> NativeDevice::GetMaxPoolForwardBackward() {
    return std::make_unique<NativeMaxPoolForwardBackward>();
}

namespace {

void Mean(const Array& a, const Axes& axis, const Array& out) {
    Device& device = a.device();
    device.Sum(a, axis, out);
    device.DivideAS(out, xchainer::internal::CountItemsAlongAxes(a.shape(), axis), out);
}

template <typename T, AveragePoolMode kAveragePoolMode>
struct GetPoolingWidthsImpl {
    void operator()(int64_t i, T& width) {
        T start = i * stride - pad;
        T end = start + kernel_size;
        switch (kAveragePoolMode) {
            case AveragePoolMode::kZero:
                // Do nothing.
                break;
            case AveragePoolMode::kIgnore: {
                if (start < 0) {
                    start = 0;
                }
                if (end > dim) {
                    end = dim;
                }
                break;
            }
            default:
                XCHAINER_NEVER_REACH();
        }
        width = end - start;
    }

    T dim;
    T kernel_size;
    T stride;
    T pad;
};

template <AveragePoolMode kAveragePoolMode>
Array GetPoolingWidths(
        const Shape& shape,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        Dtype dtype) {
    int8_t n = shape.ndim() - 2;
    assert(n == static_cast<int8_t>(kernel_size.size()));
    assert(n == static_cast<int8_t>(stride.size()));
    assert(n == static_cast<int8_t>(pad.size()));

    Array widths;
    for (int64_t i = 0; i < n; ++i) {
        int64_t dim = shape[2 + i];
        int64_t k = kernel_size[i];
        int64_t s = stride[i];
        int64_t p = pad[i];

        Array width = Empty({xchainer::internal::GetConvOutDim(dim, k, s, p, cover_all)}, dtype);
        VisitDtype(dtype, [&](auto pt) {
            using T = typename decltype(pt)::type;
            Elementwise<T>(
                    GetPoolingWidthsImpl<T, kAveragePoolMode>{static_cast<T>(dim), static_cast<T>(k), static_cast<T>(s), static_cast<T>(p)},
                    width);
        });

        if (i == 0) {
            widths = width;
        } else {
            Shape widths_expanded = widths.shape();
            widths_expanded.emplace_back(1);

            Shape width_expanded{1};
            std::copy(width.shape().begin(), width.shape().end(), std::back_inserter(width_expanded));

            widths = TensorDot(widths.Reshape(widths_expanded), width.Reshape(width_expanded), {static_cast<int8_t>(widths.ndim())}, {0});
        }
    }
    return widths;
}

}  // namespace

Array NativeDevice::AveragePool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all,
        AveragePoolMode average_pool_mode) {
    // TODO(hvy): Support cover_all.
    if (cover_all) {
        throw NotImplementedError{"Native average pooling does not yet support cover_all."};
    }

    Array col = internal::Im2Col(x.AsConstant(), kernel_size, stride, pad, cover_all, 0);

    // Average along the kernel dimensions of col with shape (batch_size, channel, k_1, k_2, ..., k_n, out_1, out_2, ..., out_n).
    Axes kernel_axes;
    kernel_axes.resize(kernel_size.size());
    std::iota(kernel_axes.begin(), kernel_axes.end(), 2);  // From k_1, up to k_n.

    Array out = xchainer::internal::EmptyReduced(col.shape(), col.dtype(), kernel_axes, false, col.device());

    switch (average_pool_mode) {
        case AveragePoolMode::kZero:
            Mean(col, kernel_axes, out);
            break;
        case AveragePoolMode::kIgnore: {
            Device& device = x.device();
            device.Sum(col, kernel_axes, out);
            const Array widths = GetPoolingWidths<AveragePoolMode::kIgnore>(x.shape(), kernel_size, stride, pad, cover_all, x.dtype())
                                         .BroadcastTo(out.shape());
            device.Divide(out, widths, out);
            break;
        }
        default:
            XCHAINER_NEVER_REACH();
    }
    return out;
}

}  // namespace native
}  // namespace xchainer
