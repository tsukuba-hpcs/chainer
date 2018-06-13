#include "xchainer/native/native_device.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/dtype.h"
#include "xchainer/native/col2im.h"
#include "xchainer/native/im2col.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/indexing.h"
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

}  // namespace

Array NativeDevice::AveragePool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
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
    Mean(col, kernel_axes, out);
    return out;
}

}  // namespace native
}  // namespace xchainer
