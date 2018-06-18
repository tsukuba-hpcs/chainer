#include "xchainer/cuda/cuda_device.h"

#include <cstdint>
#include <memory>

#include "xchainer/array.h"
#include "xchainer/backend_util.h"
#include "xchainer/constant.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {
namespace {

class CudaMaxPoolForwardBackward : public xchainer::MaxPoolForwardBackward {
public:
    explicit CudaMaxPoolForwardBackward(cudnnHandle_t cudnn_handle) : cudnn_handle_{cudnn_handle} {}

    Array Forward(
            const Array& x,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override {
        if (cover_all) {
            throw XchainerError{"CUDA pooling does not support cover_all"};
        }
        int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
        if (ndim < 2) {
            throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
        }

        assert(kernel_size.size() == static_cast<size_t>(ndim));
        assert(stride.size() == static_cast<size_t>(ndim));
        assert(pad.size() == static_cast<size_t>(ndim));

        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        Shape out_shape{x.shape()[0], x.shape()[1]};
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], kernel_size[i], stride[i], pad[i], cover_all));
            assert(out_shape.back() > 0);
        }

        Array y = Empty(out_shape, x.dtype(), x.device());
        Array x_cont = AsContiguousArray(x);

        internal::TensorDescriptor x_desc{x_cont};
        internal::TensorDescriptor y_desc{y};

        internal::PoolingDescriptor pool_desc{CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, kernel_size, pad, stride};

        CheckCudnnError(cudnnPoolingForward(
                cudnn_handle_,
                *pool_desc,
                internal::GetValuePtr<1>(x.dtype()),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                internal::GetValuePtr<0>(x.dtype()),
                *y_desc,
                xchainer::internal::GetRawOffsetData<void>(y)));

        y_ = y.AsConstant();
        return y;
    }

    Array Backward(
            const Array& x,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all,
            const Array& gout) override {
        if (cover_all) {
            throw XchainerError{"CUDA pooling does not support cover_all"};
        }
        int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
        if (ndim < 2) {
            throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
        }

        assert(kernel_size.size() == static_cast<size_t>(ndim));
        assert(stride.size() == static_cast<size_t>(ndim));
        assert(pad.size() == static_cast<size_t>(ndim));
        assert(gout.shape() == y_.shape());

        Array gx = EmptyLike(x, x.device());
        Array y_cont = AsContiguousArray(y_);
        Array gout_cont = AsContiguousArray(gout);
        Array x_cont = AsContiguousArray(x);

        internal::TensorDescriptor y_desc{y_cont};
        internal::TensorDescriptor gout_desc{gout_cont};
        internal::TensorDescriptor x_desc{x_cont};
        internal::TensorDescriptor gx_desc{gx};

        internal::PoolingDescriptor pool_desc{CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, kernel_size, pad, stride};

        CheckCudnnError(cudnnPoolingBackward(
                cudnn_handle_,
                *pool_desc,
                internal::GetValuePtr<1>(x.dtype()),
                *y_desc,
                xchainer::internal::GetRawOffsetData<void>(y_cont),
                *gout_desc,
                xchainer::internal::GetRawOffsetData<void>(gout_cont),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                internal::GetValuePtr<0>(x.dtype()),
                *gx_desc,
                xchainer::internal::GetRawOffsetData<void>(gx)));

        return gx;
    }

    // TODO(hvy): Implement me.
    Array DoubleBackward(
            const Array& /*x*/,
            const StackVector<int64_t, kMaxNdim>& /*kernel_size*/,
            const StackVector<int64_t, kMaxNdim>& /*stride*/,
            const StackVector<int64_t, kMaxNdim>& /*pad*/,
            bool /*cover_all*/,
            const Array& /*gout*/,
            const Array& /*ggx*/) override {
        return Array{};
    }

private:
    cudnnHandle_t cudnn_handle_;
    Array y_;
};

}  // namespace

std::unique_ptr<MaxPoolForwardBackward> CudaDevice::GetMaxPoolForwardBackward() {
    return std::make_unique<CudaMaxPoolForwardBackward>(cudnn_handle());
}

}  // namespace cuda
}  // namespace xchainer
