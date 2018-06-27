#include "xchainer/cuda/cuda_device.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include <cudnn.h>

#include "xchainer/array.h"
#include "xchainer/backend_util.h"
#include "xchainer/constant.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/pooling.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {
namespace {

// Uses the previously computed y to find the indices for which the upstream gradients should be propagated.
// It is faster in some cases than looking for the argmax again since we do not have to go though all elements.
template <typename T>
__global__ void MaxPoolDoubleBackwardKernel(
        IndexableArray<const T> ggx_iarray,
        IndexableArray<const T> x_iarray,
        IndexableArray<const T> y_iarray,
        IndexableArray<T> ggy_iarray,
        Indexer<> x_indexer,
        Indexer<> y_indexer,
        Indexer<> kernel_indexer,
        int64_t* kernel_size,
        int64_t* stride,
        int64_t* pad) {
    NdimIndex y_index{y_indexer.ndim()};
    NdimIndex x_index{x_indexer.ndim()};
    int8_t ndim = x_indexer.ndim() - 2;

    for (auto it = y_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        // Compute the y index (batch, channel, out_1, out_2, ..., out_n) from the raw index.
        int64_t size = it.raw_index();
        for (int8_t i = x_indexer.ndim() - 1; i >= 2; --i) {
            int64_t dim = y_indexer.shape()[i];
            y_index.index()[i] = size % dim;
            size /= dim;
        }
        int64_t channels = y_indexer.shape()[1];
        int64_t channel = size % channels;
        int64_t batch = size / channels % y_indexer.shape()[0];

        y_index.index()[0] = batch;
        y_index.index()[1] = channel;

        // Use y to find the index of x, then propagate the element from ggx corresponding to that index.
        T y = y_iarray[y_indexer.At(y_index)];

        x_index.index()[0] = batch;
        x_index.index()[1] = channel;

        for (auto it_kernel = kernel_indexer.It(0); it_kernel; ++it_kernel) {
            for (int8_t i = 0; i < ndim; ++i) {
                int64_t idx = y_index.index()[2 + i] * stride[i] + it_kernel.index()[i] - pad[i];
                idx = max(idx, int64_t{0});
                idx = min(idx, x_indexer.shape()[2 + i] - 1);
                x_index.index()[2 + i] = idx;
            }

            auto it_x = x_indexer.At(x_index);
            if (y == x_iarray[it_x]) {
                ggy_iarray[it] = ggx_iarray[it_x];
                break;
            }
        }
    }
}

class PoolImpl {
public:
    PoolImpl(
            cudnnHandle_t cudnn_handle,
            StackVector<int64_t, kMaxNdim> kernel_size,
            StackVector<int64_t, kMaxNdim> stride,
            StackVector<int64_t, kMaxNdim> pad,
            bool cover_all,
            cudnnPoolingMode_t cudnn_pooling_mode)
        : cudnn_handle_{cudnn_handle},
          kernel_size_{std::move(kernel_size)},
          stride_{std::move(stride)},
          pad_{std::move(pad)},
          cover_all_{cover_all},
          cudnn_pooling_mode_{cudnn_pooling_mode} {
        if (cover_all_) {
            throw XchainerError{"CUDA pooling does not support cover_all"};
        }
    }

    Array Forward(const Array& x) {
        int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions
        if (ndim != 2 && ndim != 3) {
            throw DimensionError{"XChainer cuDNN pooling supports only 2 and 3 spatial dimensions."};
        }

        assert(kernel_size_.size() == static_cast<size_t>(ndim));
        assert(stride_.size() == static_cast<size_t>(ndim));
        assert(pad_.size() == static_cast<size_t>(ndim));

        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        Shape out_shape{x.shape()[0], x.shape()[1]};
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], kernel_size_[i], stride_[i], pad_[i], cover_all_));
            assert(out_shape.back() > 0);
        }

        Array y = Empty(out_shape, x.dtype(), x.device());
        Array x_cont = AsContiguousArray(x);

        internal::CudnnTensorDescriptor x_desc{x_cont};
        internal::CudnnTensorDescriptor y_desc{y};

        internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kernel_size_, pad_, stride_};

        CheckCudnnError(cudnnPoolingForward(
                cudnn_handle_,
                *pool_desc,
                internal::GetValuePtr<1>(x.dtype()),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                internal::GetValuePtr<0>(x.dtype()),
                *y_desc,
                xchainer::internal::GetRawOffsetData<void>(y)));

        x_ = x.AsConstant();
        y_ = y.AsConstant();

        return y;
    }

    Array Backward(const Array& gout) {
        int8_t ndim = x_.ndim() - 2;  // Number of spacial dimensions
        if (ndim < 2) {
            throw DimensionError{"CUDA pooling requires number of spatial dimensions to be greater than or equal to 2"};
        }

        assert(kernel_size_.size() == static_cast<size_t>(ndim));
        assert(stride_.size() == static_cast<size_t>(ndim));
        assert(pad_.size() == static_cast<size_t>(ndim));
        assert(gout.shape() == y_.shape());

        Array gx = EmptyLike(x_, x_.device());
        Array y_cont = AsContiguousArray(y_);
        Array gout_cont = AsContiguousArray(gout);
        Array x_cont = AsContiguousArray(x_);

        internal::CudnnTensorDescriptor y_desc{y_cont};
        internal::CudnnTensorDescriptor gout_desc{gout_cont};
        internal::CudnnTensorDescriptor x_desc{x_cont};
        internal::CudnnTensorDescriptor gx_desc{gx};

        internal::CudnnPoolingDescriptor pool_desc{cudnn_pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, kernel_size_, pad_, stride_};

        CheckCudnnError(cudnnPoolingBackward(
                cudnn_handle_,
                *pool_desc,
                internal::GetValuePtr<1>(x_.dtype()),
                *y_desc,
                xchainer::internal::GetRawOffsetData<void>(y_cont),
                *gout_desc,
                xchainer::internal::GetRawOffsetData<void>(gout_cont),
                *x_desc,
                xchainer::internal::GetRawOffsetData<void>(x_cont),
                internal::GetValuePtr<0>(x_.dtype()),
                *gx_desc,
                xchainer::internal::GetRawOffsetData<void>(gx)));

        return gx;
    }

    Array DoubleBackward(const Array& ggx) {
        Device& device = ggx.device();
        Array ggy = EmptyLike(y_, y_.device());

        VisitDtype(ggy.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;

            IndexableArray<const T> ggx_iarray{ggx};
            IndexableArray<const T> x_iarray{x_};
            IndexableArray<const T> y_iarray{y_};
            IndexableArray<T> ggy_iarray{ggy};

            Indexer<> x_indexer{x_.shape()};
            Indexer<> y_indexer{y_.shape()};
            Indexer<> kernel_indexer{Shape{kernel_size_.begin(), kernel_size_.end()}};

            std::shared_ptr<void> kernel_size = device.Allocate(kernel_size_.size() * sizeof(int64_t));
            CheckCudaError(
                    cudaMemcpy(kernel_size.get(), kernel_size_.data(), kernel_size_.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

            std::shared_ptr<void> stride = device.Allocate(stride_.size() * sizeof(int64_t));
            CheckCudaError(cudaMemcpy(stride.get(), stride_.data(), stride_.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

            std::shared_ptr<void> pad = device.Allocate(pad_.size() * sizeof(int64_t));
            CheckCudaError(cudaMemcpy(pad.get(), pad_.data(), pad_.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

            static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&MaxPoolDoubleBackwardKernel<T>).block_size;
            int64_t total_size = y_indexer.total_size();
            int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
            int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

            MaxPoolDoubleBackwardKernel<<<grid_size, block_size>>>(
                    ggx_iarray,
                    x_iarray,
                    y_iarray,
                    ggy_iarray,
                    x_indexer,
                    y_indexer,
                    kernel_indexer,
                    static_cast<int64_t*>(kernel_size.get()),
                    static_cast<int64_t*>(stride.get()),
                    static_cast<int64_t*>(pad.get()));
        });

        return ggy;
    }

private:
    cudnnHandle_t cudnn_handle_;
    const StackVector<int64_t, kMaxNdim> kernel_size_;
    const StackVector<int64_t, kMaxNdim> stride_;
    const StackVector<int64_t, kMaxNdim> pad_;
    bool cover_all_;
    cudnnPoolingMode_t cudnn_pooling_mode_;
    Array x_;
    Array y_;
};

class CudaMaxPoolForwardBackward : public xchainer::MaxPoolForwardBackward {
public:
    explicit CudaMaxPoolForwardBackward(
            cudnnHandle_t cudnn_handle,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all)
        : pool_impl_{cudnn_handle, kernel_size, stride, pad, cover_all, CUDNN_POOLING_MAX} {}

    Array Forward(const Array& x) override { return pool_impl_.Forward(x); }

    Array Backward(const Array& gout) override { return pool_impl_.Backward(gout); }

    Array DoubleBackward(const Array& ggx) override { return pool_impl_.DoubleBackward(ggx); }

private:
    PoolImpl pool_impl_;
};

}  // namespace

std::unique_ptr<MaxPoolForwardBackward> CudaDevice::GetMaxPoolForwardBackward(
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    return std::make_unique<CudaMaxPoolForwardBackward>(cudnn_handle(), kernel_size, stride, pad, cover_all);
}

namespace {

cudnnPoolingMode_t GetCudnnPoolingMode(AveragePoolPadMode pad_mode) {
    switch (pad_mode) {
        case AveragePoolPadMode::kZero:
            return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
        case AveragePoolPadMode::kIgnore:
            return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
        default:
            XCHAINER_NEVER_REACH();
    }
}

class CudaAveragePoolForwardBackward : public xchainer::AveragePoolForwardBackward {
public:
    explicit CudaAveragePoolForwardBackward(
            cudnnHandle_t cudnn_handle,
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            AveragePoolPadMode pad_mode)
        : pool_impl_{cudnn_handle, kernel_size, stride, pad, false, GetCudnnPoolingMode(pad_mode)} {}

    Array Forward(const Array& x) override { return pool_impl_.Forward(x); }

    Array Backward(const Array& gout) override { return pool_impl_.Backward(gout); }

private:
    PoolImpl pool_impl_;
};

}  // namespace

std::unique_ptr<AveragePoolForwardBackward> CudaDevice::GetAveragePoolForwardBackward(
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        AveragePoolPadMode pad_mode) {
    return std::make_unique<CudaAveragePoolForwardBackward>(cudnn_handle(), kernel_size, stride, pad, pad_mode);
}

}  // namespace cuda
}  // namespace xchainer
