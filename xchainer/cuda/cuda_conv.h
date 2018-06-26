#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/constant.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/dtype.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {

class CudaDevice;

namespace internal {

class CudaConv {
public:
    Array Conv(
            CudaDevice& device,
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all);
    Array ConvTranspose(
            CudaDevice& device,
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& out_size);
    Array ConvGradWeight(
            CudaDevice& device,
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all);

private:
    // for unit-tests
    friend size_t GetFwdAlgoCacheMapSize(const CudaConv& cuda_conv);
    friend size_t GetBwdDataAlgoCacheMapSize(const CudaConv& cuda_conv);
    friend size_t GetBwdFilterAlgoCacheMapSize(const CudaConv& cuda_conv);

    void AddBias(cudnnHandle_t handle, const CudnnTensorDescriptor& y_desc, const Array& y, const Array& b);

    std::pair<cudnnConvolutionFwdAlgo_t, size_t> FindConvolutionForwardAlgorithm(
            cudnnHandle_t handle,
            const CudnnTensorDescriptor& x_desc,
            const Array& x,
            const CudnnFilterDescriptor& filter_desc,
            const Array& w,
            const CudnnConvolutionDescriptor& conv_desc,
            const CudnnTensorDescriptor& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> FindConvolutionBackwardDataAlgorithm(
            cudnnHandle_t handle,
            const CudnnFilterDescriptor& filter_desc,
            const Array& w,
            const CudnnTensorDescriptor& x_desc,
            const Array& x,
            const CudnnConvolutionDescriptor& conv_desc,
            const CudnnTensorDescriptor& y_desc,
            const Array& y,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);
    std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> FindConvolutionBackwardFilterAlgorithm(
            cudnnHandle_t handle,
            const CudnnTensorDescriptor& x_desc,
            const Array& x,
            const CudnnTensorDescriptor& gy_desc,
            const Array& gy,
            const CudnnConvolutionDescriptor& conv_desc,
            const CudnnFilterDescriptor& gw_desc,
            const Array& gw,
            size_t max_workspace_size,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& stride);

    struct AlgoCacheKey {
        Shape x_shape;
        Shape w_shape;
        Shape y_shape;
        StackVector<int64_t, kMaxNdim> pad;
        StackVector<int64_t, kMaxNdim> stride;
        Dtype dtype;
        size_t max_workspace_size;

        bool operator==(const AlgoCacheKey& other) const {
            return x_shape == other.x_shape && w_shape == other.w_shape && y_shape == other.y_shape && pad == other.pad &&
                   stride == other.stride && dtype == other.dtype && max_workspace_size == other.max_workspace_size;
        }

        bool operator!=(const AlgoCacheKey& other) const { return !operator==(other); }
    };

    struct AlgoCacheKeyHash {
        using result_type = std::size_t;
        std::size_t operator()(const AlgoCacheKey& key) const;
    };

    using FwdAlgoCacheMap = std::unordered_map<AlgoCacheKey, std::pair<cudnnConvolutionFwdAlgo_t, size_t>, AlgoCacheKeyHash>;
    using BwdDataAlgoCacheMap = std::unordered_map<AlgoCacheKey, std::pair<cudnnConvolutionBwdDataAlgo_t, size_t>, AlgoCacheKeyHash>;
    using BwdFilterAlgoCacheMap = std::unordered_map<AlgoCacheKey, std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t>, AlgoCacheKeyHash>;

    FwdAlgoCacheMap fwd_algo_cache_map_{};
    BwdDataAlgoCacheMap bwd_data_algo_cache_map_{};
    BwdFilterAlgoCacheMap bwd_filter_algo_cache_map_{};
};

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
