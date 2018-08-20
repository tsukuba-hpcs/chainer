#include "xchainer/cuda/cuda_conv.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <utility>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backend_util.h"
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/hash_combine.h"
#include "xchainer/macro.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/shape.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {
namespace {

void ConvCheckDtype(const Array& x, const Array& w, const nonstd::optional<Array>& b) {
    // TODO(sonots): Support float16
    if (x.dtype() != Dtype::kFloat32 && x.dtype() != Dtype::kFloat64) {
        throw XchainerError{"XChainer cuDNN supports only float32 or float64 arrays, but the input array dtype is: ", x.dtype()};
    }
    if (w.dtype() != x.dtype()) {
        throw XchainerError{"XChainer cuDNN requires the filter (kernel) array dtype: ",
                            w.dtype(),
                            " and the input array dtype: ",
                            x.dtype(),
                            " to be the same"};
    }
    if (b && b->dtype() != x.dtype()) {
        throw XchainerError{
                "XChainer cuDNN requires the bias array dtype: ", b->dtype(), " and the input array dtype: ", x.dtype(), " to be the same"};
    }
}

}  // namespace

namespace cuda_internal {

std::size_t CudaConv::AlgoCacheKeyHash::operator()(const AlgoCacheKey& key) const {
    std::size_t seed = 0;
    internal::HashCombine(seed, std::hash<int8_t>()(key.x_shape.ndim()));
    for (int64_t v : key.x_shape) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(key.w_shape.ndim()));
    for (int64_t v : key.w_shape) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(key.y_shape.ndim()));
    for (int64_t v : key.y_shape) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(gsl::narrow<int8_t>(key.pad.size())));
    for (int64_t v : key.pad) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<int8_t>()(gsl::narrow<int8_t>(key.stride.size())));
    for (int64_t v : key.stride) {
        internal::HashCombine(seed, std::hash<int64_t>()(v));
    }
    internal::HashCombine(seed, std::hash<std::underlying_type_t<Dtype>>()(static_cast<std::underlying_type_t<Dtype>>(key.dtype)));
    internal::HashCombine(seed, std::hash<size_t>()(key.max_workspace_size));
    return seed;
}

void CudaConv::AddBias(cudnnHandle_t handle, const CudnnTensorDescriptor& y_desc, const Array& y, const Array& b) {
    assert(&b.device() == &y.device());
    assert(b.dtype() == y.dtype());

    int8_t ndim = y.ndim() - 2;  // Number of spatial dimensions
    assert(ndim > 0);

    Shape new_shape{};
    new_shape.emplace_back(1);
    new_shape.emplace_back(b.GetTotalSize());
    for (int8_t i = 0; i < ndim; ++i) {
        new_shape.emplace_back(1);
    }
    Array b_cont = AsContiguousArray(b).Reshape(new_shape);

    CudnnTensorDescriptor b_desc{b_cont};
    CheckCudnnError(cudnnAddTensor(
            handle,
            GetValuePtr<1>(y.dtype()),
            *b_desc,
            internal::GetRawOffsetData<void>(b_cont),
            GetValuePtr<1>(y.dtype()),
            *y_desc,
            internal::GetRawOffsetData<void>(y)));
}

std::pair<cudnnConvolutionFwdAlgo_t, size_t> CudaConv::FindConvolutionForwardAlgorithm(
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
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = AlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = fwd_algo_cache_map_;
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        return it->second;
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionFwdAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            *x_desc,
            internal::GetRawOffsetData<void>(x),
            *filter_desc,
            internal::GetRawOffsetData<void>(w),
            *conv_desc,
            *y_desc,
            internal::GetRawOffsetData<void>(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
}

std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> CudaConv::FindConvolutionBackwardDataAlgorithm(
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
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = AlgoCacheKey{x.shape(), w.shape(), y.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = bwd_data_algo_cache_map_;
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        return it->second;
    }

    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdDataAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionBackwardDataAlgorithmEx(
            handle,
            *filter_desc,
            internal::GetRawOffsetData<void>(w),
            *x_desc,
            internal::GetRawOffsetData<void>(x),
            *conv_desc,
            *y_desc,
            internal::GetRawOffsetData<void>(y),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
}

std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> CudaConv::FindConvolutionBackwardFilterAlgorithm(
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
        const StackVector<int64_t, kMaxNdim>& stride) {
    auto key = AlgoCacheKey{x.shape(), gw.shape(), gy.shape(), pad, stride, x.dtype(), max_workspace_size};
    auto& algo_cache_map = bwd_filter_algo_cache_map_;
    auto it = algo_cache_map.find(key);
    if (it != algo_cache_map.end()) {
        return it->second;
    }

    std::shared_ptr<void> workspace = x.device().Allocate(max_workspace_size);

    cudnnConvolutionBwdFilterAlgoPerf_t perf_result{};
    int returned_algo_count{};

    CheckCudnnError(cudnnFindConvolutionBackwardFilterAlgorithmEx(
            handle,
            *x_desc,
            internal::GetRawOffsetData<void>(x),
            *gy_desc,
            internal::GetRawOffsetData<void>(gy),
            *conv_desc,
            *gw_desc,
            internal::GetRawOffsetData<void>(gw),
            1,  // requested algo count,
            &returned_algo_count,
            &perf_result,
            workspace.get(),
            max_workspace_size));
    assert(returned_algo_count == 1);

    return algo_cache_map[key] = {perf_result.algo, perf_result.memory};
}

// TODO(sonots): Support tensor core
Array CudaConv::Conv(
        CudaDevice& device,
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    if (cover_all) {
        throw XchainerError{"CUDA convolution does not support cover_all"};
    }

    if (b) {
        device.CheckDevicesCompatible(x, w, *b);
    } else {
        device.CheckDevicesCompatible(x, w);
    }

    ConvCheckDtype(x, w, b);

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spatial dimensions to be greater than or equal to 2"};
    }
    assert(w.ndim() == x.ndim());
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));

    // w.shape = (out_channels, _, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[0];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    Shape out_shape{batch_size, out_channels};
    for (int8_t i = 0; i < ndim; ++i) {
        out_shape.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], w.shape()[i + 2], stride[i], pad[i], cover_all));
        assert(out_shape.back() > 0);
    }
    Array y = Empty(out_shape, x.dtype(), device);

    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);

    CudnnTensorDescriptor x_desc{x_cont};
    CudnnTensorDescriptor y_desc{y};
    CudnnFilterDescriptor filter_desc{w_cont};
    CudnnConvolutionDescriptor conv_desc{x.dtype(), pad, stride, nonstd::nullopt /*dilation*/, 1 /*groups*/};

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();
    cudnnHandle_t handle = device.cudnn_handle();

    // auto tune
    std::pair<cudnnConvolutionFwdAlgo_t, size_t> algo_workspace_size = FindConvolutionForwardAlgorithm(
            handle, x_desc, x_cont, filter_desc, w_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);

    cudnnConvolutionFwdAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionForward(
            handle,
            GetValuePtr<1>(x.dtype()),
            *x_desc,
            internal::GetRawOffsetData<void>(x_cont),
            *filter_desc,
            internal::GetRawOffsetData<void>(w_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            *y_desc,
            internal::GetRawOffsetData<void>(y)));

    if (b) {
        AddBias(handle, y_desc, y, *b);
    }

    return y;
}

Array CudaConv::ConvTranspose(
        CudaDevice& device,
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& out_size) {
    if (b) {
        device.CheckDevicesCompatible(x, w, *b);
    } else {
        device.CheckDevicesCompatible(x, w);
    }

    ConvCheckDtype(x, w, b);

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spatial dimensions to be greater than or equal to 2"};
    }
    assert(w.ndim() == x.ndim());
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));
    assert(out_size.size() == static_cast<size_t>(ndim));

    // w.shape = (in_channels, out_channels, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[1];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    // (Note that cover_all is not supported in cuDNN implementation.)
    Shape out_shape{batch_size, out_channels};
    std::copy(out_size.begin(), out_size.end(), std::back_inserter(out_shape));

    Array y = Empty(out_shape, x.dtype(), device);

    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);

    CudnnTensorDescriptor x_desc{x_cont};
    CudnnTensorDescriptor y_desc{y};
    CudnnFilterDescriptor filter_desc{w_cont};
    CudnnConvolutionDescriptor conv_desc{x.dtype(), pad, stride, nonstd::nullopt /*dilation*/, 1 /*group*/};

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();
    cudnnHandle_t handle = device.cudnn_handle();

    // auto tune
    std::pair<cudnnConvolutionBwdDataAlgo_t, size_t> algo_workspace_size = FindConvolutionBackwardDataAlgorithm(
            handle, filter_desc, w_cont, x_desc, x_cont, conv_desc, y_desc, y, max_workspace_size, pad, stride);

    cudnnConvolutionBwdDataAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionBackwardData(
            handle,
            GetValuePtr<1>(x.dtype()),
            *filter_desc,
            internal::GetRawOffsetData<void>(w_cont),
            *x_desc,
            internal::GetRawOffsetData<void>(x_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            *y_desc,
            internal::GetRawOffsetData<void>(y)));

    if (b) {
        AddBias(handle, y_desc, y, *b);
    }

    return y;
}

Array CudaConv::ConvGradWeight(
        CudaDevice& device,
        Dtype w_dtype,
        const Shape& w_shape,
        const Array& x,
        const Array& gy,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    if (cover_all) {
        throw XchainerError{"CUDA convolution does not support cover_all"};
    }

    device.CheckDevicesCompatible(x, gy);

    int8_t ndim = x.ndim() - 2;  // Number of spatial dimensions
    if (ndim < 2) {
        throw DimensionError{"CUDA convolution requires number of spatial dimensions to be greater than or equal to 2"};
    }

    assert(x.ndim() == w_shape.ndim());
    assert(stride.size() == static_cast<size_t>(ndim));
    assert(pad.size() == static_cast<size_t>(ndim));
    assert(gy.ndim() == w_shape.ndim());

    assert(x.dtype() == w_dtype);
    assert(x.dtype() == gy.dtype());

    if (XCHAINER_DEBUG) {
        // w_shape = (out_channels, in_channels, k_1, k_2, ..., k_N)
        int64_t out_channels = w_shape[0];
        // x.shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
        int64_t batch_size = x.shape()[0];
        // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
        Shape out_shape{batch_size, out_channels};
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(internal::GetConvOutDim(x.shape()[i + 2], w_shape[i + 2], stride[i], pad[i], cover_all));
            assert(out_shape.back() > 0);
        }
        assert(gy.shape() == out_shape);
    }

    Array gw = Empty(w_shape, w_dtype, device);

    auto& backend = static_cast<CudaBackend&>(device.backend());  // NOLINT

    Array x_cont = AsContiguousArray(x);
    Array gy_cont = AsContiguousArray(gy);
    Array gw_cont = AsContiguousArray(gw);

    CudnnTensorDescriptor x_desc{x_cont};
    CudnnTensorDescriptor gy_desc{gy_cont};
    CudnnFilterDescriptor gw_desc{gw_cont};
    CudnnConvolutionDescriptor conv_desc{x.dtype(), pad, stride, nonstd::nullopt /*dilation*/, 1 /*groups*/};

    size_t max_workspace_size = backend.GetCudnnMaxWorkspaceSize();
    cudnnHandle_t handle = device.cudnn_handle();

    // auto tune
    std::pair<cudnnConvolutionBwdFilterAlgo_t, size_t> algo_workspace_size =
            FindConvolutionBackwardFilterAlgorithm(handle, x_desc, x, gy_desc, gy, conv_desc, gw_desc, gw, max_workspace_size, pad, stride);

    cudnnConvolutionBwdFilterAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionBackwardFilter(
            handle,
            GetValuePtr<1>(x.dtype()),
            *x_desc,
            internal::GetRawOffsetData<void>(x_cont),
            *gy_desc,
            internal::GetRawOffsetData<void>(gy_cont),
            *conv_desc,
            algo,
            workspace.get(),
            workspace_size,
            GetValuePtr<0>(x.dtype()),
            *gw_desc,
            internal::GetRawOffsetData<void>(gw)));

    return gw;
}

}  // namespace cuda_internal
}  // namespace cuda
}  // namespace xchainer
