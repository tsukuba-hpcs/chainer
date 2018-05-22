#include "xchainer/cuda/cudnn.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <cudnn.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_device.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {

CudnnError::CudnnError(cudnnStatus_t status) : XchainerError{cudnnGetErrorString(status)}, status_{status} {}

void CheckCudnnError(cudnnStatus_t status) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw CudnnError{status};
    }
}

namespace {

cudnnDataType_t GetCudnnDataType(Dtype dtype) {
    switch (dtype) {
        case Dtype::kFloat32:
            return CUDNN_DATA_FLOAT;
        case Dtype::kFloat64:
            return CUDNN_DATA_DOUBLE;
        // TODO(sonots): Support float16 if it becomes avaialable
        // case Dtype::kFloat16:
        //    return CUDNN_DATA_HALF;
        default:
            throw DtypeError{"Dtype ", dtype, " is not supported in cuDNN"};
    }
}

void SetTensorDescriptor(cudnnTensorDescriptor_t desc, const Array& arr, cudnnTensorFormat_t format) {
    if (!arr.IsContiguous()) {
        throw XchainerError{"CuDNN supports only c-contiguous arrays"};
    }
    constexpr int int_max = std::numeric_limits<int>::max();
    for (int8_t i = 0; i < arr.shape().ndim(); ++i) {
        if (arr.shape()[i] > int_max) {
            throw XchainerError{"The shape size: ", arr.shape()[i], " of dimension: ", i, " is larger than ", int_max};
        }
    }

    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        int n = static_cast<int>(arr.shape()[0]);
        int c = static_cast<int>(arr.shape()[1]);
        int h = static_cast<int>(arr.shape()[2]);
        int w = static_cast<int>(arr.shape()[3]);
        CheckCudnnError(cudnnSetTensor4dDescriptor(desc, format, cudnn_dtype, n, c, h, w));
    } else {
        StackVector<int, kMaxNdim> int_strides;
        for (int8_t i = 0; i < arr.strides().ndim(); ++i) {
            int64_t v = arr.strides()[i] / arr.item_size();
            if (v > int_max) {
                throw XchainerError{"The array stride: ", v, " (in number of items) of dimension: ", i, " is larger than ", int_max};
            }
            int_strides.emplace_back(static_cast<int>(v));
        }
        StackVector<int, kMaxNdim> int_shape;
        for (int64_t v : arr.shape()) {
            int_shape.emplace_back(static_cast<int>(v));
        }
        CheckCudnnError(cudnnSetTensorNdDescriptor(desc, cudnn_dtype, arr.ndim(), &int_shape[0], &int_strides[0]));
    }
}

void SetFilterDescriptor(cudnnFilterDescriptor_t desc, const Array& arr, cudnnTensorFormat_t format) {
    if (!arr.IsContiguous()) {
        throw XchainerError{"CuDNN supports only c-contiguous arrays"};
    }
    constexpr int int_max = std::numeric_limits<int>::max();
    for (int8_t i = 0; i < arr.shape().ndim(); ++i) {
        if (arr.shape()[i] > int_max) {
            throw XchainerError{"The shape size: ", arr.shape()[i], " of dimension: ", i, " is larger than ", int_max};
        }
    }

    cudnnDataType_t cudnn_dtype = GetCudnnDataType(arr.dtype());
    if (arr.shape().ndim() == 4) {
        int n = static_cast<int>(arr.shape()[0]);
        int c = static_cast<int>(arr.shape()[1]);
        int h = static_cast<int>(arr.shape()[2]);
        int w = static_cast<int>(arr.shape()[3]);
        CheckCudnnError(cudnnSetFilter4dDescriptor(desc, cudnn_dtype, format, n, c, h, w));
    } else {
        StackVector<int, kMaxNdim> int_shape;
        for (int64_t v : arr.shape()) {
            int_shape.emplace_back(static_cast<int>(v));
        }
        CheckCudnnError(cudnnSetFilterNdDescriptor(desc, cudnn_dtype, format, arr.ndim(), &int_shape[0]));
    }
}

void SetConvolutionDescriptor(
        cudnnConvolutionDescriptor_t desc,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups,
        Dtype dtype,
        cudnnConvolutionMode_t mode) {
    size_t ndim = pad.size();
    if (ndim != stride.size()) {
        throw DimensionError{"pad and stride must be of same length"};
    }
    if (dilation && ndim != dilation->size()) {
        throw DimensionError{"pad and dilation must be of same length"};
    }

    constexpr int int_max = std::numeric_limits<int>::max();
    StackVector<int, kMaxNdim> int_stride;
    for (size_t i = 0; i < stride.size(); ++i) {
        if (stride[i] > int_max) {
            throw XchainerError{"The stride: ", stride[i], " of dimension: ", i, " is larger than ", int_max};
        }
        int_stride.emplace_back(static_cast<int>(stride[i]));
    }
    StackVector<int, kMaxNdim> int_pad;
    for (size_t i = 0; i < pad.size(); ++i) {
        if (pad[i] > int_max) {
            throw XchainerError{"The pad: ", pad[i], " of dimension: ", i, " is larger than ", int_max};
        }
        int_pad.emplace_back(static_cast<int>(pad[i]));
    }
    if (dilation) {
        for (size_t i = 0; i < dilation->size(); ++i) {
            if ((*dilation)[i] > int_max) {
                throw XchainerError{"The dilation, ", (*dilation)[i], " of dimension: ", i, " is larger than ", int_max};
            }
        }
    }

    cudnnDataType_t compute_type = GetCudnnDataType(dtype);

    if (ndim != 2) {
        StackVector<int, kMaxNdim> int_dilation;
        if (!dilation) {
            // TODO(sonots): Use assign(ndim, 1) if it becomes available
            for (size_t i = 0; i < ndim; ++i) {
                int_dilation.emplace_back(1);
            }
        } else {
            for (int64_t v : *dilation) {
                int_dilation.emplace_back(static_cast<int>(v));
            }
        }
        CheckCudnnError(cudnnSetConvolutionNdDescriptor(desc, ndim, &int_pad[0], &int_stride[0], &int_dilation[0], mode, compute_type));
    } else {
        int d0 = 0, d1 = 0;
        if (!dilation) {
            d0 = d1 = 1;
        } else {
            d0 = static_cast<int>((*dilation)[0]);
            d1 = static_cast<int>((*dilation)[1]);
        }
        int p0 = static_cast<int>(pad[0]);
        int p1 = static_cast<int>(pad[1]);
        int s0 = static_cast<int>(stride[0]);
        int s1 = static_cast<int>(stride[1]);

        CheckCudnnError(cudnnSetConvolution2dDescriptor(desc, p0, p1, s0, s1, d0, d1, mode, compute_type));
    }
    if (groups > 1) {
        CheckCudnnError(cudnnSetConvolutionGroupCount(desc, groups));
    }
}

std::shared_ptr<cudnnTensorStruct> CreateTensorDescriptor(const Array& arr, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    cudnnTensorDescriptor_t desc{};
    CheckCudnnError(cudnnCreateTensorDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnTensorStruct>{
            desc, [](cudnnTensorDescriptor_t desc) { CheckCudnnError(cudnnDestroyTensorDescriptor(desc)); }};
    SetTensorDescriptor(desc, arr, format);
    return shared_desc;
}

std::shared_ptr<cudnnFilterStruct> CreateFilterDescriptor(const Array& arr, cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW) {
    cudnnFilterDescriptor_t desc{};
    CheckCudnnError(cudnnCreateFilterDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnFilterStruct>{
            desc, [](cudnnFilterDescriptor_t desc) { CheckCudnnError(cudnnDestroyFilterDescriptor(desc)); }};
    SetFilterDescriptor(desc, arr, format);
    return shared_desc;
}

std::shared_ptr<cudnnConvolutionStruct> CreateConvolutionDescriptor(
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        Dtype dtype,
        cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation = nonstd::nullopt,
        int groups = 1) {
    cudnnConvolutionDescriptor_t desc{};
    CheckCudnnError(cudnnCreateConvolutionDescriptor(&desc));
    auto shared_desc = std::shared_ptr<cudnnConvolutionStruct>{
            desc, [](cudnnConvolutionDescriptor_t desc) { CheckCudnnError(cudnnDestroyConvolutionDescriptor(desc)); }};
    SetConvolutionDescriptor(desc, pad, stride, dilation, groups, dtype, mode);
    return shared_desc;
}

std::pair<cudnnConvolutionFwdAlgo_t, size_t> FindConvolutionForwardAlgorithm(
        cudnnHandle_t handle,
        const std::shared_ptr<cudnnTensorStruct>& x_desc,
        const Array& x,
        const std::shared_ptr<cudnnFilterStruct>& filter_desc,
        const Array& w,
        const std::shared_ptr<cudnnConvolutionStruct>& conv_desc,
        const std::shared_ptr<cudnnTensorStruct>& y_desc,
        const Array& y,
        size_t max_workspace_size) {
    std::shared_ptr<void> workspace = y.device().Allocate(max_workspace_size);

    int requested_algo_count = 1;
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results(requested_algo_count);
    int returned_algo_count;

    CheckCudnnError(cudnnFindConvolutionForwardAlgorithmEx(
            handle,
            x_desc.get(),
            x.data().get(),
            filter_desc.get(),
            w.data().get(),
            conv_desc.get(),
            y_desc.get(),
            y.data().get(),
            requested_algo_count,
            &returned_algo_count,
            &perf_results[0],
            workspace.get(),
            max_workspace_size));

    return {perf_results[0].algo, perf_results[0].memory};
}

}  // namespace

namespace internal {

// TODO(sonots): Support tensor core
void ConvolutionForward(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        Array& y,
        const StackVector<int64_t, kMaxNdim>& pad,
        const StackVector<int64_t, kMaxNdim>& stride,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& dilation,
        int groups) {
    CudaDevice& device = *dynamic_cast<CudaDevice*>(&x.device());
    CudaBackend& backend = *dynamic_cast<CudaBackend*>(&device.backend());

    if (&y.device() != &device) {
        throw XchainerError{
                "The output array device: ", y.device().name(), " must be same with the input array device: ", x.device().name()};
    }
    if (&w.device() != &device) {
        throw XchainerError{
                "The filter (kernel) array device: ", w.device().name(), " must be same with the input array device: ", x.device().name()};
    }
    // TODO(sonots): Support float16
    if (x.dtype() != Dtype::kFloat32 && x.dtype() != Dtype::kFloat64) {
        throw XchainerError{"XChainer cuDNN supports only float32 or float64 arrays, but the input array dtype is: ", x.dtype()};
    }
    if (y.dtype() != x.dtype()) {
        throw XchainerError{"The output array dtype: ", y.dtype(), " must be same with the input array dtype: ", x.dtype()};
    }
    if (w.dtype() != x.dtype()) {
        throw XchainerError{"The filter (kernel) array dtype: ", w.dtype(), " must be same with the input array dtype: ", x.dtype()};
    }

    float float_zero = 0, float_one = 1;
    double double_zero = 0, double_one = 1;
    void* zero;
    void* one;
    if (x.dtype() == Dtype::kFloat64) {
        zero = &double_zero;
        one = &double_one;
    } else {
        zero = &float_zero;
        one = &float_one;
    }

    cudnnHandle_t handle = device.cudnn_handle();
    Array x_cont = AsContiguousArray(x);
    Array w_cont = AsContiguousArray(w);
    assert(y.IsContiguous());

    std::shared_ptr<cudnnTensorStruct> x_desc = CreateTensorDescriptor(x_cont);
    std::shared_ptr<cudnnTensorStruct> y_desc = CreateTensorDescriptor(y);
    std::shared_ptr<cudnnFilterStruct> filter_desc = CreateFilterDescriptor(w_cont, CUDNN_TENSOR_NCHW);
    std::shared_ptr<cudnnConvolutionStruct> conv_desc =
            CreateConvolutionDescriptor(pad, stride, x.dtype(), CUDNN_CROSS_CORRELATION, dilation, groups);
    size_t max_workspace_size = backend.max_workspace_size();

    // auto tune
    std::pair<cudnnConvolutionFwdAlgo_t, size_t> algo_workspace_size =
            FindConvolutionForwardAlgorithm(handle, x_desc, x_cont, filter_desc, w_cont, conv_desc, y_desc, y, max_workspace_size);
    cudnnConvolutionFwdAlgo_t algo = std::get<0>(algo_workspace_size);
    size_t workspace_size = std::max(max_workspace_size, std::get<1>(algo_workspace_size));
    std::shared_ptr<void> workspace = device.Allocate(workspace_size);

    CheckCudnnError(cudnnConvolutionForward(
            handle,
            one,
            x_desc.get(),
            x_cont.data().get(),
            filter_desc.get(),
            w_cont.data().get(),
            conv_desc.get(),
            algo,
            workspace.get(),
            workspace_size,
            zero,
            y_desc.get(),
            y.data().get()));

    if (b) {
        if (&b->device() != &device) {
            throw XchainerError{
                    "The bias array device: ", b->device().name(), " must be same with the input array device: ", x.device().name()};
        }
        if (b->dtype() != x.dtype()) {
            throw XchainerError{"The bias array dtype: ", b->dtype(), " must be same with the input array dtype: ", x.dtype()};
        }

        int8_t ndim = x.ndim() - 2;
        assert(ndim > 0);

        Shape new_shape;
        new_shape.emplace_back(1);
        new_shape.emplace_back(b->GetTotalSize());
        // TODO(sonots): Use assign(ndim, 1) if it becomes available
        for (int8_t i = 0; i < ndim; ++i) {
            new_shape.emplace_back(1);
        }
        Array b_cont = AsContiguousArray(*b).Reshape(new_shape);
        std::shared_ptr<cudnnTensorStruct> b_desc = CreateTensorDescriptor(b_cont);
        CheckCudnnError(cudnnAddTensor(handle, one, b_desc.get(), b_cont.data().get(), one, y_desc.get(), y.data().get()));
    }
}

}  // namespace internal
}  // namespace cuda
}  // namespace xchainer
