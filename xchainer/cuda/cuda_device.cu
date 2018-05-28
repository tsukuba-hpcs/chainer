#include "xchainer/cuda/cuda_device.h"

#include <cublas_v2.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>

#include <cuda_runtime.h>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backend_util.h"
#include "xchainer/cuda/cast.cuh"
#include "xchainer/cuda/cublas.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/cuda/elementwise.cuh"
#include "xchainer/cuda/reduce.cuh"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_device.h"
#include "xchainer/numeric_limits.h"
#include "xchainer/reduction_kernel_arg.h"
#include "xchainer/routines/connection.h"
#include "xchainer/routines/creation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace cuda {

CudaDevice::~CudaDevice() {
    if (cublas_handle_) {
        cudaSetDevice(index());
        cublasDestroy(cublas_handle_);
    }
}

cublasHandle_t CudaDevice::cublas_handle() {
    if (!cublas_handle_) {
        CheckCudaError(cudaSetDevice(index()));
        CheckCublasError(cublasCreate(&cublas_handle_));
    }
    return cublas_handle_;
}

std::shared_ptr<void> CudaDevice::Allocate(size_t bytesize) {
    CheckCudaError(cudaSetDevice(index()));
    void* ptr = memory_pool_.Malloc(bytesize);
    return std::shared_ptr<void>{ptr, [this](void* ptr) { memory_pool_.Free(ptr); }};
}

std::shared_ptr<void> CudaDevice::MakeDataFromForeignPointer(const std::shared_ptr<void>& data) {
    // check memory validity
    void* ptr = data.get();
    cudaPointerAttributes attr{};
    cudaError_t status = cudaPointerGetAttributes(&attr, ptr);
    switch (status) {
        case cudaSuccess:
            if (attr.isManaged == 0) {
                throw XchainerError{"CUDA memory: ", ptr, " must be a managed (unified) memory"};
            }
            if (attr.device != index()) {
                throw XchainerError{"CUDA memory: ", ptr, " must reside on the device: ", index()};
            }
            break;
        case cudaErrorInvalidValue:
            throw XchainerError{"Memory: ", ptr, " is not a CUDA memory"};
        default:
            Throw(status);
    }
    return data;
}

void CudaDevice::MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) {
    assert(bytesize == 0 || IsPointerCudaMemory(dst));
    if (bytesize == 0) {
        return;
    }
    CheckCudaError(cudaSetDevice(index()));
    if (&src_device == this || nullptr != dynamic_cast<CudaDevice*>(&src_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<native::NativeDevice*>(&src_device) &&
               "CudaDevice only supports copy between cuda or native devices.");
        // Copy from native device
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyHostToDevice));
    }
}

void CudaDevice::MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) {
    assert(bytesize == 0 || src == nullptr || IsPointerCudaMemory(src));
    if (bytesize == 0) {
        return;
    }
    CheckCudaError(cudaSetDevice(index()));
    if (&dst_device == this || nullptr != dynamic_cast<CudaDevice*>(&dst_device)) {
        // Copy between CUDA devices
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToDevice));
    } else {
        assert(nullptr != dynamic_cast<native::NativeDevice*>(&dst_device) &&
               "CudaDevice only supports copy between cuda or native devices.");
        // Copy to native device
        CheckCudaError(cudaMemcpy(dst, src, bytesize, cudaMemcpyDeviceToHost));
    }
}

std::shared_ptr<void> CudaDevice::TransferDataFrom(
        Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    MemoryCopyFrom(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, src_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = dst_device.Allocate(bytesize);
    MemoryCopyTo(dst_ptr.get(), &(static_cast<int8_t*>(src_ptr.get())[offset]), bytesize, dst_device);
    return dst_ptr;
}

std::shared_ptr<void> CudaDevice::FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) {
    std::shared_ptr<void> dst_ptr = Allocate(bytesize);
    CheckCudaError(cudaMemcpy(dst_ptr.get(), src_ptr.get(), bytesize, cudaMemcpyHostToDevice));
    return dst_ptr;
}

namespace {

template <typename T>
struct FillImpl {
    __device__ void operator()(int64_t /*i*/, T& out) { out = value; }
    T value;
};

}  // namespace

void CudaDevice::Fill(const Array& out, Scalar value) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(FillImpl<T>{static_cast<T>(value)}, out);
    });
}

namespace {

template <typename T>
struct ArangeImpl {
    __device__ void operator()(int64_t i, T& out) { out = start + step * i; }
    T start;
    T step;
};

}  // namespace

void CudaDevice::Arange(Scalar start, Scalar step, const Array& out) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(ArangeImpl<T>{static_cast<T>(start), static_cast<T>(step)}, out);
    });
}

namespace {

template <typename T>
struct ArgMaxImpl {
    struct MaxAndArgMax {
        T max;
        int64_t argmax;
    };
    __device__ MaxAndArgMax Identity() { return {T{}, -1}; }
    __device__ MaxAndArgMax MapIn(T in, int64_t index) { return {in, index}; }
    __device__ void Reduce(MaxAndArgMax next, MaxAndArgMax& accum) {
        if (accum.argmax < 0 || accum.max < next.max) {
            accum = next;
        }
    }
    __device__ int64_t MapOut(MaxAndArgMax accum) { return accum.argmax; }
};

}  // namespace

void CudaDevice::ArgMax(const Array& a, const Axes& axis, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(a.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, int64_t>(a, axis, out), ArgMaxImpl<T>{});
    });
}

namespace {

template <typename T>
struct SumImpl {
    __device__ T Identity() { return T{0}; }
    __device__ T MapIn(T in, int64_t /*index*/) { return in; }
    __device__ void Reduce(T next, T& accum) { accum += next; }
    __device__ T MapOut(T accum) { return accum; }
};

}  // namespace

void CudaDevice::Sum(const Array& a, const Axes& axis, const Array& out) {
    assert(xchainer::internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), SumImpl<T>{});
    });
}

namespace {
template <typename T>
__device__ bool IsNan(T /*value*/) {
    return false;
}
__device__ bool IsNan(double value) { return ::isnan(value); }
__device__ bool IsNan(float value) { return ::isnan(value); }

template <typename T>
struct AMaxImpl {
    __device__ T Identity() { return NumericLimits<T>::LowestOrInf(); }
    __device__ T MapIn(T in, int64_t /*index*/) { return in; }
    __device__ void Reduce(T next, T& accum) {
        if (IsNan(next) || accum < next) {
            accum = next;
        }
    }
    __device__ T MapOut(T accum) { return accum; }
};
}  // namespace

void CudaDevice::AMax(const Array& a, const Axes& axis, const Array& out) {
    assert(xchainer::internal::IsValidReductionShape(a.shape(), axis, out.shape(), true));
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Reduce(MakeReductionKernelArg<T, T>(a, axis, out), AMaxImpl<T>{});
    });
}

namespace {

template <typename T>
struct CopyImpl {
    __device__ void operator()(int64_t /*i*/, T a, T& out) { out = a; }
};

}  // namespace

void CudaDevice::Copy(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(CopyImpl<T>{}, a, out);
    });
}

namespace {

template <typename InT, typename OutT>
struct AsTypeImpl {
    __device__ void operator()(int64_t /*i*/, InT a, OutT& out) { out = static_cast<OutT>(a); }
};

}  // namespace

void CudaDevice::AsType(const Array& a, const Array& out) {
    CheckDevicesCompatible(a, out);
    CheckCudaError(cudaSetDevice(index()));
    auto do_astype = [&](auto in_pt, auto out_pt) {
        using InT = typename decltype(in_pt)::type;
        using OutT = typename decltype(out_pt)::type;
        Elementwise<const InT, OutT>(AsTypeImpl<InT, OutT>{}, a, out);
    };
    VisitDtype(out.dtype(), [&](auto out_pt) { VisitDtype(a.dtype(), do_astype, out_pt); });
}

namespace {

template <typename T>
struct EqualImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, bool& out) { out = x1 == x2; }
};

}  // namespace

void CudaDevice::Equal(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(x1.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, bool>(EqualImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct AddImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 + x2; }
};

}  // namespace

// TODO(sonots): support stream
void CudaDevice::Add(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(AddImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct AddASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 + x2; }
    T x2;
};

}  // namespace

void CudaDevice::AddAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(AddASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct SubtractImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 - x2; }
};

}  // namespace

void CudaDevice::Subtract(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(SubtractImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct SubtractASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 - x2; }
    T x2;
};

}  // namespace

void CudaDevice::SubtractAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(SubtractASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct MultiplyImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 * x2; }
};

}  // namespace

// TODO(sonots): support stream
void CudaDevice::Multiply(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(MultiplyImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct MultiplyASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 * x2; }
    T x2;
};

}  // namespace

void CudaDevice::MultiplyAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(MultiplyASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct DivideImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T x2, T& out) { out = x1 / x2; }
};

}  // namespace

void CudaDevice::Divide(const Array& x1, const Array& x2, const Array& out) {
    CheckDevicesCompatible(x1, x2, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(DivideImpl<T>{}, x1, x2, out);
    });
}

namespace {

template <typename T>
struct DivideASImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T& out) { out = x1 / x2; }
    T x2;
};

}  // namespace

void CudaDevice::DivideAS(const Array& x1, Scalar x2, const Array& out) {
    CheckDevicesCompatible(x1, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(DivideASImpl<T>{static_cast<T>(x2)}, x1, out);
    });
}

namespace {

template <typename T>
struct IfLessElseASSAImpl {
    __device__ void operator()(int64_t /*i*/, T x1, T neg, T& out) { out = x1 < x2 ? pos : neg; }
    T x2;
    T pos;
};

}  // namespace

void CudaDevice::IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) {
    CheckDevicesCompatible(x1, neg, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, const T, T>(IfLessElseASSAImpl<T>{static_cast<T>(x2), static_cast<T>(pos)}, x1, neg, out);
    });
}

namespace {

// Dispatch gemm routines based on the element type T
template <typename T>
struct Gemm;

template <>
struct Gemm<float> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        CheckCublasError(cublasSgemm(std::forward<Args>(args)...));
    }
};

template <>
struct Gemm<double> {
    template <typename... Args>
    void operator()(Args&&... args) const {
        CheckCublasError(cublasDgemm(std::forward<Args>(args)...));
    }
};

struct GemmInputLayout {
    int64_t ld = 0;
    cublasOperation_t trans = CUBLAS_OP_T;

    // Makes the array C or Fortran contiguous and configure leading dimension and transposition accordingly.
    Array Configure(const Array& a) {
        assert(a.ndim() == 2);
        if (a.strides()[0] == a.item_size() && a.strides()[0] * a.shape()[0] == a.strides()[1]) {
            // Fortran contiguous
            ld = a.shape()[0];
            return a;
        }
        // Force C contiguous
        ld = a.shape()[1];
        trans = CUBLAS_OP_N;  // transposed
        return a.IsContiguous() ? a : a.AsConstant(CopyKind::kCopy);
    }
};

}  // namespace

void CudaDevice::Dot(const Array& a, const Array& b, const Array& out) {
    CheckDevicesCompatible(a, b, out);
    CheckCudaError(cudaSetDevice(index()));

    assert(a.ndim() == 2);
    assert(b.ndim() == 2);
    assert(out.ndim() == 2);

    int64_t m = a.shape()[0];
    int64_t k = a.shape()[1];
    int64_t n = b.shape()[1];
    assert(b.shape()[0] == k);
    assert(out.shape()[0] == m);
    assert(out.shape()[1] == n);

    if (m == 1 && n == 1) {
        // TODO(beam2d): Write a custom reduction kernel.
        Array l = a.AsConstant();
        Array r = b.AsConstant();
        Array o = out.AsConstant();
        Sum(l.Reshape({k}) * r.Reshape({k}), {0}, o.Reshape({}));
        return;
    }

    bool is_out_contiguous = out.IsContiguous();
    Array out_contiguous = is_out_contiguous ? out : EmptyLike(out, *this);

    auto gemm_impl = [&](auto pt) {
        using T = typename decltype(pt)::type;

        // Note that cuBLAS uses Fortran order.
        // To compute out = a x b, we use cuBLAS to compute out^T = b^T x a^T (here x is the matrix product).

        GemmInputLayout a_layout;
        GemmInputLayout b_layout;
        Array a_config = a_layout.Configure(a);
        Array b_config = b_layout.Configure(b);

        const T one = 1;
        const T zero = 0;
        const T* a_ptr = xchainer::internal::GetRawOffsetData<const T>(a_config);
        const T* b_ptr = xchainer::internal::GetRawOffsetData<const T>(b_config);
        T* out_ptr = xchainer::internal::GetRawOffsetData<T>(out_contiguous);
        Gemm<T>{}(
                cublas_handle(), b_layout.trans, a_layout.trans, n, m, k, &one, b_ptr, b_layout.ld, a_ptr, a_layout.ld, &zero, out_ptr, n);
    };

    if (a.dtype() == Dtype::kFloat32) {
        gemm_impl(PrimitiveType<float>{});
    } else if (a.dtype() == Dtype::kFloat64) {
        gemm_impl(PrimitiveType<double>{});
    } else {
        throw NotImplementedError("dot is not implemented for non-float types in CUDA");
    }

    if (!is_out_contiguous) {
        Copy(out_contiguous, out);
    }
}

namespace {

template <typename T>
struct ExpImpl {
    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = std::exp(x); }
};

}  // namespace

void CudaDevice::Exp(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(ExpImpl<T>{}, x, out);
    });
}

namespace {

template <typename T>
struct LogImpl {
    __device__ void operator()(int64_t /*i*/, T x, T& out) { out = std::log(x); }
};

}  // namespace

void CudaDevice::Log(const Array& x, const Array& out) {
    CheckDevicesCompatible(x, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<const T, T>(LogImpl<T>{}, x, out);
    });
}

namespace {

// Makes axes for permutation that moves [first_axis, last_axis) to the head.
Axes MakeRollingPermutation(int8_t first_axis, int8_t last_axis, int8_t ndim) {
    assert(0 <= first_axis);
    assert(first_axis < last_axis);
    assert(last_axis <= ndim);

    Axes permutation{};
    permutation.resize(ndim);
    auto head_end = permutation.begin() + (last_axis - first_axis);
    auto last = permutation.begin() + last_axis;
    std::iota(permutation.begin(), head_end, first_axis);
    std::iota(head_end, last, int8_t{0});
    std::iota(last, permutation.end(), last_axis);
    return permutation;
}

template <typename T>
__global__ void TakeKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const int64_t> indices_iarray,
        Indexer<> a_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        int64_t common_total_size,
        int64_t axis_dim) {
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        int64_t indices_pos = it.raw_index() / common_total_size;
        int64_t common_pos = it.raw_index() % common_total_size;

        int64_t index = indices_iarray[indices_indexer.It(indices_pos)];
        if (index < 0) {
            index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
        } else {
            index = index % axis_dim;
        }
        assert(0 <= index);
        assert(index < axis_dim);

        out_iarray[it] = a_iarray[a_indexer.It(index * common_total_size + common_pos)];
    }
}

template <typename T>
__global__ void AddAtKernel(
        IndexableArray<const T> a_iarray,
        IndexableArray<const T> b_iarray,
        IndexableArray<T> out_iarray,
        IndexableArray<const int64_t> indices_iarray,
        Indexer<> b_indexer,
        Indexer<> out_indexer,
        Indexer<> indices_indexer,
        int64_t common_total_size,
        int64_t axis_dim) {
    for (auto it = out_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); it; ++it) {
        int64_t axis_pos = it.raw_index() / common_total_size;
        int64_t common_pos = it.raw_index() % common_total_size;

        T out_value = a_iarray[it];

        for (auto it_indices = indices_indexer.It(0); it_indices; ++it_indices) {
            int64_t index = indices_iarray[it_indices];

            if (index < 0) {
                index = axis_dim - ((-index + axis_dim - 1) % axis_dim + 1);
            } else {
                index = index % axis_dim;
            }
            assert(0 <= index);
            assert(index < axis_dim);

            if (index == axis_pos) {
                out_value += b_iarray[b_indexer.It(it_indices.raw_index() * common_total_size + common_pos)];
            }
        }

        out_iarray[it] = out_value;
    }
}

}  // namespace

void CudaDevice::Take(const Array& a, const Array& indices, int8_t axis, const Array& out) {
    CheckDevicesCompatible(a, indices, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // a and out are transposed as follows.
        // a:       (Ni..., N,     Nj...) => (N,     Ni..., Nj...)
        // out:     (Ni..., Nk..., Nj...) => (Nk..., Ni..., Nj...)
        //
        // indices is used as is.
        // indices: (Nk...)

        IndexableArray<const T> a_iarray{a};
        Axes a_perm = MakeRollingPermutation(axis, axis + 1, a.ndim());
        a_iarray.Permute(a_perm);
        Shape a_shape = xchainer::internal::TransposeShape(a.shape(), a_perm);
        Indexer<> a_indexer{a_shape};

        IndexableArray<T> out_iarray{out};
        Axes out_perm = MakeRollingPermutation(axis, axis + indices.ndim(), out.ndim());
        out_iarray.Permute(out_perm);
        Shape out_shape = xchainer::internal::TransposeShape(out.shape(), out_perm);
        Indexer<> out_indexer{out_shape};

        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        int64_t common_total_size = a_indexer.total_size() / a_shape[0];

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&TakeKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        TakeKernel<<<grid_size, block_size>>>(
                a_iarray, out_iarray, indices_iarray, a_indexer, out_indexer, indices_indexer, common_total_size, a_shape[0]);
    });
}

void CudaDevice::AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) {
    // TODO(niboshi): Current implementation only distributes output elements in respective threads. Summation on the indices is performed
    // serially in each thread. This implementation can be improved by distributing indices as well, possibly using atomicAdd.

    assert(a.shape() == out.shape());
    CheckDevicesCompatible(a, indices, out);
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // b and out are transposed as follows.
        // a:       (Ni..., N,     Nj...) => (N,     Ni..., Nj...)
        // b:       (Ni..., Nk..., Nj...) => (Nk..., Ni..., Nj...)
        // out:     (Ni..., N    , Nj...) => (N    , Ni..., Nj...)
        //
        // indices is used as is.
        // indices: (Nk...)

        IndexableArray<const T> a_iarray{a};
        Axes a_perm = MakeRollingPermutation(axis, axis + 1, a.ndim());
        a_iarray.Permute(a_perm);
        Shape a_shape = xchainer::internal::TransposeShape(a.shape(), a_perm);
        Indexer<> a_indexer{a_shape};

        IndexableArray<const T> b_iarray{b};
        Axes b_perm = MakeRollingPermutation(axis, axis + indices.ndim(), b.ndim());
        b_iarray.Permute(b_perm);
        Shape b_shape = xchainer::internal::TransposeShape(b.shape(), b_perm);
        Indexer<> b_indexer{b_shape};

        IndexableArray<T> out_iarray{out};
        Axes out_perm = MakeRollingPermutation(axis, axis + 1, out.ndim());
        out_iarray.Permute(out_perm);
        Shape out_shape = xchainer::internal::TransposeShape(out.shape(), out_perm);
        Indexer<> out_indexer{out_shape};

        IndexableArray<const int64_t> indices_iarray{indices};
        Indexer<> indices_indexer{indices.shape()};

        // size of (Ni..., Nj...) part
        int64_t common_total_size = a_indexer.total_size() / a_shape[0];

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&AddAtKernel<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        AddAtKernel<<<grid_size, block_size>>>(
                a_iarray, b_iarray, out_iarray, indices_iarray, b_indexer, out_indexer, indices_indexer, common_total_size, a_shape[0]);
    });
}

namespace {

template <typename T>
struct IdentityImpl {
    explicit IdentityImpl(int64_t n) : n_plus_one{n + 1} {}
    __device__ void operator()(int64_t i, T& out) { out = i % n_plus_one == 0 ? T{1} : T{0}; }
    int64_t n_plus_one;
};

}  // namespace

void CudaDevice::Identity(const Array& out) {
    assert(out.ndim() == 2);
    assert(out.shape()[0] == out.shape()[1]);

    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(IdentityImpl<T>{out.shape()[0]}, out);
    });
}

namespace {

template <typename T>
struct EyeImpl {
    EyeImpl(int64_t m, int64_t k) : start{k < 0 ? -k * m : k}, stop{m * (m - k)}, step{m + 1} {}
    __device__ void operator()(int64_t i, T& out) { out = start <= i && i < stop && (i - start) % step == 0 ? T{1} : T{0}; }
    int64_t start;
    int64_t stop;
    int64_t step;
};

}  // namespace

void CudaDevice::Eye(int64_t k, const Array& out) {
    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [k, &out](auto pt) {
        using T = typename decltype(pt)::type;
        Elementwise<T>(EyeImpl<T>{out.shape()[1], k}, out);
    });
}

namespace {

template <typename T>
__global__ void SetVecInMat(
        IndexableArray<const T, 1> vec_iarray,
        IndexableArray<T, 2> mat_iarray,
        Indexer<1> vec_indexer,
        Indexer<1> mat_row_indexer,
        Indexer<1> mat_col_indexer,
        Indexer<2> mat_indexer,
        int64_t mat_row_start,
        int64_t mat_col_start) {
    for (auto vec_it = vec_indexer.It(blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x); vec_it; ++vec_it) {
        auto mat_row_it = mat_row_indexer.It(mat_row_start + vec_it.raw_index());
        auto mat_col_it = mat_col_indexer.It(mat_col_start + vec_it.raw_index());
        auto mat_it = mat_indexer.At(mat_row_it, mat_col_it);
        mat_iarray[mat_it] = vec_iarray[vec_it];
    }
}

}  // namespace

void CudaDevice::Diagflat(const Array& v, int64_t k, const Array& out) {
    assert(v.ndim() == 1);
    assert(out.ndim() == 2);

    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;

        // Start indices for the 2-D array axes with applied offset k.
        int64_t row_start{0};
        int64_t col_start{0};

        if (k >= 0) {
            col_start += k;
        } else {
            row_start -= k;
        }

        // Initialize all elements to 0 first instead of conditionally filling in the diagonal.
        Fill(out, T{0});

        IndexableArray<const T, 1> v_iarray{v};
        IndexableArray<T, 2> out_iarray{out};
        Indexer<1> v_indexer{v.shape()};
        Indexer<1> out_row_indexer{Shape{out.shape()[0]}};
        Indexer<1> out_col_indexer{Shape{out.shape()[1]}};
        Indexer<2> out_indexer{out.shape()};

        static const int kMaxBlockSize = CudaOccupancyMaxPotentialBlockSize(&SetVecInMat<T>).block_size;
        int64_t total_size = out_indexer.total_size();
        int64_t grid_size = (total_size + kMaxBlockSize - 1) / kMaxBlockSize;
        int64_t block_size = std::min<int64_t>(total_size, kMaxBlockSize);

        SetVecInMat<<<grid_size, block_size>>>(
                v_iarray, out_iarray, v_indexer, out_row_indexer, out_col_indexer, out_indexer, row_start, col_start);
    });
}

namespace {

template <typename T>
struct LinspaceImpl {
    __device__ void operator()(int64_t i, T& out) {
        double value = n == 1 ? start : (start * (n - 1 - i) + stop * i) / (n - 1);
        out = cuda_numeric_cast<T>(value);
    }
    int64_t n;
    double start;
    double stop;
};

}  // namespace

void CudaDevice::Linspace(double start, double stop, const Array& out) {
    assert(out.ndim() == 1);
    assert(out.shape()[0] > 0);

    CheckCudaError(cudaSetDevice(index()));
    VisitDtype(out.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        int64_t n = out.shape()[0];
        Elementwise<T>(LinspaceImpl<T>{n, start, stop}, out);
    });
}

namespace {

void ConvCheckDevice(const Array& x, const Array& w, const nonstd::optional<Array>& b) {
    if (b) {
        CheckDevicesCompatible(x, w, *b);
    } else {
        CheckDevicesCompatible(x, w);
    }
}

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
                            " are same"};
    }
    if (b && b->dtype() != x.dtype()) {
        throw XchainerError{
            "XChainer cuDNN requires the bias array dtype: ", b->dtype(), " and the input array dtype: ", x.dtype(), " are same"};
    }
}

}  // namespace

Array CudaDevice::Conv(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    if (cover_all) {
        throw XchainerError{"CUDA convolution does not support cover_all"};
    }
    ConvCheckDevice(x, w, b);
    ConvCheckDtype(x, w, b);

    int8_t ndim = w.ndim() - 2;  // Number of spacial dimensions
    if (ndim <= 0) {
        throw DimensionError{"Number of spacial dimensions must be greater than 0"};
    }
    if (static_cast<size_t>(ndim) != stride.size()) {
        throw DimensionError{"Number of dimensions of stride does not match the number of spacial dimensions"};
    }
    if (static_cast<size_t>(ndim) != pad.size()) {
        throw DimensionError{"Number of dimensions of pad does not match the number of spacial dimensions"};
    }

    // w.shape = (out_channels, _, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[0];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    Shape out_shape{batch_size, out_channels};
    for (int8_t i = 0; i < ndim; ++i) {
        out_shape.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], w.shape()[i + 2], stride[i], pad[i], cover_all));
        assert(out_shape.back() > 0);
    }
    Array y = Empty(out_shape, x.dtype(), *this);

    cudnn_context_.ConvolutionForward(x, w, b, y, pad, stride, nonstd::nullopt, 1);

    return y;
}

Array CudaDevice::ConvTranspose(
        const Array& x,
        const Array& w,
        const nonstd::optional<Array>& b,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        const nonstd::optional<StackVector<int64_t, kMaxNdim>>& out_size) {
    // def _forward_cudnn(self, x, W, b):
    //     c = W.shape[1]          # W: C_I, C_O, k_1, k_2, ..., k_N
    //     n, in_c = x.shape[:2]   # x: n, C_I, d_1, d_2, ..., d_N

    //     # Make empty array for output.
    //     y_shape = (n, c) + self.outs  # (n, c_O, out_1, out_2, ..., out_N)
    //     y = cuda.cupy.empty(y_shape, dtype=x.dtype)

    //     pad = self.pad
    //     stride = self.stride
    //     dilation = (1,) * self.ndim
    //     groups = 1
    //     deterministic = configuration.config.cudnn_deterministic
    //     auto_tune = configuration.config.autotune
    //     tensor_core = configuration.config.use_cudnn_tensor_core

    //     cuda.cudnn.convolution_backward_data(
    //         W, x, b, y, pad, stride, dilation, groups,
    //         deterministic=deterministic, auto_tune=auto_tune,
    //         tensor_core=tensor_core)

    //     return y,
    ConvCheckDevice(x, w, b);
    ConvCheckDtype(x, w, b);

    int8_t ndim = w.ndim() - 2;
    assert(ndim > 0);

    // w.shape = (in_channels, out_channels, k_1, k_2, ..., k_N)
    int64_t out_channels = w.shape()[1];
    // x_shape = (batch_size, in_channels, d_1, d_2, ..., d_N)
    int64_t batch_size = x.shape()[0];

    // out_shape = (batch_size, out_channels, out_1, out_2, ..., out_N)
    Shape out_shape{batch_size, out_channels};
    if (out_size) {
        std::copy_n(out_size->begin(), ndim, std::back_inserter(out_shape));
    } else  {
        for (int8_t i = 0; i < ndim; ++i) {
            out_shape.emplace_back(xchainer::internal::GetConvOutDim(x.shape()[i + 2], w.shape()[i + 2], stride[i], pad[i], false));
            assert(out_shape.back() > 0);
        }
    }
    Array y = Empty(out_shape, x.dtype(), *this);

    cudnn_.ConvolutionBackwardData(w, x, b, y, pad, stride, nonstd::nullopt, 1);

    return y;
}

void CudaDevice::Synchronize() {
    CheckCudaError(cudaSetDevice(index()));
    CheckCudaError(cudaDeviceSynchronize());
}

}  // namespace cuda
}  // namespace xchainer
