#pragma once

#include <cublas_v2.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <memory>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cudnn.h"
#include "xchainer/cuda/memory_pool.h"
#include "xchainer/device.h"
#include "xchainer/scalar.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace cuda {

// TODO(sonots): Support thread-safety
class CudaDevice : public Device {
public:
    ~CudaDevice() override;

    void Synchronize() override;

    cublasHandle_t cublas_handle();

    // memory.cc

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    std::shared_ptr<void> MakeDataFromForeignPointer(const std::shared_ptr<void>& data) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    // fill.cu

    void Fill(const Array& out, Scalar value) override;

    void Arange(Scalar start, Scalar step, const Array& out) override;

    void Identity(const Array& out) override;

    void Eye(int64_t k, const Array& out) override;

    void Diagflat(const Array& v, int64_t k, const Array& out) override;

    void Linspace(double start, double stop, const Array& out) override;

    // arithmetic.cu

    void Add(const Array& x1, const Array& x2, const Array& out) override;
    void AddAS(const Array& x1, Scalar x2, const Array& out) override;

    void Subtract(const Array& x1, const Array& x2, const Array& out) override;
    void SubtractAS(const Array& x1, Scalar x2, const Array& out) override;

    void Multiply(const Array& x1, const Array& x2, const Array& out) override;
    void MultiplyAS(const Array& x1, Scalar x2, const Array& out) override;

    void Divide(const Array& x1, const Array& x2, const Array& out) override;
    void DivideAS(const Array& x1, Scalar x2, const Array& out) override;

    // reduction.cu

    void ArgMax(const Array& a, const Axes& axis, const Array& out) override;

    void Sum(const Array& a, const Axes& axis, const Array& out) override;
    void AMax(const Array& a, const Axes& axis, const Array& out) override;

    // copy.cu

    void Copy(const Array& a, const Array& out) override;

    void AsType(const Array& a, const Array& out) override;

    // comparison.cu

    void Equal(const Array& x1, const Array& x2, const Array& out) override;

    // activation.cu

    void IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;

    // dot.cc

    void Dot(const Array& a, const Array& b, const Array& out) override;

    // exp_log.cu

    void Exp(const Array& x, const Array& out) override;
    void Log(const Array& x, const Array& out) override;

    // misc.cu

    void Sqrt(const Array& x, const Array& out) override;

    // indexing.cu

    void Take(const Array& a, const Array& indices, int8_t axis, const Array& out) override;

    void AddAt(const Array& a, const Array& indices, int8_t axis, const Array& b, const Array& out) override;

    // conv.cc

    Array Conv(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    Array ConvGradWeight(
            Dtype w_dtype,
            const Shape& w_shape,
            const Array& x,
            const Array& gy,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    Array ConvTranspose(
            const Array& x,
            const Array& w,
            const nonstd::optional<Array>& b,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            const StackVector<int64_t, kMaxNdim>& out_size) override;

    // pool.cc

    std::unique_ptr<MaxPoolForwardBackward> GetMaxPoolForwardBackward() override;

    // batch_norm.cc

    std::unique_ptr<BatchNormForwardBackward> GetBatchNormForwardBackward() override;

protected:
    CudaDevice(CudaBackend& backend, int index) : Device{backend, index}, memory_pool_{index}, cudnn_context_{index} {}

private:
    friend CudaDevice* xchainer::cuda::internal::CreateDevice(CudaBackend&, int);

    MemoryPool memory_pool_;
    internal::CudnnContext cudnn_context_;
    cublasHandle_t cublas_handle_{};
};

}  // namespace cuda
}  // namespace xchainer
