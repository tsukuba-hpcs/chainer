#pragma once

#include <cublas_v2.h>
#include <cudnn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <utility>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/cuda/cublas.h"
#include "chainerx/cuda/cuda_backend.h"
#include "chainerx/cuda/cuda_conv.h"
#include "chainerx/cuda/cudnn.h"
#include "chainerx/cuda/memory_pool.h"
#include "chainerx/device.h"
#include "chainerx/routines/pooling.h"
#include "chainerx/scalar.h"
#include "chainerx/stack_vector.h"

namespace chainerx {
namespace cuda {

class CudaDevice;

namespace cuda_internal {

class CudaConvTest;  // for unit-tests

// Keeps any memory from being freed before CUDA asynchronous operations are finished.
// Operations in this class are thread safe.
class MemoryKeeper {
public:
    ~MemoryKeeper();

    // Registers a pointer to a memory chunk.
    // The memory is only freed after all preceding CUDA operations in the stream are finished.
    // TODO(niboshi): Currently only the default stream is supported.
    void Add(cudaStream_t stream, std::shared_ptr<void> memory);

    // Checks for recorded events and frees the associated memories.
    void Collect();

private:
    std::mutex mutex_{};
    std::queue<std::pair<cudaEvent_t, std::shared_ptr<void>>> queue_{};
};

// Keeps handles and other device internals.
// These internals are exposed through `GetDeviceInternals` for CUDA internal usages.
class DeviceInternals {
public:
    DeviceInternals(const DeviceInternals&) = delete;
    DeviceInternals(DeviceInternals&&) = delete;
    DeviceInternals& operator=(const DeviceInternals&) = delete;
    DeviceInternals& operator=(DeviceInternals&&) = delete;

    explicit DeviceInternals(int device_index) : cublas_handle_{device_index}, cudnn_handle_{device_index} {}

    cuda_internal::CublasHandle& cublas_handle() { return cublas_handle_; }

    cuda_internal::CudnnHandle& cudnn_handle() { return cudnn_handle_; }

    cuda_internal::CudaConv& cuda_conv() { return cuda_conv_; }

private:
    cuda_internal::CublasHandle cublas_handle_;

    cuda_internal::CudnnHandle cudnn_handle_;

    cuda_internal::CudaConv cuda_conv_{};
};

DeviceInternals& GetDeviceInternals(CudaDevice& device);

}  // namespace cuda_internal

class CudaDevice : public Device {
public:
    const std::shared_ptr<MemoryPool>& device_memory_pool() { return device_memory_pool_; }

    void Synchronize() override;

    // memory.cc

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    std::shared_ptr<void> MakeDataFromForeignPointer(const std::shared_ptr<void>& data) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::shared_ptr<void> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> TransferDataTo(Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromHostMemory(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    // reduction.cu

    void Sum(const Array& a, const Axes& axis, const Array& out) override;
    void AMax(const Array& a, const Axes& axis, const Array& out) override;

    // activation.cu

    void IfLessElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;

    void IfGreaterElseASSA(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override;
    void IfGreaterElseAAAA(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) override;

    void Tanh(const Array& x, const Array& out) override;

    // exp_log.cu

    void Exp(const Array& x, const Array& out) override;
    void Log(const Array& x, const Array& out) override;

    // misc.cu

    void Square(const Array& x, const Array& out) override;

    void Sqrt(const Array& x, const Array& out) override;

    void IsNan(const Array& x, const Array& out) override;
    void IsInf(const Array& x, const Array& out) override;

    // pool.cc

    std::unique_ptr<MaxPoolForwardBackward> GetMaxPoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            bool cover_all) override;

    std::unique_ptr<AveragePoolForwardBackward> GetAveragePoolForwardBackward(
            const StackVector<int64_t, kMaxNdim>& kernel_size,
            const StackVector<int64_t, kMaxNdim>& stride,
            const StackVector<int64_t, kMaxNdim>& pad,
            AveragePoolPadMode pad_mode) override;

protected:
    CudaDevice(CudaBackend& backend, int index)
        : Device{backend, index},
          device_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<DeviceMemoryAllocator>())},
          pinned_memory_pool_{std::make_shared<MemoryPool>(index, std::make_unique<PinnedMemoryAllocator>())},
          device_internals_{index} {}

private:
    friend CudaDevice* cuda_internal::CreateDevice(CudaBackend&, int);

    friend cuda_internal::DeviceInternals& cuda_internal::GetDeviceInternals(CudaDevice& device);

    friend class cuda_internal::CudaConvTest;  // for unit-tests

    // Allocates pinned memory.
    // The pinned memory is used internally by the CUDA device for asynchronous memory transfer, i.e. cudaMemcpyAsync.
    std::shared_ptr<void> AllocatePinnedMemory(size_t bytesize);

    // Asynchronous transfer from host to this device, w.r.t. host, using temporary pinned memory.
    // The current device must be set to this device, prior to calling this function.
    void MemoryCopyFromHostAsync(void* dst, const void* src, size_t bytesize);

    std::shared_ptr<MemoryPool> device_memory_pool_;

    // TODO(hvy): Consider checking if pinned memory is available by querying canMapHostMemory.
    std::shared_ptr<MemoryPool> pinned_memory_pool_;

    cuda_internal::DeviceInternals device_internals_;

    // Memory keeper.
    cuda_internal::MemoryKeeper memory_keeper_{};
};

}  // namespace cuda
}  // namespace chainerx
