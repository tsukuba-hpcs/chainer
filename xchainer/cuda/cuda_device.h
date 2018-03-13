#pragma once

#include "xchainer/array.h"
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/device.h"

namespace xchainer {
namespace cuda {

class CudaDevice : public Device {
public:
    CudaDevice(CudaBackend& backend, int index) : Device(backend, index) {}

    std::shared_ptr<void> Allocate(size_t bytesize) override;

    void MemoryCopyFrom(void* dst, const void* src, size_t bytesize, Device& src_device) override;

    void MemoryCopyTo(void* dst, const void* src, size_t bytesize, Device& dst_device) override;

    std::tuple<std::shared_ptr<void>, size_t> TransferDataFrom(
            Device& src_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::tuple<std::shared_ptr<void>, size_t> TransferDataTo(
            Device& dst_device, const std::shared_ptr<void>& src_ptr, size_t offset, size_t bytesize) override;

    std::shared_ptr<void> FromBuffer(const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    void Fill(Array& out, Scalar value) override;

    void Copy(const Array& src, Array& out) override;

    void Add(const Array& lhs, const Array& rhs, Array& out) override;
    void Mul(const Array& lhs, const Array& rhs, Array& out) override;

    void AddAt(const Array& in, const std::vector<ArrayIndex>& indices, Array& out) override;

    void Synchronize() override;
};

}  // namespace cuda
}  // namespace xchainer
