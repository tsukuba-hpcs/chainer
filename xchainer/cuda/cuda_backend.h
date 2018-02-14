#pragma once

#include "xchainer/backend.h"

namespace xchainer {
namespace cuda {

class CudaBackend : public Backend {
public:
    CudaBackend(const std::string& name = "cuda") : Backend(name) {}

    std::shared_ptr<void> Allocate(const Device& device, size_t bytesize) override;
    void MemoryCopy(void* dst_ptr, const void* src_ptr, size_t bytesize) override;
    std::shared_ptr<void> FromBuffer(const Device& device, const std::shared_ptr<void>& src_ptr, size_t bytesize) override;

    void Fill(Array& out, Scalar value) override;

    void Add(const Array& lhs, const Array& rhs, Array& out) override;
    void Mul(const Array& lhs, const Array& rhs, Array& out) override;

    void Synchronize() override;
};

}  // namespace cuda
}  // namespace xchainer
