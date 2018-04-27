#pragma once

#include <vector>
#include <unordered_map>

namespace xchainer {
namespace cuda {

constexpr size_t kAllocationUnitSize = 512;

class MemoryPool {
public:
    explicit MemoryPool(int device_index) : device_index_{device_index} {}

    void* Malloc(size_t bytesize);
    void Free(void* ptr);
private:
    std::unordered_map<void*, size_t> in_use_;
    std::vector<std::vector<void*>> free_bins_;
    int device_index_;
};

}  // namespace cuda
}  // namespace xchainer
