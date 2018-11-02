#include "chainerx/cuda/memory_pool.h"

#include <memory>

#include <gtest/gtest.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {
namespace cuda_internal {

class MemoryPoolTestFriend {
public:
    static const std::vector<std::vector<void*>>& GetFreeBins(const MemoryPool& pool) { return pool.free_bins_; }
};

}  // namespace cuda_internal

class MemoryPoolTest : public ::testing::TestWithParam<std::shared_ptr<MemoryPool>> {};

TEST_P(MemoryPoolTest, Malloc) {
    MemoryPool& memory_pool = *GetParam();

    void* ptr1 = memory_pool.Malloc(1);
    void* ptr2 = memory_pool.Malloc(1);
    EXPECT_NE(ptr1, ptr2);

    memory_pool.Free(ptr2);

    void* ptr3 = memory_pool.Malloc(1);
    EXPECT_EQ(ptr2, ptr3);
    memory_pool.Free(ptr3);

    memory_pool.Free(ptr1);
}

TEST_P(MemoryPoolTest, AllocationUnitSize) {
    MemoryPool& memory_pool = *GetParam();

    void* ptr1 = memory_pool.Malloc(100);
    memory_pool.Free(ptr1);

    void* ptr2 = memory_pool.Malloc(100 + kAllocationUnitSize);
    EXPECT_NE(ptr1, ptr2);

    memory_pool.Free(ptr2);
}

TEST_P(MemoryPoolTest, ZeroByte) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = memory_pool.Malloc(0);
    EXPECT_EQ(nullptr, ptr);
    memory_pool.Free(ptr);  // no throw
}

TEST_P(MemoryPoolTest, DoubleFree) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = memory_pool.Malloc(1);
    memory_pool.Free(ptr);
    EXPECT_THROW(memory_pool.Free(ptr), ChainerxError);
}

TEST_P(MemoryPoolTest, FreeForeignPointer) {
    MemoryPool& memory_pool = *GetParam();
    void* ptr = &memory_pool;
    EXPECT_THROW(memory_pool.Free(ptr), ChainerxError);
}

TEST_P(MemoryPoolTest, FreeAllBlocks) {
    MemoryPool& memory_pool = *GetParam();
    const std::vector<std::vector<void*>>& free_bins = cuda_internal::MemoryPoolTestFriend::GetFreeBins(memory_pool);

    void* ptr1 = memory_pool.Malloc(1);
    memory_pool.Free(ptr1);
    EXPECT_FALSE(free_bins.empty());

    memory_pool.FreeAllBlocks();
    EXPECT_TRUE(free_bins.empty());
}

INSTANTIATE_TEST_CASE_P(
        ForEachAllocator,
        MemoryPoolTest,
        ::testing::Values(
                std::make_shared<MemoryPool>(0, std::make_unique<DeviceMemoryAllocator>()),
                std::make_shared<MemoryPool>(0, std::make_unique<PinnedMemoryAllocator>())));

}  // namespace cuda
}  // namespace chainerx
