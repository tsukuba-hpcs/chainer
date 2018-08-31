#include "chainerx/cuda/memory_pool.h"

#include <gtest/gtest.h>

#include "chainerx/error.h"

namespace chainerx {
namespace cuda {

TEST(MemoryPoolTest, Malloc) {
    MemoryPool memory_pool{0};

    void* ptr1 = memory_pool.Malloc(1);
    void* ptr2 = memory_pool.Malloc(1);
    EXPECT_NE(ptr1, ptr2);

    memory_pool.Free(ptr2);

    void* ptr3 = memory_pool.Malloc(1);
    EXPECT_EQ(ptr2, ptr3);
    memory_pool.Free(ptr3);

    memory_pool.Free(ptr1);
}

TEST(MemoryPoolTest, AllocationUnitSize) {
    MemoryPool memory_pool{0};

    void* ptr1 = memory_pool.Malloc(100);
    memory_pool.Free(ptr1);

    void* ptr2 = memory_pool.Malloc(100 + kAllocationUnitSize);
    EXPECT_NE(ptr1, ptr2);

    memory_pool.Free(ptr2);
}

TEST(MemoryPoolTest, ZeroByte) {
    MemoryPool memory_pool{0};
    void* ptr = memory_pool.Malloc(0);
    EXPECT_EQ(nullptr, ptr);
    memory_pool.Free(ptr);  // no throw
}

TEST(MemoryPoolTest, DoubleFree) {
    MemoryPool memory_pool{0};
    void* ptr = memory_pool.Malloc(1);
    memory_pool.Free(ptr);
    EXPECT_THROW(memory_pool.Free(ptr), XchainerError);
}

TEST(MemoryPoolTest, FreeForeignPointer) {
    MemoryPool memory_pool{0};
    void* ptr = &memory_pool;
    EXPECT_THROW(memory_pool.Free(ptr), XchainerError);
}

}  // namespace cuda
}  // namespace chainerx
