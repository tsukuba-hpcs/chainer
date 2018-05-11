#include "xchainer/native/native_device.h"

#include <gtest/gtest.h>

#include "xchainer/context.h"
#include "xchainer/native/native_backend.h"

namespace xchainer {
namespace native {
namespace {

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

TEST(NativeDeviceTest, Ctor) {
    Context ctx;
    NativeBackend backend{ctx};
    {
        NativeDevice device{backend, 0};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        NativeDevice device{backend, 1};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST(NativeDeviceTest, Allocate) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    size_t bytesize = 3;
    std::shared_ptr<void> ptr = device.Allocate(bytesize);
    EXPECT_NE(nullptr, ptr);
}

TEST(NativeDeviceTest, CheckMemoryValidity) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    EXPECT_NO_THROW(device.CheckMemoryValidity(&ctx));
}

TEST(NativeDeviceTest, FromHostMemory) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};

    std::shared_ptr<void> dst = device.FromHostMemory(src, bytesize);
    EXPECT_EQ(src.get(), dst.get());
}

TEST(NativeDeviceTest, Synchronize) {
    Context ctx;
    NativeBackend backend{ctx};
    NativeDevice device{backend, 0};
    EXPECT_NO_THROW(device.Synchronize());
}

}  // namespace
}  // namespace native
}  // namespace xchainer
