#include "xchainer/cuda/cuda_device.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "xchainer/array.h"
#include "xchainer/context.h"
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_runtime.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"
#include "xchainer/testing/util.h"

namespace xchainer {
namespace cuda {
namespace {

template <typename T>
void ExpectDataEqual(const std::shared_ptr<void>& expected, const std::shared_ptr<void>& actual, size_t size) {
    auto expected_raw_ptr = static_cast<const T*>(expected.get());
    auto actual_raw_ptr = static_cast<const T*>(actual.get());
    for (size_t i = 0; i < size; ++i) {
        EXPECT_EQ(expected_raw_ptr[i], actual_raw_ptr[i]);
    }
}

TEST(CudaDeviceTest, Ctor) {
    Context ctx;
    CudaBackend backend{ctx};
    {
        CudaDevice device{backend, 0};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(0, device.index());
    }
    {
        CudaDevice device{backend, 1};
        EXPECT_EQ(&backend, &device.backend());
        EXPECT_EQ(1, device.index());
    }
}

TEST(CudaDeviceTest, Allocate) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    {
        size_t bytesize = 3;
        std::shared_ptr<void> ptr = device.Allocate(bytesize);

        cudaPointerAttributes attr = {};
        CheckCudaError(cudaPointerGetAttributes(&attr, ptr.get()));
        EXPECT_TRUE(attr.isManaged);
        EXPECT_EQ(device.index(), attr.device);
    }
    {
        size_t bytesize = 0;
        EXPECT_NO_THROW(device.Allocate(bytesize));
    }
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    std::shared_ptr<void> cuda_data = device.Allocate(3);
    EXPECT_EQ(cuda_data.get(), device.MakeDataFromForeignPointer(cuda_data).get());
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer_NonCudaMemory) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    std::shared_ptr<void> cpu_data = std::make_unique<uint8_t[]>(3);
    EXPECT_THROW(device.MakeDataFromForeignPointer(cpu_data), XchainerError) << "must throw an exception if non CUDA memory is given";
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer_NonUnifiedMemory) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    void* raw_ptr = nullptr;
    cuda::CheckCudaError(cudaMalloc(&raw_ptr, 3));
    auto cuda_data = std::shared_ptr<void>{raw_ptr, cudaFree};
    EXPECT_THROW(device.MakeDataFromForeignPointer(cuda_data), XchainerError)
            << "must throw an exception if non-managed CUDA memory is given";
}

TEST(CudaDeviceTest, MakeDataFromForeignPointer_FromAnotherDevice) {
    Context ctx;
    CudaBackend backend{ctx};

    XCHAINER_REQUIRE_DEVICE(backend, 2);

    CudaDevice device{backend, 0};
    CudaDevice another_device{backend, 1};

    std::shared_ptr<void> cuda_data = another_device.Allocate(3);
    EXPECT_THROW(device.MakeDataFromForeignPointer(cuda_data), XchainerError)
            << "must throw an exception if CUDA memory resides on another device";
}

TEST(CudaDeviceTest, FromHostMemory) {
    size_t size = 3;
    size_t bytesize = size * sizeof(float);
    float raw_data[] = {0, 1, 2};
    std::shared_ptr<void> src(raw_data, [](const float* ptr) {
        (void)ptr;  // unused
    });

    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};

    std::shared_ptr<void> dst = device.FromHostMemory(src, bytesize);
    ExpectDataEqual<float>(src, dst, size);
    EXPECT_NE(src.get(), dst.get());

    cudaPointerAttributes attr = {};
    CheckCudaError(cudaPointerGetAttributes(&attr, dst.get()));
    EXPECT_TRUE(attr.isManaged);
    EXPECT_EQ(device.index(), attr.device);
}

TEST(CudaDeviceTest, DotNonContiguousOut) {
    testing::DeviceSession session{{"cuda", 0}};
    Array a = testing::BuildArray({2, 3}).WithLinearData(1.f);
    Array b = testing::BuildArray({3, 2}).WithData<float>({1.f, 2.f, -1.f, -3.f, 2.f, 4.f});
    Array c = testing::BuildArray({2, 2}).WithData<float>({0.f, 0.f, 0.f, 0.f}).WithPadding(1);
    a.device().Dot(a, b, c);

    Array e = testing::BuildArray({2, 2}).WithData<float>({5.f, 8.f, 11.f, 17.f});
    testing::ExpectEqual(e, c);
}

// TODO(sonots): Any ways to test cudaDeviceSynchronize()?
TEST(CudaDeviceTest, Synchronize) {
    Context ctx;
    CudaBackend backend{ctx};
    CudaDevice device{backend, 0};
    EXPECT_NO_THROW(device.Synchronize());
}

}  // namespace
}  // namespace cuda
}  // namespace xchainer
