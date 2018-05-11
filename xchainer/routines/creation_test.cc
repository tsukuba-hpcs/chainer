#include "xchainer/routines/creation.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/check_backward.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_backend.h"
#include "xchainer/cuda/cuda_device.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/device_id.h"
#include "xchainer/dtype.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/array_check.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class CreationTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

public:
    template <typename T>
    void CheckEmpty() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Empty(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckEmptyLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Empty(Shape{3, 2}, dtype);
        Array x = EmptyLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullWithGivenDtype(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Full(Shape{3, 2}, scalar, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        testing::ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullWithGivenDtype(T value) {
        CheckFullWithGivenDtype(value, value);
    }

    template <typename T>
    void CheckFullWithScalarDtype(T value) {
        Scalar scalar = {value};
        Array x = Full(Shape{3, 2}, scalar);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), scalar.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        testing::ExpectDataEqual(value, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullLike(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Empty(Shape{3, 2}, dtype);
        Array x = FullLike(x_orig, scalar);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        testing::ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullLike(T value) {
        CheckFullLike(value, value);
    }

    template <typename T>
    void CheckZeros() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Zeros(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{0};
        testing::ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckZerosLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Empty(Shape{3, 2}, dtype);
        Array x = ZerosLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{0};
        testing::ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckOnes() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Ones(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{1};
        testing::ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckOnesLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Empty(Shape{3, 2}, dtype);
        Array x = OnesLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{1};
        testing::ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(CreationTest, FromContiguousHostData) {
    using T = int32_t;
    Shape shape{3, 2};

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<T> data{raw_data, [](const T*) {}};

    Dtype dtype = TypeToDtype<T>;
    Array x = internal::FromContiguousHostData(shape, dtype, data);

    // Basic attributes
    EXPECT_EQ(shape, x.shape());
    EXPECT_EQ(dtype, x.dtype());
    EXPECT_EQ(2, x.ndim());
    EXPECT_EQ(3 * 2, x.GetTotalSize());
    EXPECT_EQ(int64_t{sizeof(T)}, x.item_size());
    EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetNBytes());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(0, x.offset());

    // Array::data
    testing::ExpectDataEqual<T>(data.get(), x);

    Device& device = GetDefaultDevice();
    EXPECT_EQ(&device, &x.device());
    if (device.backend().GetName() == "native") {
        EXPECT_EQ(data.get(), x.data().get());
    } else if (device.backend().GetName() == "cuda") {
        EXPECT_NE(data.get(), x.data().get());
    } else {
        FAIL() << "invalid device_id";
    }
}

TEST_P(CreationTest, FromData) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Device& device = GetDefaultDevice();

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<void> host_data{raw_data, [](const T*) {}};

    // non-contiguous array like a[:,1]
    T sub_raw_data[] = {1, 4};
    Shape shape{2};
    Strides strides{sizeof(T) * 3};
    int64_t offset = sizeof(T);

    Array x;
    void* data_ptr;
    {
        // test potential freed memory
        std::shared_ptr<void> data = device.FromHostMemory(host_data, sizeof(raw_data));
        data_ptr = data.get();
        x = FromData(shape, dtype, data, strides, offset);
    }

    // Basic attributes
    EXPECT_EQ(shape, x.shape());
    EXPECT_EQ(dtype, x.dtype());
    EXPECT_EQ(strides, x.strides());
    EXPECT_EQ(1, x.ndim());
    EXPECT_EQ(2, x.GetTotalSize());
    EXPECT_EQ(int64_t{sizeof(T)}, x.item_size());
    EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetNBytes());
    EXPECT_FALSE(x.IsContiguous());
    EXPECT_EQ(offset, x.offset());

    // Array::data
    testing::ExpectDataEqual<T>(sub_raw_data, x);

    EXPECT_EQ(&device, &x.device());
    EXPECT_EQ(data_ptr, x.data().get());
}

TEST_P(CreationTest, FromDataContiguos) {
    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Device& device = GetDefaultDevice();

    T raw_data[] = {0, 1, 2, 3, 4, 5};
    std::shared_ptr<void> host_data{raw_data, [](const T*) {}};

    // contiguous array like a[1,:]
    T* sub_raw_data = raw_data + 3;
    Shape shape{3};
    Strides strides{sizeof(T)};
    int64_t offset = sizeof(T) * 3;

    Array x;
    void* data_ptr;
    {
        // test potential freed memory
        std::shared_ptr<void> data = device.FromHostMemory(host_data, sizeof(raw_data));
        data_ptr = data.get();
        x = FromData(shape, dtype, data, nonstd::nullopt, offset);
    }

    // Basic attributes
    EXPECT_EQ(shape, x.shape());
    EXPECT_EQ(dtype, x.dtype());
    EXPECT_EQ(strides, x.strides());
    EXPECT_EQ(1, x.ndim());
    EXPECT_EQ(3, x.GetTotalSize());
    EXPECT_EQ(int64_t{sizeof(T)}, x.item_size());
    EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetNBytes());
    EXPECT_TRUE(x.IsContiguous());
    EXPECT_EQ(offset, x.offset());

    // Array::data
    testing::ExpectDataEqual<T>(sub_raw_data, x);

    EXPECT_EQ(&device, &x.device());
    EXPECT_EQ(data_ptr, x.data().get());
}

#ifdef XCHAINER_ENABLE_CUDA
TEST_P(CreationTest, FromDataFromAnotherDevice) {
    Context ctx;
    cuda::CudaBackend cuda_backend{ctx};
    cuda::CudaDevice cuda_device{cuda_backend, 0};

    using T = int32_t;
    Dtype dtype = TypeToDtype<T>;
    Shape shape{3};
    Strides strides{shape, dtype};
    int64_t offset = 0;
    Device& device = GetDefaultDevice();
    std::shared_ptr<void> data = device.Allocate(3 * sizeof(T));

    if (device.name() == cuda_device.name()) {
        EXPECT_NO_THROW(FromData(shape, dtype, data, strides, offset, cuda_device));
    } else {
        EXPECT_THROW(FromData(shape, dtype, data, strides, offset, cuda_device), XchainerError);
    }
}
#endif  // XCHAINER_ENABLE_CUDA

TEST_P(CreationTest, Empty) {
    CheckEmpty<bool>();
    CheckEmpty<int8_t>();
    CheckEmpty<int16_t>();
    CheckEmpty<int32_t>();
    CheckEmpty<int64_t>();
    CheckEmpty<uint8_t>();
    CheckEmpty<float>();
    CheckEmpty<double>();
}

TEST_P(CreationTest, EmptyWithVariousShapes) {
    {
        Array x = Empty(Shape{}, Dtype::kFloat32);
        EXPECT_EQ(0, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{0}, Dtype::kFloat32);
        EXPECT_EQ(1, x.ndim());
        EXPECT_EQ(0, x.GetTotalSize());
        EXPECT_EQ(0, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{1}, Dtype::kFloat32);
        EXPECT_EQ(1, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{2, 3}, Dtype::kFloat32);
        EXPECT_EQ(2, x.ndim());
        EXPECT_EQ(6, x.GetTotalSize());
        EXPECT_EQ(6 * int64_t{sizeof(float)}, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{1, 1, 1}, Dtype::kFloat32);
        EXPECT_EQ(3, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{2, 0, 3}, Dtype::kFloat32);
        EXPECT_EQ(3, x.ndim());
        EXPECT_EQ(0, x.GetTotalSize());
        EXPECT_EQ(0, x.GetNBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
}

TEST_P(CreationTest, EmptyLike) {
    CheckEmptyLike<bool>();
    CheckEmptyLike<int8_t>();
    CheckEmptyLike<int16_t>();
    CheckEmptyLike<int32_t>();
    CheckEmptyLike<int64_t>();
    CheckEmptyLike<uint8_t>();
    CheckEmptyLike<float>();
    CheckEmptyLike<double>();
}

TEST_P(CreationTest, FullWithGivenDtype) {
    CheckFullWithGivenDtype(true);
    CheckFullWithGivenDtype(int8_t{2});
    CheckFullWithGivenDtype(int16_t{2});
    CheckFullWithGivenDtype(int32_t{2});
    CheckFullWithGivenDtype(int64_t{2});
    CheckFullWithGivenDtype(uint8_t{2});
    CheckFullWithGivenDtype(float{2.0f});
    CheckFullWithGivenDtype(double{2.0});

    CheckFullWithGivenDtype(true, Scalar(int32_t{1}));
    CheckFullWithGivenDtype(true, Scalar(int32_t{2}));
    CheckFullWithGivenDtype(true, Scalar(int32_t{-1}));
    CheckFullWithGivenDtype(false, Scalar(int32_t{0}));
}

TEST_P(CreationTest, FullWithScalarDtype) {
    CheckFullWithScalarDtype(true);
    CheckFullWithScalarDtype(int8_t{2});
    CheckFullWithScalarDtype(int16_t{2});
    CheckFullWithScalarDtype(int32_t{2});
    CheckFullWithScalarDtype(int64_t{2});
    CheckFullWithScalarDtype(uint8_t{2});
    CheckFullWithScalarDtype(float{2.0f});
    CheckFullWithScalarDtype(double{2.0});
}

TEST_P(CreationTest, FullLike) {
    CheckFullLike(true);
    CheckFullLike(int8_t{2});
    CheckFullLike(int16_t{2});
    CheckFullLike(int32_t{2});
    CheckFullLike(int64_t{2});
    CheckFullLike(uint8_t{2});
    CheckFullLike(float{2.0f});
    CheckFullLike(double{2.0});

    CheckFullLike(true, Scalar(int32_t{1}));
    CheckFullLike(true, Scalar(int32_t{2}));
    CheckFullLike(true, Scalar(int32_t{-1}));
    CheckFullLike(false, Scalar(int32_t{0}));
}

TEST_P(CreationTest, Zeros) {
    CheckZeros<bool>();
    CheckZeros<int8_t>();
    CheckZeros<int16_t>();
    CheckZeros<int32_t>();
    CheckZeros<int64_t>();
    CheckZeros<uint8_t>();
    CheckZeros<float>();
    CheckZeros<double>();
}

TEST_P(CreationTest, ZerosLike) {
    CheckZerosLike<bool>();
    CheckZerosLike<int8_t>();
    CheckZerosLike<int16_t>();
    CheckZerosLike<int32_t>();
    CheckZerosLike<int64_t>();
    CheckZerosLike<uint8_t>();
    CheckZerosLike<float>();
    CheckZerosLike<double>();
}

TEST_P(CreationTest, Ones) {
    CheckOnes<bool>();
    CheckOnes<int8_t>();
    CheckOnes<int16_t>();
    CheckOnes<int32_t>();
    CheckOnes<int64_t>();
    CheckOnes<uint8_t>();
    CheckOnes<float>();
    CheckOnes<double>();
}

TEST_P(CreationTest, OnesLike) {
    CheckOnesLike<bool>();
    CheckOnesLike<int8_t>();
    CheckOnesLike<int16_t>();
    CheckOnesLike<int32_t>();
    CheckOnesLike<int64_t>();
    CheckOnesLike<uint8_t>();
    CheckOnesLike<float>();
    CheckOnesLike<double>();
}

TEST_P(CreationTest, Arange) {
    Array a = Arange(0, 3, 1);
    Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStopDtype) {
    Array a = Arange(3, Dtype::kInt32);
    Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStopDevice) {
    Array a = Arange(Scalar{3, Dtype::kInt32}, GetDefaultDevice());
    Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStopDtypeDevice) {
    Array a = Arange(3, Dtype::kInt32, GetDefaultDevice());
    Array e = testing::BuildArray({3}).WithData<int32_t>({0, 1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStartStopDtype) {
    Array a = Arange(1, 3, Dtype::kInt32);
    Array e = testing::BuildArray({2}).WithData<int32_t>({1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStartStopDevice) {
    Array a = Arange(1, 3, GetDefaultDevice());
    Array e = testing::BuildArray({2}).WithData<int32_t>({1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStartStopDtypeDevice) {
    Array a = Arange(1, 3, Dtype::kInt32, GetDefaultDevice());
    Array e = testing::BuildArray({2}).WithData<int32_t>({1, 2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStartStopStepDtype) {
    Array a = Arange(1, 7, 2, Dtype::kInt32);
    Array e = testing::BuildArray({3}).WithData<int32_t>({1, 3, 5});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStartStopStepDevice) {
    Array a = Arange(1, 7, 2, GetDefaultDevice());
    Array e = testing::BuildArray({3}).WithData<int32_t>({1, 3, 5});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeStartStopStepDtypeDevice) {
    Array a = Arange(1, 7, 2, Dtype::kInt32, GetDefaultDevice());
    Array e = testing::BuildArray({3}).WithData<int32_t>({1, 3, 5});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeNegativeStep) {
    Array a = Arange(4.f, 0.f, -1.5f, Dtype::kFloat32);
    Array e = testing::BuildArray({3}).WithData<float>({4.f, 2.5f, 1.f});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeLargeStep) {
    Array a = Arange(2, 3, 5, Dtype::kInt32);
    Array e = testing::BuildArray({1}).WithData<int32_t>({2});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeEmpty) {
    Array a = Arange(2, 1, 1, Dtype::kInt32);
    Array e = testing::BuildArray({0}).WithData<int32_t>({});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, ArangeNoDtype) {
    Array a = Arange(Scalar{1, Dtype::kUInt8}, Scalar{4, Dtype::kUInt8}, Scalar{1, Dtype::kUInt8});
    Array e = testing::BuildArray({3}).WithData<uint8_t>({1, 2, 3});
    testing::ExpectEqual(e, a);
}

TEST_P(CreationTest, InvalidTooLongBooleanArange) { EXPECT_THROW(Arange(0, 3, 1, Dtype::kBool), DtypeError); }

TEST_P(CreationTest, Copy) {
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        Array o = Copy(a);
        testing::ExpectEqualCopy(a, o);
    }
    {
        Array a = testing::BuildArray<int8_t>({3, 1}, {1, 2, 3});
        Array o = Copy(a);
        testing::ExpectEqualCopy(a, o);
    }
    {
        Array a = testing::BuildArray<float>({3, 1}, {1.0f, 2.0f, 3.0f});
        Array o = Copy(a);
        testing::ExpectEqualCopy(a, o);
    }

    // with padding
    {
        Array a = testing::BuildArray<float>({3, 1}, {1.0f, 2.0f, 3.0f}).WithPadding(1);
        Array o = Copy(a);
        testing::ExpectEqualCopy(a, o);
    }
}

TEST_P(CreationTest, Identity) {
    Array o = Identity(3, Dtype::kFloat32);
    Array e = testing::BuildArray<float>({3, 3}, {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f});
    testing::ExpectEqual(e, o);
}

TEST_P(CreationTest, IdentityInvalidN) { EXPECT_THROW(Identity(-1, Dtype::kFloat32), DimensionError); }

TEST_P(CreationTest, Eye) {
    {
        Array o = Eye(2, 3, 1, Dtype::kFloat32);
        Array e = testing::BuildArray<float>({2, 3}, {0.f, 1.f, 0.f, 0.f, 0.f, 1.f});
        testing::ExpectEqual(e, o);
    }
    {
        Array o = Eye(3, 2, -2, Dtype::kFloat32);
        Array e = testing::BuildArray<float>({3, 2}, {0.f, 0.f, 0.f, 0.f, 1.f, 0.f});
        testing::ExpectEqual(e, o);
    }
}

TEST_P(CreationTest, EyeInvalidNM) {
    EXPECT_THROW(Eye(-1, 2, 1, Dtype::kFloat32), DimensionError);
    EXPECT_THROW(Eye(1, -2, 1, Dtype::kFloat32), DimensionError);
    EXPECT_THROW(Eye(-1, -2, 1, Dtype::kFloat32), DimensionError);
}

TEST_P(CreationTest, AsContiguousArray) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<int32_t>().WithPadding(1);
    ASSERT_FALSE(a.IsContiguous());  // test precondition
    Array b = AsContiguousArray(a);

    EXPECT_TRUE(b.IsContiguous());
    testing::ExpectEqual(b, a);
}

TEST_P(CreationTest, AsContiguousArrayNoCopy) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<int32_t>();
    ASSERT_TRUE(a.IsContiguous());  // test precondition
    Array b = AsContiguousArray(a);

    EXPECT_EQ(b.body(), a.body());
}

TEST_P(CreationTest, AsContiguousArrayDtypeMismatch) {
    Array a = testing::BuildArray({2, 3}).WithLinearData<int32_t>();
    ASSERT_TRUE(a.IsContiguous());  // test precondition
    Array b = AsContiguousArray(a, Dtype::kInt64);

    EXPECT_NE(b.body(), a.body());
    EXPECT_TRUE(b.IsContiguous());
    EXPECT_EQ(Dtype::kInt64, b.dtype());
    testing::ExpectEqual(b, a.AsType(Dtype::kInt64));
}

TEST_P(CreationTest, AsContiguousArrayBackward) {
    CheckBackward(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {AsContiguousArray(xs[0])}; },
            {(*testing::BuildArray({2, 3}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {testing::BuildArray({2, 3}).WithLinearData<float>(-2.4f, 0.8f)},
            {Full({2, 3}, 1e-1f)});
}

TEST_P(CreationTest, AsContiguousArrayDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = AsContiguousArray(xs[0]);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::BuildArray({2, 3}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {(*testing::BuildArray({2, 3}).WithLinearData<float>(-2.4f, 0.8f)).RequireGrad()},
            {testing::BuildArray({2, 3}).WithLinearData<float>(5.2f, -0.5f)},
            {Full({2, 3}, 1e-1f), Full({2, 3}, 1e-1f)});
}

TEST_P(CreationTest, DiagVecToMat) {
    {
        Array v = Arange(1, 3, Dtype::kFloat32);
        Array o = Diag(v);
        Array e = testing::BuildArray<float>({2, 2}, {1.f, 0.f, 0.f, 2.f});
        testing::ExpectEqual(e, o);
    }
    {
        Array v = Arange(1, 4, Dtype::kFloat32);
        Array o = Diag(v, 1);
        Array e = testing::BuildArray<float>({4, 4}, {0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 3.f, 0.f, 0.f, 0.f, 0.f});
        testing::ExpectEqual(e, o);
    }
    {
        Array v = Arange(1, 3, Dtype::kFloat32);
        Array o = Diag(v, -2);
        Array e = testing::BuildArray<float>({4, 4}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f});
        testing::ExpectEqual(e, o);
    }
}

TEST_P(CreationTest, DiagMatToVec) {
    {
        Array v = Arange(6, Dtype::kFloat32).Reshape({2, 3});
        Array o = Diag(v);
        Array e = testing::BuildArray<float>({2}, {0.f, 4.f});
        testing::ExpectEqual(e, o);
        EXPECT_EQ(v.data().get(), o.data().get());
    }
    {
        Array v = Arange(6, Dtype::kFloat32).Reshape({2, 3});
        Array o = Diag(v, 1);
        Array e = testing::BuildArray<float>({2}, {1.f, 5.f});
        testing::ExpectEqual(e, o);
        EXPECT_EQ(v.data().get(), o.data().get());
    }
    {
        Array v = Arange(6, Dtype::kFloat32).Reshape({2, 3});
        Array o = Diag(v, -1);
        Array e = testing::BuildArray<float>({1}, {3.f});
        testing::ExpectEqual(e, o);
        EXPECT_EQ(v.data().get(), o.data().get());
    }
}

TEST_P(CreationTest, DiagVecToMatBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({3}, 1e-3);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Diag(xs[0], -1)}; }, {v}, {go}, {eps});
}

TEST_P(CreationTest, DiagMatToVecBackward) {
    using T = double;
    Array v = (*testing::BuildArray({4, 4}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({4, 4}, 1e-3);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Diag(xs[0], 1)}; }, {v}, {go}, {eps});
}

TEST_P(CreationTest, DiagVecToMatDoubleBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggv = testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_v = Full(Shape{3}, 1e-3);
    Array eps_go = Full(Shape{4, 4}, 1e-3);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Diag(xs[0], -1);
                return {y * y};  // to make it nonlinear
            },
            {v},
            {go},
            {ggv},
            {eps_v, eps_go});
}

TEST_P(CreationTest, DiagMatToVecDoubleBackward) {
    using T = double;
    Array v = (*testing::BuildArray(Shape{4, 4}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray(Shape{3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggv = testing::BuildArray(Shape{4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_v = Full(Shape{4, 4}, 1e-3);
    Array eps_go = Full(Shape{3}, 1e-3);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Diag(xs[0], -1);
                return {y * y};  // to make it nonlinear
            },
            {v},
            {go},
            {ggv},
            {eps_v, eps_go});
}

TEST_P(CreationTest, Diagflat) {
    {
        Array v = Arange(1, 3, Dtype::kFloat32);
        Array o = Diagflat(v);
        Array e = testing::BuildArray<float>({2, 2}, {1.f, 0.f, 0.f, 2.f});
        testing::ExpectEqual(e, o);
    }
    {
        Array v = Arange(1, 5, Dtype::kFloat32).Reshape({2, 2});
        Array o = Diagflat(v, 1);
        Array e = testing::BuildArray<float>({5, 5}, {0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 2.f, 0.f, 0.f, 0.f, 0.f, 0.f,
                                                      3.f, 0.f, 0.f, 0.f, 0.f, 0.f, 4.f, 0.f, 0.f, 0.f, 0.f, 0.f});
        testing::ExpectEqual(e, o);
    }
    {
        Array v = Arange(1, 3, Dtype::kFloat32).Reshape({1, 2});
        Array o = Diagflat(v, -1);
        Array e = testing::BuildArray<float>({3, 3}, {0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 2.f, 0.f});
        testing::ExpectEqual(e, o);
    }
}

TEST_P(CreationTest, DiagflatBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps = Full({3}, 1e-3);

    CheckBackward([](const std::vector<Array>& xs) -> std::vector<Array> { return {Diagflat(xs[0], 1)}; }, {v}, {go}, {eps});
}

TEST_P(CreationTest, DiagflatDoubleBackward) {
    using T = double;
    Array v = (*testing::BuildArray({3}).WithLinearData<T>(-3).WithPadding(1)).RequireGrad();
    Array go = (*testing::BuildArray({4, 4}).WithLinearData<T>(-0.1, 0.1).WithPadding(1)).RequireGrad();
    Array ggv = testing::BuildArray({3}).WithLinearData<T>(-0.1, 0.1).WithPadding(1);
    Array eps_v = Full(Shape{3}, 1e-3);
    Array eps_go = Full(Shape{4, 4}, 1e-3);

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = Diagflat(xs[0], -1);
                return {y * y};  // to make it nonlinear
            },
            {v},
            {go},
            {ggv},
            {eps_v, eps_go});
}

TEST_P(CreationTest, Linspace) {
    Array o = Linspace(3.0, 10.0, 4, true, Dtype::kInt32);
    Array e = testing::BuildArray<int32_t>({4}, {3, 5, 7, 10});
    testing::ExpectEqual(e, o);
}

TEST_P(CreationTest, LinspaceEndPointFalse) {
    Array o = Linspace(3.0, 10.0, 4, false, Dtype::kInt32);
    Array e = testing::BuildArray<int32_t>({4}, {3, 4, 6, 8});
    testing::ExpectEqual(e, o);
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        CreationTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
