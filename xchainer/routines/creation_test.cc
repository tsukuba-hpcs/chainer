#include "xchainer/routines/creation.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
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
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{0}, Dtype::kFloat32);
        EXPECT_EQ(1, x.ndim());
        EXPECT_EQ(0, x.GetTotalSize());
        EXPECT_EQ(0, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{1}, Dtype::kFloat32);
        EXPECT_EQ(1, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{2, 3}, Dtype::kFloat32);
        EXPECT_EQ(2, x.ndim());
        EXPECT_EQ(6, x.GetTotalSize());
        EXPECT_EQ(6 * int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{1, 1, 1}, Dtype::kFloat32);
        EXPECT_EQ(3, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Empty(Shape{2, 0, 3}, Dtype::kFloat32);
        EXPECT_EQ(3, x.ndim());
        EXPECT_EQ(0, x.GetTotalSize());
        EXPECT_EQ(0, x.GetTotalBytes());
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

TEST_P(CreationTest, Copy) {
    {
        Array a = testing::BuildArray<bool>({4, 1}, {true, true, false, false});
        Array o = Copy(a);
        testing::ExpectEqualCopy<bool>(a, o);
    }
    {
        Array a = testing::BuildArray<int8_t>({3, 1}, {1, 2, 3});
        Array o = Copy(a);
        testing::ExpectEqualCopy<int8_t>(a, o);
    }
    {
        Array a = testing::BuildArray<float>({3, 1}, {1.0f, 2.0f, 3.0f});
        Array o = Copy(a);
        testing::ExpectEqualCopy<float>(a, o);
    }

    // with padding
    {
        Array a = testing::BuildArray<float>({3, 1}, {1.0f, 2.0f, 3.0f}).WithPadding(1);
        Array o = Copy(a);
        testing::ExpectEqualCopy<float>(a, o);
    }
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
