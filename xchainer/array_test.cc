#include "xchainer/array.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/array_node.h"
#include "xchainer/backend.h"
#include "xchainer/check_backward.h"
#include "xchainer/context.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native_backend.h"
#include "xchainer/op_node.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    void SetUp() override {
        const std::string& backend_name = ::testing::get<0>(GetParam());
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

public:
    template <typename T>
    void ExpectEqualCopy(const Array& expected, const Array& actual) {
        EXPECT_EQ(expected.dtype(), actual.dtype());
        EXPECT_EQ(expected.shape(), actual.shape());
        EXPECT_EQ(&expected.device(), &actual.device());

        // Deep copy, therefore assert different addresses to data
        EXPECT_NE(expected.data().get(), actual.data().get());

        EXPECT_TRUE(actual.IsContiguous());
        EXPECT_EQ(0, actual.offset());

        ExpectDataEqual<T>(expected, actual);
    }

    template <typename T>
    void ExpectEqualView(const Array& expected, const Array& actual) {
        ExpectEqual<T>(expected, actual);
        ExpectArraysEqualAttributes(expected, actual);

        // Shallow copy, therefore assert the same address to data
        EXPECT_EQ(expected.data().get(), actual.data().get());
        EXPECT_EQ(&expected.device(), &actual.device());

        // Views should have different array bodies.
        EXPECT_NE(expected.body(), actual.body());
    }

    template <typename T>
    void ExpectEqual(const Array& expected, const Array& actual) {
        EXPECT_EQ(expected.dtype(), actual.dtype());
        EXPECT_EQ(expected.shape(), actual.shape());
        EXPECT_EQ(&expected.device(), &actual.device());
        ExpectDataEqual<T>(expected, actual);
    }

    template <typename T>
    void ExpectDataEqual(const Array& expected, const Array& actual) {
        actual.device().Synchronize();
        IndexableArray<const T> expected_iarray{expected};
        IndexableArray<const T> actual_iarray{actual};
        Indexer<> indexer{actual.shape()};
        for (int64_t i = 0; i < indexer.total_size(); i++) {
            indexer.Set(i);
            const auto& expected = expected_iarray[indexer];
            const auto& actual = actual_iarray[indexer];
            if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(actual)) << "where i is " << i;
            } else {
                EXPECT_EQ(expected, actual) << "where i is " << i;
            }
        }
    }

    template <typename T>
    void ExpectDataEqual(const T* expected_data, const Array& actual) {
        actual.device().Synchronize();
        auto total_size = actual.shape().GetTotalSize();
        const T* actual_data = static_cast<const T*>(actual.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            EXPECT_EQ(expected_data[i], actual_data[i]) << "where i is " << i;
        }
    }

    template <typename T>
    void ExpectDataEqual(T expected, const Array& actual) {
        actual.device().Synchronize();
        auto total_size = actual.shape().GetTotalSize();
        const T* actual_data = static_cast<const T*>(actual.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            if (std::isnan(expected)) {
                EXPECT_TRUE(std::isnan(actual_data[i])) << "where i is " << i;
            } else {
                EXPECT_EQ(expected, actual_data[i]) << "where i is " << i;
            }
        }
    }

    void ExpectArraysEqualAttributes(const Array& a, const Array& b) {
        EXPECT_EQ(a.dtype(), b.dtype());
        EXPECT_EQ(a.shape(), b.shape());
        EXPECT_EQ(a.IsContiguous(), b.IsContiguous());
        EXPECT_EQ(a.offset(), b.offset());
    }

    template <bool is_const, typename T>
    void CheckFromBuffer(const Shape& shape, std::initializer_list<T> raw_data) {
        using TargetArray = std::conditional_t<is_const, const Array, Array>;

        // Check test data
        ASSERT_EQ(shape.GetTotalSize(), static_cast<int64_t>(raw_data.size()));

        std::shared_ptr<T> data = std::make_unique<T[]>(shape.GetTotalSize());
        std::copy(raw_data.begin(), raw_data.end(), data.get());

        Dtype dtype = TypeToDtype<T>;
        TargetArray x = Array::FromBuffer(shape, dtype, data);

        // Basic attributes
        EXPECT_EQ(shape, x.shape());
        EXPECT_EQ(dtype, x.dtype());
        EXPECT_EQ(2, x.ndim());
        EXPECT_EQ(3 * 2, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(T)}, x.element_bytes());
        EXPECT_EQ(shape.GetTotalSize() * int64_t{sizeof(T)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());

        // Array::data
        ExpectDataEqual<T>(data.get(), x);

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

    template <typename T>
    void CheckEmpty() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
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
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::EmptyLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFill(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(scalar);
        ExpectDataEqual(expected, x);
    }

    template <typename T>
    void CheckFill(T value) {
        CheckFill(value, value);
    }

    template <typename T>
    void CheckFullWithGivenDtype(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Full(Shape{3, 2}, scalar, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullWithGivenDtype(T value) {
        CheckFullWithGivenDtype(value, value);
    }

    template <typename T>
    void CheckFullWithScalarDtype(T value) {
        Scalar scalar = {value};
        Array x = Array::Full(Shape{3, 2}, scalar);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), scalar.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataEqual(value, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullLike(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::FullLike(x_orig, scalar);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckFullLike(T value) {
        CheckFullLike(value, value);
    }

    template <typename T>
    void CheckZeros() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Zeros(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{0};
        ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckZerosLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::ZerosLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{0};
        ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckOnes() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Ones(Shape{3, 2}, dtype);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_EQ(x.shape(), Shape({3, 2}));
        EXPECT_EQ(x.dtype(), dtype);
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{1};
        ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

    template <typename T>
    void CheckOnesLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::OnesLike(x_orig);
        EXPECT_NE(x.data(), nullptr);
        EXPECT_NE(x.data(), x_orig.data());
        EXPECT_EQ(x.shape(), x_orig.shape());
        EXPECT_EQ(x.dtype(), x_orig.dtype());
        EXPECT_TRUE(x.IsContiguous());
        EXPECT_EQ(0, x.offset());
        T expected{1};
        ExpectDataEqual(expected, x);
        EXPECT_EQ(&GetDefaultDevice(), &x.device());
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(ArrayTest, CopyCtor) {
    Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = a;

    // A copy-constructed instance must be a view
    {
        ExpectEqualView<bool>(a, b);
        EXPECT_THROW(internal::GetArrayNode(a), XchainerError);
        EXPECT_THROW(internal::GetArrayNode(b), XchainerError);
    }

    // A view must not share requires_grad with the original array.
    {
        // Precondition of the test
        ASSERT_FALSE(a.IsGradRequired());
        ASSERT_FALSE(b.IsGradRequired());

        a.RequireGrad();
        EXPECT_NE(a.IsGradRequired(), b.IsGradRequired());
    }
}

TEST_P(ArrayTest, ArrayMoveCtor) {
    { EXPECT_TRUE(std::is_nothrow_move_constructible<Array>::value); }

    // A view must not be affected by move
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = a;  // view
        Array c = std::move(a);
        ExpectEqual<float>(b, c);
    }

    // A copy must not be affected by move
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = a.Copy();  // copy
        Array c = std::move(a);
        ExpectEqualCopy<float>(b, c);
    }

    // Array body must be transferred by move
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        auto body = a.body();
        Array c = std::move(a);
        EXPECT_EQ(body, c.body());
    }
}

TEST_P(ArrayTest, ArrayBodyCtor) {
    Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
    auto body = a.body();
    Array b{body};
    EXPECT_EQ(body, b.body());
    ExpectArraysEqualAttributes(a, b);
    EXPECT_EQ(a.data(), b.data());
    EXPECT_THROW(internal::GetArrayNode(a), XchainerError);
    EXPECT_THROW(internal::GetArrayNode(b), XchainerError);
}

TEST_P(ArrayTest, SetRequiresGrad) {
    // Default graph
    {
        Array x = testing::MakeArray<bool>({1}, {true});
        ASSERT_FALSE(x.IsGradRequired());
        x.RequireGrad();
        ASSERT_TRUE(x.IsGradRequired());
    }

    // User-specified graph
    {
        GraphId graph_id = "graph_1";
        Array x = testing::MakeArray<bool>({1}, {true});
        ASSERT_FALSE(x.IsGradRequired(graph_id));
        x.RequireGrad(graph_id);
        ASSERT_TRUE(x.IsGradRequired(graph_id));
    }
}

TEST_P(ArrayTest, Grad) {
    GraphId graph_id = "graph_1";
    Shape shape{2, 3};
    using T = float;

    Array x = testing::MakeArray<T>(shape, {5, 3, 2, 1, 4, 6});
    Array g = testing::MakeArray<T>(shape, {8, 4, 6, 3, 2, 1});

    x.RequireGrad(graph_id);
    g.RequireGrad(graph_id);

    EXPECT_FALSE(x.GetGrad(graph_id)) << "grad must be initially unset";

    // Set and get grad
    {
        x.SetGrad(g, graph_id);

        ExpectEqual<T>(g, *x.GetGrad(graph_id));
    }

    // Get grad multiple times
    {
        const nonstd::optional<Array>& grad1 = x.GetGrad(graph_id);
        const nonstd::optional<Array>& grad2 = x.GetGrad(graph_id);
        EXPECT_EQ(&*grad1, &*grad2) << "Multiple retrieval of grad must return the same arrays";
    }

    // ClearGrad
    {
        Array grad_view = *x.GetGrad(graph_id);  // Make a view of grad

        x.ClearGrad(graph_id);

        EXPECT_FALSE(x.GetGrad(graph_id)) << "grad must be cleared after calling ClearGrad()";

        // ClearGrad() must not affect previously retrieved view to grad
        ExpectEqual<T>(grad_view, g);
    }
}

TEST_P(ArrayTest, ArrayFromBuffer) {
    Shape shape{3, 2};
    CheckFromBuffer<false, bool>(shape, {true, false, false, true, false, true});
    CheckFromBuffer<false, int8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, int16_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, int32_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, int64_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, uint8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, float>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<false, double>(shape, {0, 1, 2, 3, 4, 5});
}

TEST_P(ArrayTest, ConstArrayFromBuffer) {
    Shape shape{3, 2};
    CheckFromBuffer<true, bool>(shape, {true, false, false, true, false, true});
    CheckFromBuffer<true, int8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, int16_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, int32_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, int64_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, uint8_t>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, float>(shape, {0, 1, 2, 3, 4, 5});
    CheckFromBuffer<true, double>(shape, {0, 1, 2, 3, 4, 5});
}

TEST_P(ArrayTest, Empty) {
    CheckEmpty<bool>();
    CheckEmpty<int8_t>();
    CheckEmpty<int16_t>();
    CheckEmpty<int32_t>();
    CheckEmpty<int64_t>();
    CheckEmpty<uint8_t>();
    CheckEmpty<float>();
    CheckEmpty<double>();
}

TEST_P(ArrayTest, EmptyWithVariousShapes) {
    {
        Array x = Array::Empty(Shape{}, Dtype::kFloat32);
        EXPECT_EQ(0, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Array::Empty(Shape{0}, Dtype::kFloat32);
        EXPECT_EQ(1, x.ndim());
        EXPECT_EQ(0, x.GetTotalSize());
        EXPECT_EQ(0, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Array::Empty(Shape{1}, Dtype::kFloat32);
        EXPECT_EQ(1, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Array::Empty(Shape{2, 3}, Dtype::kFloat32);
        EXPECT_EQ(2, x.ndim());
        EXPECT_EQ(6, x.GetTotalSize());
        EXPECT_EQ(6 * int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Array::Empty(Shape{1, 1, 1}, Dtype::kFloat32);
        EXPECT_EQ(3, x.ndim());
        EXPECT_EQ(1, x.GetTotalSize());
        EXPECT_EQ(int64_t{sizeof(float)}, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
    {
        Array x = Array::Empty(Shape{2, 0, 3}, Dtype::kFloat32);
        EXPECT_EQ(3, x.ndim());
        EXPECT_EQ(0, x.GetTotalSize());
        EXPECT_EQ(0, x.GetTotalBytes());
        EXPECT_TRUE(x.IsContiguous());
    }
}

TEST_P(ArrayTest, EmptyLike) {
    CheckEmptyLike<bool>();
    CheckEmptyLike<int8_t>();
    CheckEmptyLike<int16_t>();
    CheckEmptyLike<int32_t>();
    CheckEmptyLike<int64_t>();
    CheckEmptyLike<uint8_t>();
    CheckEmptyLike<float>();
    CheckEmptyLike<double>();
}

TEST_P(ArrayTest, Fill) {
    CheckFill(true);
    CheckFill(false);
    CheckFill(int8_t{0});
    CheckFill(int8_t{-1});
    CheckFill(int8_t{5});
    CheckFill(int8_t{-128});
    CheckFill(int8_t{127});
    CheckFill(int16_t{0});
    CheckFill(int16_t{-3});
    CheckFill(int32_t{0});
    CheckFill(int32_t{-3});
    CheckFill(int64_t{0});
    CheckFill(int64_t{-3});
    CheckFill(uint8_t{0});
    CheckFill(uint8_t{255});
    CheckFill(float{0});
    CheckFill(float{std::numeric_limits<float>::infinity()});
    CheckFill(float{std::nanf("")});
    CheckFill(double{0});
    CheckFill(double{std::numeric_limits<double>::infinity()});
    CheckFill(double{std::nan("")});

    CheckFill(true, Scalar(int32_t{1}));
    CheckFill(true, Scalar(int32_t{2}));
    CheckFill(true, Scalar(int32_t{-1}));
    CheckFill(false, Scalar(int32_t{0}));
    CheckFill(int8_t{1}, Scalar(int32_t{1}));
    CheckFill(int8_t{1}, Scalar(int64_t{1}));
    CheckFill(int8_t{1}, Scalar(uint8_t{1}));
    CheckFill(int8_t{1}, Scalar(true));
    CheckFill(int8_t{1}, Scalar(1.0f));
    CheckFill(int8_t{1}, Scalar(1.0));
    CheckFill(int16_t{1}, Scalar(int32_t{1}));
    CheckFill(int16_t{1}, Scalar(int64_t{1}));
    CheckFill(int16_t{1}, Scalar(uint8_t{1}));
    CheckFill(int16_t{1}, Scalar(true));
    CheckFill(int16_t{1}, Scalar(1.0f));
    CheckFill(int16_t{1}, Scalar(1.0));
    CheckFill(int32_t{1}, Scalar(int32_t{1}));
    CheckFill(int32_t{1}, Scalar(int64_t{1}));
    CheckFill(int32_t{1}, Scalar(uint8_t{1}));
    CheckFill(int32_t{1}, Scalar(true));
    CheckFill(int32_t{1}, Scalar(1.0f));
    CheckFill(int32_t{1}, Scalar(1.0));
    CheckFill(int64_t{1}, Scalar(int32_t{1}));
    CheckFill(int64_t{1}, Scalar(int64_t{1}));
    CheckFill(int64_t{1}, Scalar(uint8_t{1}));
    CheckFill(int64_t{1}, Scalar(true));
    CheckFill(int64_t{1}, Scalar(1.0f));
    CheckFill(int64_t{1}, Scalar(1.0));
    CheckFill(uint8_t{1}, Scalar(int32_t{1}));
    CheckFill(uint8_t{1}, Scalar(int64_t{1}));
    CheckFill(uint8_t{1}, Scalar(uint8_t{1}));
    CheckFill(uint8_t{1}, Scalar(true));
    CheckFill(uint8_t{1}, Scalar(1.0f));
    CheckFill(uint8_t{1}, Scalar(1.0));
    CheckFill(float{1}, Scalar(int32_t{1}));
    CheckFill(float{1}, Scalar(int64_t{1}));
    CheckFill(float{1}, Scalar(uint8_t{1}));
    CheckFill(float{1}, Scalar(true));
    CheckFill(float{1}, Scalar(1.0f));
    CheckFill(float{1}, Scalar(1.0));
    CheckFill(double{1}, Scalar(int32_t{1}));
    CheckFill(double{1}, Scalar(int64_t{1}));
    CheckFill(double{1}, Scalar(uint8_t{1}));
    CheckFill(double{1}, Scalar(true));
    CheckFill(double{1}, Scalar(1.0f));
    CheckFill(double{1}, Scalar(1.0));
}

TEST_P(ArrayTest, FullWithGivenDtype) {
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

TEST_P(ArrayTest, FullWithScalarDtype) {
    CheckFullWithScalarDtype(true);
    CheckFullWithScalarDtype(int8_t{2});
    CheckFullWithScalarDtype(int16_t{2});
    CheckFullWithScalarDtype(int32_t{2});
    CheckFullWithScalarDtype(int64_t{2});
    CheckFullWithScalarDtype(uint8_t{2});
    CheckFullWithScalarDtype(float{2.0f});
    CheckFullWithScalarDtype(double{2.0});
}

TEST_P(ArrayTest, FullLike) {
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

TEST_P(ArrayTest, Zeros) {
    CheckZeros<bool>();
    CheckZeros<int8_t>();
    CheckZeros<int16_t>();
    CheckZeros<int32_t>();
    CheckZeros<int64_t>();
    CheckZeros<uint8_t>();
    CheckZeros<float>();
    CheckZeros<double>();
}

TEST_P(ArrayTest, ZerosLike) {
    CheckZerosLike<bool>();
    CheckZerosLike<int8_t>();
    CheckZerosLike<int16_t>();
    CheckZerosLike<int32_t>();
    CheckZerosLike<int64_t>();
    CheckZerosLike<uint8_t>();
    CheckZerosLike<float>();
    CheckZerosLike<double>();
}

TEST_P(ArrayTest, Ones) {
    CheckOnes<bool>();
    CheckOnes<int8_t>();
    CheckOnes<int16_t>();
    CheckOnes<int32_t>();
    CheckOnes<int64_t>();
    CheckOnes<uint8_t>();
    CheckOnes<float>();
    CheckOnes<double>();
}

TEST_P(ArrayTest, OnesLike) {
    CheckOnesLike<bool>();
    CheckOnesLike<int8_t>();
    CheckOnesLike<int16_t>();
    CheckOnesLike<int32_t>();
    CheckOnesLike<int64_t>();
    CheckOnesLike<uint8_t>();
    CheckOnesLike<float>();
    CheckOnesLike<double>();
}

TEST_P(ArrayTest, IAdd) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = testing::MakeArray<bool>({4, 1}, {true, true, true, false});
        a += b;
        ExpectEqual<bool>(e, a);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<int8_t>({3, 1}, {2, 4, 6});
        a += b;
        ExpectEqual<int8_t>(e, a);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<float>({3, 1}, {2, 4, 6});
        a += b;
        ExpectEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, IMul) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = testing::MakeArray<bool>({4, 1}, {true, false, false, false});
        a *= b;
        ExpectEqual<bool>(e, a);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<int8_t>({3, 1}, {1, 4, 9});
        a *= b;
        ExpectEqual<int8_t>(e, a);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<float>({3, 1}, {1, 4, 9});
        a *= b;
        ExpectEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, Add) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = testing::MakeArray<bool>({4, 1}, {true, true, true, false});
        Array o = a + b;
        ExpectEqual<bool>(e, o);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<int8_t>({3, 1}, {2, 4, 6});
        Array o = a + b;
        ExpectEqual<int8_t>(e, o);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<float>({3, 1}, {2, 4, 6});
        Array o = a + b;
        ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, Mul) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = testing::MakeArray<bool>({4, 1}, {true, false, false, false});
        Array o = a * b;
        ExpectEqual<bool>(e, o);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<int8_t>({3, 1}, {1, 4, 9});
        Array o = a * b;
        ExpectEqual<int8_t>(e, o);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<float>({3, 1}, {1, 4, 9});
        Array o = a * b;
        ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedMath) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array c = a * b;
        Array o = a + c;
        ExpectEqual<bool>(e, o);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<int8_t>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        ExpectEqual<int8_t>(e, o);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<float>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        ExpectEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedInplaceMath) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        b *= a;
        a += b;
        ExpectEqual<bool>(e, a);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<int8_t>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        ExpectEqual<int8_t>(e, a);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = testing::MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = testing::MakeArray<float>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        ExpectEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, ComputationalGraph) {
    // c = a + b
    // o = a * c
    Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});

    GraphId graph_id = "graph_1";
    a.RequireGrad(graph_id);
    b.RequireGrad(graph_id);

    {
        auto a_node = internal::GetArrayNode(a, graph_id);
        auto b_node = internal::GetArrayNode(b, graph_id);
        EXPECT_NE(a_node, nullptr);
        EXPECT_NE(b_node, nullptr);
        auto a_op_node = a_node->next_node();
        auto b_op_node = b_node->next_node();
        EXPECT_EQ(a_op_node, nullptr);
        EXPECT_EQ(b_op_node, nullptr);
    }

    Array c = a + b;
    {
        auto a_node = internal::GetArrayNode(a, graph_id);
        auto b_node = internal::GetArrayNode(b, graph_id);
        auto c_node = internal::GetArrayNode(c, graph_id);
        EXPECT_NE(a_node, nullptr);
        EXPECT_NE(b_node, nullptr);
        EXPECT_NE(c_node, nullptr);
        auto a_op_node = a_node->next_node();
        auto b_op_node = b_node->next_node();
        auto c_op_node = c_node->next_node();
        EXPECT_EQ(a_op_node, nullptr);
        EXPECT_EQ(b_op_node, nullptr);
        EXPECT_NE(c_op_node, nullptr);
        EXPECT_EQ(c_op_node->name(), "add");
    }

    Array o = a * c;
    {
        auto a_node = internal::GetArrayNode(a, graph_id);
        auto b_node = internal::GetArrayNode(b, graph_id);
        auto c_node = internal::GetArrayNode(c, graph_id);
        auto o_node = internal::GetArrayNode(o, graph_id);
        EXPECT_NE(a_node, nullptr);
        EXPECT_NE(b_node, nullptr);
        EXPECT_NE(c_node, nullptr);
        EXPECT_NE(o_node, nullptr);
        auto a_op_node = a_node->next_node();
        auto b_op_node = b_node->next_node();
        auto c_op_node = c_node->next_node();
        auto o_op_node = o_node->next_node();
        EXPECT_EQ(a_op_node, nullptr);
        EXPECT_EQ(b_op_node, nullptr);
        EXPECT_NE(c_op_node, nullptr);
        EXPECT_NE(o_op_node, nullptr);
        EXPECT_EQ(c_op_node->name(), "add");
        EXPECT_EQ(o_op_node->name(), "mul");
    }
}

TEST_P(ArrayTest, InplaceNotAllowedWithRequiresGrad) {
    GraphId graph_id = "graph_1";
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        a.RequireGrad(graph_id);
        b.RequireGrad(graph_id);
        EXPECT_THROW({ a += b; }, XchainerError);
    }

    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        a.RequireGrad(graph_id);
        b.RequireGrad(graph_id);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }

    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        a.RequireGrad(graph_id);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }

    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});
        b.RequireGrad(graph_id);
        EXPECT_THROW({ a *= b; }, XchainerError);
    }
}

TEST_P(ArrayTest, Transpose) {
    Array a = testing::MakeArray<int32_t>({2, 3}, {0, 1, 2, 3, 4, 5});
    Array b = a.Transpose();

    EXPECT_EQ(Shape({3, 2}), b.shape());
    EXPECT_EQ(Strides({4, 12}), b.strides());

    Array e = testing::MakeArray<int32_t>({3, 2}, {0, 3, 1, 4, 2, 5});
    ExpectEqual<int32_t>(e, b);
}

TEST_P(ArrayTest, TransposeBackward) {
    CheckBackwardComputation([](const std::vector<Array>& xs) -> std::vector<Array> { return {xs[0].Transpose()}; },
                             {Array::Zeros({2, 3}, Dtype::kFloat32).RequireGrad()},
                             {testing::MakeArray({3, 2}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f})},
                             {Array::Full({2, 3}, 1e-5f)});
}

TEST_P(ArrayTest, TransposeDoubleBackward) {
    CheckDoubleBackwardComputation(
        [](const std::vector<Array>& xs) -> std::vector<Array> {
            auto t = xs[0].Transpose();
            return {t * t};  // to make it nonlinear
        },
        {testing::MakeArray({2, 3}, {1.f, -1.f, 2.f, -2.f, 3.f, -3.f}).RequireGrad()},
        {Array::Ones({3, 2}, Dtype::kFloat32).RequireGrad()},
        {Array::Ones({2, 3}, Dtype::kFloat32)},
        {Array::Full({2, 3}, 0.01f), Array::Full({3, 2}, 0.01f)});
}

TEST_P(ArrayTest, Copy) {
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        Array o = a.Copy();
        ExpectEqualCopy<bool>(a, o);
    }
    {
        Array a = testing::MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array o = a.Copy();
        ExpectEqualCopy<int8_t>(a, o);
    }
    {
        Array a = testing::MakeArray<float>({3, 1}, {1.0f, 2.0f, 3.0f});
        Array o = a.Copy();
        ExpectEqualCopy<float>(a, o);
    }
}

TEST_P(ArrayTest, AsConstantCopy) {
    // Stop gradients on all graphs
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        Array b = a.AsConstant(CopyKind::kCopy);

        EXPECT_EQ(&b.device(), &a.device());

        ExpectEqualCopy<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
    }

    // Stop gradients on graphs
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        a.RequireGrad("graph_3");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        ASSERT_TRUE(a.IsGradRequired("graph_3"));
        Array b = a.AsConstant({"graph_1", "graph_2"}, CopyKind::kCopy);

        EXPECT_EQ(&b.device(), &a.device());

        ExpectEqualCopy<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));
        EXPECT_TRUE(b.IsGradRequired("graph_3"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
        EXPECT_TRUE(a.IsGradRequired("graph_3"));
    }
}

TEST_P(ArrayTest, AsConstantView) {
    // Stop gradients on all graphs
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        Array b = a.AsConstant();

        ExpectEqualView<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
    }

    // Stop gradients on some graphs
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
        a.RequireGrad("graph_1");
        a.RequireGrad("graph_2");
        a.RequireGrad("graph_3");
        ASSERT_TRUE(a.IsGradRequired("graph_1"));
        ASSERT_TRUE(a.IsGradRequired("graph_2"));
        ASSERT_TRUE(a.IsGradRequired("graph_3"));
        Array b = a.AsConstant({"graph_1", "graph_2"});

        ExpectEqualView<bool>(a, b);
        EXPECT_FALSE(b.IsGradRequired("graph_1"));
        EXPECT_FALSE(b.IsGradRequired("graph_2"));
        EXPECT_TRUE(b.IsGradRequired("graph_3"));

        EXPECT_TRUE(a.IsGradRequired("graph_1"));
        EXPECT_TRUE(a.IsGradRequired("graph_2"));
        EXPECT_TRUE(a.IsGradRequired("graph_3"));
    }
}

TEST_P(ArrayTest, AddBackward) {
    Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});

    a.RequireGrad();
    b.RequireGrad();

    Array o = a + b;

    auto op_node = internal::GetArrayNode(o)->next_node();
    Array go = testing::MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go, {kDefaultGraphId});
    Array gb = op_node->backward_functions()[1](go, {kDefaultGraphId});

    ExpectEqual<bool>(ga, go);
    ExpectEqual<bool>(gb, go);
}

TEST_P(ArrayTest, MulBackward) {
    Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});

    a.RequireGrad();
    b.RequireGrad();

    Array o = a * b;

    auto op_node = internal::GetArrayNode(o)->next_node();
    Array go = testing::MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go, {kDefaultGraphId});
    Array gb = op_node->backward_functions()[1](go, {kDefaultGraphId});

    ExpectEqual<bool>(ga, go * b);
    ExpectEqual<bool>(gb, go * a);

    EXPECT_FALSE(ga.IsGradRequired());
    EXPECT_FALSE(gb.IsGradRequired());
}

TEST_P(ArrayTest, MulBackwardCapture) {
    Array y = [this]() {
        Array x1 = testing::MakeArray<float>({1}, {2.0f});
        Array x2 = testing::MakeArray<float>({1}, {3.0f});
        x1.RequireGrad();
        x2.RequireGrad();
        return x1 * x2;
    }();
    auto op_node = internal::GetArrayNode(y)->next_node();
    auto lhs_func = op_node->backward_functions()[0];
    auto rhs_func = op_node->backward_functions()[1];
    Array gy = testing::MakeArray<float>({1}, {1.0f});

    Array gx1 = lhs_func(gy, {kDefaultGraphId});
    Array e1 = testing::MakeArray<float>({1}, {3.0f});
    ExpectEqual<float>(e1, gx1);
    EXPECT_FALSE(gx1.IsGradRequired());

    Array gx2 = rhs_func(gy, {kDefaultGraphId});
    Array e2 = testing::MakeArray<float>({1}, {2.0f});
    ExpectEqual<float>(e2, gx2);
    EXPECT_FALSE(gx2.IsGradRequired());
}

TEST_P(ArrayTest, MulBackwardMultipleGraphs) {
    GraphId graph_id1 = "graph_1";
    GraphId graph_id2 = "graph_2";

    Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = testing::MakeArray<bool>({4, 1}, {true, false, true, false});

    a.RequireGrad(graph_id1);
    b.RequireGrad(graph_id2);

    Array o = a * b;
    Array go = testing::MakeArray<bool>({4, 1}, {true, true, true, true});

    auto op_node1 = internal::GetArrayNode(o, graph_id1)->next_node();
    Array ga = op_node1->backward_functions()[0](go, {graph_id1});

    auto op_node2 = internal::GetArrayNode(o, graph_id2)->next_node();
    Array gb = op_node2->backward_functions()[0](go, {graph_id2});

    EXPECT_FALSE(ga.IsGradRequired(graph_id1));
    EXPECT_TRUE(ga.IsGradRequired(graph_id2));

    EXPECT_TRUE(gb.IsGradRequired(graph_id1));
    EXPECT_FALSE(gb.IsGradRequired(graph_id2));
}

TEST_P(ArrayTest, MultipleGraphsRequireGradDefault) {
    Array a = testing::MakeArray<float>({1}, {2.0f});

    EXPECT_FALSE(a.IsGradRequired());

    a.RequireGrad();

    EXPECT_TRUE(a.IsGradRequired());
    EXPECT_THROW(a.RequireGrad(), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsRequireGradNamed) {
    GraphId graph_id = "graph_1";

    Array a = testing::MakeArray<float>({1}, {2.0f});

    ASSERT_FALSE(a.IsGradRequired(graph_id));

    a.RequireGrad(graph_id);

    EXPECT_TRUE(a.IsGradRequired(graph_id));
    EXPECT_THROW(a.RequireGrad(graph_id), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsRequireGradChainedCallsCtor) {
    Array a = testing::MakeArray<float>({1}, {2.0f}).RequireGrad();

    EXPECT_TRUE(a.IsGradRequired());
    EXPECT_THROW(a.RequireGrad(), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsRequireGradChainedCallsRequireGrad) {
    Array a = testing::MakeArray<float>({1}, {2.0f});

    EXPECT_THROW(a.RequireGrad().RequireGrad(), XchainerError);
}

TEST_P(ArrayTest, MultipleGraphsForward) {
    Array a = testing::MakeArray<float>({1}, {2.0f});
    Array b = testing::MakeArray<float>({1}, {2.0f});

    GraphId graph_id_1 = "graph_1";
    GraphId graph_id_2 = "graph_2";

    a.RequireGrad(graph_id_1);
    b.RequireGrad(graph_id_2);

    EXPECT_TRUE(a.IsGradRequired(graph_id_1));
    EXPECT_FALSE(a.IsGradRequired(graph_id_2));

    EXPECT_FALSE(b.IsGradRequired(graph_id_1));
    EXPECT_TRUE(b.IsGradRequired(graph_id_2));

    Array o = a * b;

    EXPECT_TRUE(o.IsGradRequired(graph_id_1));
    EXPECT_TRUE(o.IsGradRequired(graph_id_2));

    // No unspecified graphs are generated
    EXPECT_FALSE(o.IsGradRequired(kDefaultGraphId));
    EXPECT_FALSE(o.IsGradRequired("graph_3"));
}

INSTANTIATE_TEST_CASE_P(ForEachBackend,
                        ArrayTest,
                        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                            std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                            std::string{"native"}));

}  // namespace
}  // namespace xchainer
