#include "xchainer/array.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

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
#include "xchainer/slice.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/context_session.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

template <typename T>
void ExpectDataEqual(const Array& expected, const Array& actual) {
    actual.device().Synchronize();
    IndexableArray<const T> expected_iarray{expected};
    IndexableArray<const T> actual_iarray{actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
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
    IndexableArray<const T> actual_iarray{actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        const auto& actual = actual_iarray[indexer];
        EXPECT_EQ(expected_data[i], actual) << "where i is " << i;
    }
}

template <typename T>
void ExpectDataEqual(T expected, const Array& actual) {
    actual.device().Synchronize();
    IndexableArray<const T> actual_iarray{actual};
    Indexer<> indexer{actual.shape()};
    for (int64_t i = 0; i < indexer.total_size(); ++i) {
        indexer.Set(i);
        const auto& actual = actual_iarray[indexer];
        if (std::isnan(expected)) {
            EXPECT_TRUE(std::isnan(actual)) << "where i is " << i;
        } else {
            EXPECT_EQ(expected, actual) << "where i is " << i;
        }
    }
}

template <typename T>
void ExpectEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    EXPECT_EQ(&expected.device(), &actual.device());
    ExpectDataEqual<T>(expected, actual);
}

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

void ExpectArraysEqualAttributes(const Array& a, const Array& b) {
    EXPECT_EQ(a.dtype(), b.dtype());
    EXPECT_EQ(a.shape(), b.shape());
    EXPECT_EQ(a.IsContiguous(), b.IsContiguous());
    EXPECT_EQ(a.offset(), b.offset());
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

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    void SetUp() override {
        const std::string& backend_name = ::testing::get<0>(GetParam());
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

public:
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
    void CheckContiguousFill(T expected, Scalar scalar) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(scalar);
        ExpectDataEqual(expected, x);
    }

    template <typename T>
    void CheckContiguousFill(T value) {
        CheckContiguousFill(value, value);
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

TEST_P(ArrayTest, ContiguousFill) {
    CheckContiguousFill(true);
    CheckContiguousFill(false);
    CheckContiguousFill(int8_t{0});
    CheckContiguousFill(int8_t{-1});
    CheckContiguousFill(int8_t{5});
    CheckContiguousFill(int8_t{-128});
    CheckContiguousFill(int8_t{127});
    CheckContiguousFill(int16_t{0});
    CheckContiguousFill(int16_t{-3});
    CheckContiguousFill(int32_t{0});
    CheckContiguousFill(int32_t{-3});
    CheckContiguousFill(int64_t{0});
    CheckContiguousFill(int64_t{-3});
    CheckContiguousFill(uint8_t{0});
    CheckContiguousFill(uint8_t{255});
    CheckContiguousFill(float{0});
    CheckContiguousFill(float{std::numeric_limits<float>::infinity()});
    CheckContiguousFill(float{std::nanf("")});
    CheckContiguousFill(double{0});
    CheckContiguousFill(double{std::numeric_limits<double>::infinity()});
    CheckContiguousFill(double{std::nan("")});

    CheckContiguousFill(true, Scalar(int32_t{1}));
    CheckContiguousFill(true, Scalar(int32_t{2}));
    CheckContiguousFill(true, Scalar(int32_t{-1}));
    CheckContiguousFill(false, Scalar(int32_t{0}));
    CheckContiguousFill(int8_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int8_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int8_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int8_t{1}, Scalar(true));
    CheckContiguousFill(int8_t{1}, Scalar(1.0f));
    CheckContiguousFill(int8_t{1}, Scalar(1.0));
    CheckContiguousFill(int16_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int16_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int16_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int16_t{1}, Scalar(true));
    CheckContiguousFill(int16_t{1}, Scalar(1.0f));
    CheckContiguousFill(int16_t{1}, Scalar(1.0));
    CheckContiguousFill(int32_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int32_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int32_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int32_t{1}, Scalar(true));
    CheckContiguousFill(int32_t{1}, Scalar(1.0f));
    CheckContiguousFill(int32_t{1}, Scalar(1.0));
    CheckContiguousFill(int64_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(int64_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(int64_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(int64_t{1}, Scalar(true));
    CheckContiguousFill(int64_t{1}, Scalar(1.0f));
    CheckContiguousFill(int64_t{1}, Scalar(1.0));
    CheckContiguousFill(uint8_t{1}, Scalar(int32_t{1}));
    CheckContiguousFill(uint8_t{1}, Scalar(int64_t{1}));
    CheckContiguousFill(uint8_t{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(uint8_t{1}, Scalar(true));
    CheckContiguousFill(uint8_t{1}, Scalar(1.0f));
    CheckContiguousFill(uint8_t{1}, Scalar(1.0));
    CheckContiguousFill(float{1}, Scalar(int32_t{1}));
    CheckContiguousFill(float{1}, Scalar(int64_t{1}));
    CheckContiguousFill(float{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(float{1}, Scalar(true));
    CheckContiguousFill(float{1}, Scalar(1.0f));
    CheckContiguousFill(float{1}, Scalar(1.0));
    CheckContiguousFill(double{1}, Scalar(int32_t{1}));
    CheckContiguousFill(double{1}, Scalar(int64_t{1}));
    CheckContiguousFill(double{1}, Scalar(uint8_t{1}));
    CheckContiguousFill(double{1}, Scalar(true));
    CheckContiguousFill(double{1}, Scalar(1.0f));
    CheckContiguousFill(double{1}, Scalar(1.0));
}

TEST_P(ArrayTest, NonContiguousFill) {
    Dtype dtype = Dtype::kFloat32;
    float value = 1.0f;
    {
        Array a = Array::Zeros(Shape{3, 3}, dtype);
        Array b = a.Transpose();
        b.Fill(value);
        ExpectDataEqual(value, b);
        ExpectDataEqual(value, a);
    }
    {
        Array a = Array::Zeros(Shape{3, 3}, dtype);
        a.At({1}).Fill(value);
        ExpectDataEqual(value, a.At({1}));
        // check other rows are not affected
        ExpectDataEqual(0.0f, a.At({0}));
        ExpectDataEqual(0.0f, a.At({2}));
    }
    {
        Array a = Array::Zeros(Shape{3, 3}, dtype);
        a.At({Slice{}, {1}}).Fill(value);
        ExpectDataEqual(value, a.At({Slice{}, {1}}));
        // check other columns are not affected
        ExpectDataEqual(0.0f, a.At({Slice{}, {0}}));
        ExpectDataEqual(0.0f, a.At({Slice{}, {2}}));
    }
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

    // non-contiguous
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a_view);
        Array e_view = testing::MakeArray<int32_t>({3, 1}, {2, 5, 8});
        Array e = testing::MakeArray<int32_t>({3, 3}, {0, 2, 2, 3, 5, 5, 6, 8, 8});
        a_view += b;
        ExpectEqual<int32_t>(e_view, a_view);
        ExpectEqual<int32_t>(e, a);
    }

    // broadcast
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(1);
        a += b;
        ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(1);
        a += b;
        ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({4}, Dtype::kInt32);
        EXPECT_THROW(a += b, XchainerError);
    }
    {
        Array a = testing::MakeArray({3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 3}, Dtype::kInt32);
        EXPECT_THROW(a += b, XchainerError);
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

    // non-contiguous
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array a_view = a.At({Slice{}, Slice{1, 2}});
        Array b = Array::FullLike(a_view, 2);
        Array e = testing::MakeArray<int32_t>({3, 3}, {0, 2, 2, 3, 8, 5, 6, 14, 8});
        Array e_view = testing::MakeArray<int32_t>({3, 1}, {2, 8, 14});
        a_view *= b;
        ExpectEqual<int32_t>(e_view, a_view);
        ExpectEqual<int32_t>(e, a);
    }

    // broadcast
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 1}, 2, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(0, 2);
        a *= b;
        ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3}, 2, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(0, 2);
        a *= b;
        ExpectEqual<int32_t>(e, a);
    }
    {
        Array a = testing::MakeArray({3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 3}, 2, Dtype::kInt32);
        Array e = testing::MakeArray<int32_t>({3, 3}, {0, 2, 4, 0, 2, 4, 0, 2, 4});
        EXPECT_THROW(a *= b, XchainerError);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({4}, 2, Dtype::kInt32);
        EXPECT_THROW(a *= b, XchainerError);
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

    // non-contiguous
    {
        Array a = Array(testing::MakeArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
        Array b = Array::OnesLike(a);
        Array e = testing::MakeArray<int32_t>({3, 1}, {2, 5, 8});
        Array o = a + b;
        ExpectEqual<int32_t>(e, o);
    }

    // broadcast
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 1}, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(1);
        Array o = a + b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3}, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(1);
        Array o = a + b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3}).WithLinearData<int32_t>();
        Array b = Array::Ones({3, 3}, Dtype::kInt32);
        Array e = testing::MakeArray<int32_t>({3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
        Array o = a + b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3, 1}).WithLinearData<int32_t>();
        Array b = testing::MakeArray({1, 2}).WithLinearData<int32_t>(1);
        Array e = testing::MakeArray<int32_t>({3, 2}, {1, 2, 2, 3, 3, 4});
        Array o = a + b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Ones({4}, Dtype::kInt32);
        EXPECT_THROW(a + b, XchainerError);
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

    // non-contiguous
    {
        Array a = Array(testing::MakeArray({3, 3}).WithLinearData<int32_t>()).At({Slice{}, Slice{1, 2}});
        Array b = Array::FullLike(a, 2);
        Array e = testing::MakeArray<int32_t>({3, 1}, {2, 8, 14});
        Array o = a * b;
        ExpectEqual<int32_t>(e, o);
    }

    // broadcast
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 1}, 2, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(0, 2);
        Array o = a * b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({3}, 2, Dtype::kInt32);
        Array e = testing::MakeArray({3, 3}).WithLinearData<int32_t>(0, 2);
        Array o = a * b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3}).WithLinearData<int32_t>();
        Array b = Array::Full({3, 3}, 2, Dtype::kInt32);
        Array e = testing::MakeArray<int32_t>({3, 3}, {0, 2, 4, 0, 2, 4, 0, 2, 4});
        Array o = a * b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3, 1}).WithLinearData<int32_t>(1);
        Array b = testing::MakeArray({1, 2}).WithLinearData<int32_t>(1);
        Array e = testing::MakeArray<int32_t>({3, 2}, {1, 2, 2, 4, 3, 6});
        Array o = a * b;
        ExpectEqual<int32_t>(e, o);
    }
    {
        Array a = testing::MakeArray({3, 3}).WithLinearData<int32_t>();
        Array b = Array::Full({4}, 2, Dtype::kInt32);
        EXPECT_THROW(a * b, XchainerError);
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
    Array a = testing::MakeArray({2, 3})          //
                      .WithLinearData<int32_t>()  //
                      .WithPadding(0);
    Array b = a.Transpose();

    EXPECT_EQ(Shape({3, 2}), b.shape());
    EXPECT_EQ(Strides({4, 12}), b.strides());

    Array e = testing::MakeArray({3, 2}).WithData<int32_t>({0, 3, 1, 4, 2, 5});
    ExpectEqual<int32_t>(e, b);
}

TEST_P(ArrayTest, TransposeNoncontiguous) {
    Array a = testing::MakeArray({2, 3})          //
                      .WithLinearData<int32_t>()  //
                      .WithPadding(1);
    Array b = a.Transpose();

    EXPECT_EQ(Shape({3, 2}), b.shape());

    Array e = testing::MakeArray({3, 2}).WithData<int32_t>({0, 3, 1, 4, 2, 5});
    ExpectEqual<int32_t>(e, b);
}

TEST_P(ArrayTest, TransposeBackward) {
    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {xs[0].Transpose()}; },
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
            {(*testing::MakeArray({2, 3}, {1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
            {Array::Ones({3, 2}, Dtype::kFloat32).RequireGrad()},
            {Array::Ones({2, 3}, Dtype::kFloat32)},
            {Array::Full({2, 3}, 0.01f), Array::Full({3, 2}, 0.01f)});
}

TEST_P(ArrayTest, AtBackward) {
    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<ArrayIndex> indices{1, NewAxis{}, Slice{1, 3}};
                return {xs[0].At(indices)};
            },
            {(*testing::MakeArray({2, 3}, {1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
            {Array::Ones({1, 2}, Dtype::kFloat32)},
            {Array::Full({2, 3}, 1e-3f)});
}

TEST_P(ArrayTest, AtDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                std::vector<ArrayIndex> indices{0, NewAxis{}, Slice{1, 3}};
                auto y = xs[0].At(indices);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::MakeArray({2, 3}, {1.f, -1.f, 2.f, -2.f, 3.f, -3.f})).RequireGrad()},
            {Array::Ones({1, 2}, Dtype::kFloat32).RequireGrad()},
            {Array::Ones({2, 3}, Dtype::kFloat32)},
            {Array::Full({2, 3}, 1e-3f), Array::Full({1, 2}, 1e-3f)});
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
    {
        Array a = testing::MakeArray<float>({3, 1}, {1.0f, 2.0f, 3.0f})  //
                          .WithPadding(1);
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

    // Non-contiguous
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false})  //
                          .WithPadding(4);
        Array b = a.AsConstant(CopyKind::kCopy);
        EXPECT_EQ(&b.device(), &a.device());
        ExpectEqualCopy<bool>(a, b);
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
    // Non-contiguous
    {
        Array a = testing::MakeArray<bool>({4, 1}, {true, true, false, false})  //
                          .WithPadding(4);
        Array b = a.AsConstant(CopyKind::kView);
        EXPECT_EQ(&b.device(), &a.device());
        ExpectEqualView<bool>(a, b);
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
    Array a = (*testing::MakeArray<float>({1}, {2.0f})).RequireGrad();

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

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ArrayTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

TEST(ArrayAtTest, At) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 1};
    Shape output_shape{1, 2, 1};
    std::vector<ArrayIndex> indices{-1, NewAxis{}, Slice{1, 3}};
    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    Array b = a.At(indices);

    EXPECT_EQ(output_shape, b.shape());
    Array e = testing::MakeArray(output_shape).WithData<T>({4, 5});
    ExpectEqual<T>(e, b);

    // Check if strides are 0 for newaxis.
    EXPECT_EQ(0, b.strides()[0]);
    EXPECT_NE(0, b.strides()[1]);
    EXPECT_NE(0, b.strides()[2]);
}

// Index out of bounds
TEST(ArrayAtTest, InvalidAt1) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3};
    std::vector<ArrayIndex> indices{0, 0, 0};
    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(a.At(indices), DimensionError);
}

// Too large dimension
TEST(ArrayAtTest, InvalidAt2) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3};
    std::vector<ArrayIndex> indices{2};
    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(a.At(indices), DimensionError);
}

TEST(ArrayReshapeTest, Reshape) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, 4, 2};

    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    Array b = a.Reshape(output_shape);
    ASSERT_EQ(output_shape, b.shape());
    EXPECT_EQ(a.data().get(), b.data().get()) << "Reshape must be done without copying data";
    Array e = testing::MakeArray(output_shape).WithLinearData<T>();
    ExpectEqual<T>(e, b);
}

// If an input array has a unit-length axis with 0-stride, that axis should not give rise to any copies.
TEST(ArrayReshapeTest, ReshapeNoCopyZeroStrideAxis) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape_before_newaxis{2, 3, 4};
    Shape output_shape{3, 4, 2};

    // The shape of the input array is (2, 1, 3, 4) with strides (48, 0, 16, 4).
    Array a = (*testing::MakeArray(input_shape_before_newaxis).WithLinearData<T>()).At({Slice{}, NewAxis{}, Slice{}, Slice{}});
    assert(std::find(a.strides().begin(), a.strides().end(), 0) != a.strides().end());

    Array b = a.Reshape(output_shape);
    ASSERT_EQ(output_shape, b.shape());
    EXPECT_EQ(a.data().get(), b.data().get()) << "Reshape must be done without copying data";
    Array e = testing::MakeArray(output_shape).WithLinearData<T>();
    ExpectEqual<T>(e, b);
}

TEST(ArrayReshapeTest, InvalidReshape) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 4};
    Shape output_shape{2, 4, 4};

    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(a.Reshape(output_shape), DimensionError);
}

TEST(ArraySqueezeTest, SqueezeAllUnitLengthAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array b = a.Squeeze();
    Array e = testing::MakeArray({2, 3, 4}).WithLinearData<T>();
    ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeSpecifiedUnitLenghtAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array b = a.Squeeze(std::vector<int8_t>{2, 0, 4});
    Array e = testing::MakeArray({2, 3, 1, 4}).WithLinearData<T>();
    ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeAllAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 1, 1}).WithLinearData<T>();
    Array b = a.Squeeze();
    Array e = testing::MakeArray<T>({}, std::vector<T>(1, 0));
    ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeMultipleCalls) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    Array b = a.Squeeze(std::vector<int8_t>{0, 2});
    Array c = b.Squeeze(std::vector<int8_t>{3});
    Array e = testing::MakeArray({2, 3, 1, 4}).WithLinearData<T>();
    ExpectEqual<T>(e, c);
}

TEST(ArraySqueezeTest, SqueezeNonContiguous) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>().WithPadding(1);
    Array b = a.Squeeze(std::vector<int8_t>{0, 2, 4});
    Array e = testing::MakeArray({2, 3, 1, 4}).WithLinearData<T>();
    ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeNegativeAxis) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({2, 3, 4, 1}).WithLinearData<T>();
    Array b = a.Squeeze(std::vector<int8_t>{-1});
    Array e = testing::MakeArray({2, 3, 4}).WithLinearData<T>();
    ExpectEqual<T>(e, b);
}

TEST(ArraySqueezeTest, SqueezeNoSqueezableAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({2, 3, 4}).WithLinearData<T>();
    Array e = a.Squeeze();
    ExpectEqual<T>(e, a);
    EXPECT_EQ(e.body(), a.body());
}

TEST(ArraySqueezeTest, InvalidSqueezeNonUnitLengthAxis) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    EXPECT_THROW(Array b = a.Squeeze(std::vector<int8_t>{1}), DimensionError);
}

TEST(ArraySqueezeTest, InvalidSqueezeDuplicateAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<T>();
    EXPECT_THROW(a.Squeeze(std::vector<int8_t>{0, 2, 2}), XchainerError);
}

TEST(ArraySqueezeTest, InvalidSqueezeOutOfRangeAxes) {
    using T = int32_t;
    testing::ContextSession context_session{};

    Array a = testing::MakeArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(a.Squeeze(std::vector<int8_t>{3}), DimensionError);
}

TEST_P(ArrayTest, SqueezeBackward) {
    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {xs[0].Squeeze(std::vector<int8_t>{0, 2, 4})};
            },
            {(*testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {testing::MakeArray({2, 3, 1, 4}).WithLinearData<float>(0.f, 0.1f)},
            {Array::Full({1, 2, 1, 3, 1, 1, 4}, 1e-2f)});
}

TEST_P(ArrayTest, SqueezeDoubleBackward) {
    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = xs[0].Squeeze(std::vector<int8_t>{0, 2, 4});
                return {y * y};  // to make it nonlinear
            },
            {(*testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<float>().WithPadding(1)).RequireGrad()},
            {(*testing::MakeArray({2, 3, 1, 4}).WithLinearData<float>(0.f, 0.1f)).RequireGrad()},
            {testing::MakeArray({1, 2, 1, 3, 1, 1, 4}).WithLinearData<float>()},
            {Array::Full({1, 2, 1, 3, 1, 1, 4}, 1e-2f), Array::Full({2, 3, 1, 4}, 1e-2f)},
            1e-4f,
            1e-3f);
}

TEST(ArrayBroadcastToTest, BroadcastTo) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 1};
    Shape output_shape{3, 1, 2, 3, 1, 2};

    Array aa = testing::MakeArray(input_shape).WithData<T>({1, 2, 3, 4, 5, 6});
    Array a = aa.At({Slice(), Slice(), Slice(), NewAxis{}});  // Make a broadcastable axis.
    ASSERT_EQ(Shape({2, 3, 1, 1}), a.shape());                // Check test precondition

    Array b = a.BroadcastTo(output_shape);
    ASSERT_EQ(output_shape, b.shape());
    EXPECT_EQ(a.data().get(), b.data().get()) << "BroadcastTo must be done without copying data";
    ASSERT_EQ(0, b.strides()[1]) << "Stride of broadcasted dimension must be 0";

    std::vector<int64_t> output_data;
    for (int i = 0; i < 3; ++i) {
        output_data.insert(output_data.end(), {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6});
    }
    Array e = testing::MakeArray(output_shape).WithData<T>(output_data.begin(), output_data.end());
    ExpectEqual<T>(e, b);
}

// Can't broadcast to smaller dimensions
TEST(ArrayBroadcastToTest, InvalidBroadcastTo_NotEnoughDimension) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 4};
    Shape output_shape{3, 4};

    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(a.BroadcastTo(output_shape), DimensionError);
}

// Can't broadcast with incompatible axis
TEST(ArrayBroadcastToTest, InvalidBroadcastTo_IncompatibleDimension) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3, 3};
    Shape output_shape{2, 4, 3};

    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(a.BroadcastTo(output_shape), DimensionError);
}

// Can't broadcast at the end
TEST(ArrayBroadcastToTest, InvalidBroadcastTo_NotBroadcastableAtEnd) {
    using T = int32_t;
    testing::ContextSession context_session{};
    Shape input_shape{2, 3};
    Shape output_shape{2, 3, 4};

    Array a = testing::MakeArray(input_shape).WithLinearData<T>();
    EXPECT_THROW(a.BroadcastTo(output_shape), DimensionError);
}

TEST(ArrayBroadcastToTest, BroadcastToBackward) {
    using T = double;
    testing::ContextSession context_session{};

    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {xs[0].BroadcastTo({2, 3, 4, 3})};
            },
            {(*testing::MakeArray({1, 3, 1, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>(-0.1, 0.1)},
            {Array::Full({1, 3, 1, 3}, 1e-1)});
}

TEST(ArrayBroadcastToTest, BroadcastToDoubleBackward) {
    using T = double;
    testing::ContextSession context_session{};

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = xs[0].BroadcastTo({2, 3, 4, 3});
                return {y * y};  // to make it nonlinear
            },
            {(*testing::MakeArray({1, 3, 1, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::MakeArray({1, 3, 1, 3}).WithLinearData<T>()},
            {Array::Full({1, 3, 1, 3}, 1e-1), Array::Full({2, 3, 4, 3}, 1e-1)});
}

class ArraySumTest : public ::testing::TestWithParam<std::string> {
protected:
    void SetUp() override {
        const std::string& backend_name = GetParam();
        device_session_.emplace(DeviceId{backend_name, 0});
    }

    void TearDown() override { device_session_.reset(); }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(ArraySumTest, Sum) {
    using T = float;

    Array a = testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum(std::vector<int8_t>{2, 1, -1});
    EXPECT_EQ(Shape{2}, b.shape());
    Array e = testing::MakeArray(Shape{2}).WithData<T>({630.0f, 1926.0f});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, SumAllAxes) {
    using T = float;

    Array a = testing::MakeArray({2, 3, 3}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum();
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::MakeArray(Shape{}).WithData<T>({153.0f});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, SumZero) {
    using T = float;

    Array a = testing::MakeArray({0}).WithData<T>({});
    Array b = a.Sum();
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::MakeArray(Shape{}).WithData<T>({0.0f});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, SumOne) {
    using T = float;

    Array a = testing::MakeArray({}).WithData<T>({42.0f}).WithPadding(1);
    Array b = a.Sum();
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::MakeArray(Shape{}).WithData<T>({42.0f});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, SumTwo) {
    using T = float;

    Array a = testing::MakeArray({2}).WithData<T>({42.0f, 37.0f}).WithPadding(1);
    Array b = a.Sum();
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::MakeArray(Shape{}).WithData<T>({79.0f});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, SumLarge) {
    using T = int64_t;

    Array a = testing::MakeArray({0x100000}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum(std::vector<int8_t>{0});
    EXPECT_EQ(Shape{}, b.shape());
    Array e = testing::MakeArray(Shape{}).WithData<T>({0x7ffff80000});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, SumKeepDims) {
    using T = float;

    Array a = testing::MakeArray({2, 3, 2, 4}).WithLinearData<T>().WithPadding(1);
    Array b = a.Sum(std::vector<int8_t>{-1, 1}, true);
    EXPECT_EQ(Shape({2, 1, 2, 1}), b.shape());
    EXPECT_EQ(0, b.strides()[1]);
    EXPECT_EQ(0, b.strides()[3]);
    Array e = testing::MakeArray(Shape{2, 1, 2, 1}).WithData<T>({114.0f, 162.0f, 402.0f, 450.0f});
    ExpectEqual<T>(e, b);
}

TEST_P(ArraySumTest, InvalidSumDuplicateAxes) {
    using T = float;

    Array a = testing::MakeArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(a.Sum(std::vector<int8_t>{1, 1}), XchainerError);
}

TEST_P(ArraySumTest, InvalidSumOutOfRangeAxes) {
    using T = float;

    Array a = testing::MakeArray({2, 3, 4}).WithLinearData<T>();
    EXPECT_THROW(a.Sum(std::vector<int8_t>{3}), DimensionError);
}

TEST_P(ArraySumTest, SumBackward) {
    using T = double;
    testing::ContextSession context_session{};

    CheckBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                return {xs[0].Sum(std::vector<int8_t>{1, 3})};
            },
            {(*testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {testing::MakeArray({2, 4}).WithLinearData<T>(-0.1, 0.1)},
            {Array::Full({2, 3, 4, 3}, 1e-1)});
}

TEST_P(ArraySumTest, SumDoubleBackward_Keepdims) {
    using T = double;
    testing::ContextSession context_session{};

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = xs[0].Sum(std::vector<int8_t>{1, 3}, true);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::MakeArray({2, 1, 4, 1}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Array::Full({2, 3, 4, 3}, 1e-1), Array::Full({2, 1, 4, 1}, 1e-1)});
}

TEST_P(ArraySumTest, SumDoubleBackward_NoKeepdims) {
    using T = double;
    testing::ContextSession context_session{};

    CheckDoubleBackwardComputation(
            [](const std::vector<Array>& xs) -> std::vector<Array> {
                auto y = xs[0].Sum(std::vector<int8_t>{1, 3}, false);
                return {y * y};  // to make it nonlinear
            },
            {(*testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>().WithPadding(1)).RequireGrad()},
            {(*testing::MakeArray({2, 4}).WithLinearData<T>(-0.1, 0.1)).RequireGrad()},
            {testing::MakeArray({2, 3, 4, 3}).WithLinearData<T>()},
            {Array::Full({2, 3, 4, 3}, 1e-1), Array::Full({2, 4}, 1e-1)});
}

INSTANTIATE_TEST_CASE_P(
        ForEachBackend,
        ArraySumTest,
        ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                std::string{"native"}));

}  // namespace
}  // namespace xchainer
