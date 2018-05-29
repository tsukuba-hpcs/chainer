#include "xchainer/shape.h"

#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include <gsl/gsl>

#include "xchainer/axes.h"
#include "xchainer/dtype.h"
#include "xchainer/strides.h"

namespace xchainer {
namespace {

void CheckSpanEqual(std::initializer_list<int64_t> expect, gsl::span<const int64_t> actual) {
    EXPECT_EQ(gsl::make_span(expect.begin(), expect.end()), actual);
}

TEST(ShapeTest, Ctor) {
    {  // Default ctor
        const Shape shape{};
        EXPECT_EQ(0, shape.ndim());
        EXPECT_EQ(size_t{0}, shape.size());
    }
    {  // From std::initializer_list
        const Shape shape{2, 3, 4};
        EXPECT_EQ(3, shape.ndim());
        EXPECT_EQ(size_t{3}, shape.size());
        CheckSpanEqual({2, 3, 4}, shape.span());
        EXPECT_EQ(2 * 3 * 4, shape.GetTotalSize());
    }
    {  // From gsl::span
        const std::array<int64_t, 3> dims{2, 3, 4};
        const Shape shape{gsl::make_span(dims)};
        EXPECT_EQ(3, shape.ndim());
        CheckSpanEqual({2, 3, 4}, shape.span());
    }
    {  // From iterators
        const std::vector<int64_t> dims{2, 3, 4};
        const Shape shape{dims.begin(), dims.end()};
        EXPECT_EQ(3, shape.ndim());
        CheckSpanEqual({2, 3, 4}, shape.span());
    }
    {  // From empty std::initializer_list
        const Shape shape(std::initializer_list<int64_t>{});
        EXPECT_EQ(0, shape.ndim());
        CheckSpanEqual({}, shape.span());
    }
    {  // From empty gsl::span
        const std::array<int64_t, 0> dims{};
        const Shape shape{gsl::make_span(dims)};
        EXPECT_EQ(0, shape.ndim());
        CheckSpanEqual({}, shape.span());
    }
    {  // From empty iterators
        const std::vector<int64_t> dims{};
        const Shape shape{dims.begin(), dims.end()};
        EXPECT_EQ(0, shape.ndim());
        CheckSpanEqual({}, shape.span());
    }
    {  // Too long std::initializer_list
        EXPECT_THROW(Shape({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}), DimensionError);
    }
    {  // Too long gsl::span
        const std::array<int64_t, kMaxNdim + 1> too_long{1};
        EXPECT_THROW(Shape{gsl::make_span(too_long)}, DimensionError);
    }
    {  // Too long iterators
        std::vector<int64_t> dims{};
        dims.resize(kMaxNdim + 1);
        std::iota(dims.begin(), dims.end(), int64_t{1});
        EXPECT_THROW(Shape({dims.begin(), dims.end()}), DimensionError);
    }
}

TEST(ShapeTest, Subscript) {
    const Shape shape = {2, 3, 4};
    EXPECT_EQ(2, shape[0]);
    EXPECT_EQ(3, shape[1]);
    EXPECT_EQ(4, shape[2]);
    EXPECT_THROW(shape[-1], DimensionError);
    EXPECT_THROW(shape[3], DimensionError);
}

TEST(ShapeTest, Compare) {
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {2, 3, 4};
        EXPECT_TRUE(shape == shape2);
    }
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {2, 3};
        EXPECT_TRUE(shape != shape2);
    }
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {1, 2, 3};
        EXPECT_TRUE(shape != shape2);
    }
}

TEST(ShapeTest, CheckEqual) {
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {2, 3, 4};
        EXPECT_NO_THROW(CheckEqual(shape, shape2));
    }
    {
        const Shape shape = {2, 3, 4};
        const Shape shape2 = {};
        EXPECT_THROW(CheckEqual(shape, shape2), DimensionError);
    }
}

TEST(ShapeTest, Iterator) {
    const Shape shape = {2, 3, 4};
    CheckSpanEqual({2, 3, 4}, gsl::make_span(shape.begin(), shape.end()));
    CheckSpanEqual({4, 3, 2}, gsl::make_span(std::vector<int64_t>{shape.rbegin(), shape.rend()}));
}

TEST(ShapeTest, ToString) {
    {
        const Shape shape = {};
        EXPECT_EQ(shape.ToString(), "()");
    }
    {
        const Shape shape = {1};
        EXPECT_EQ(shape.ToString(), "(1,)");
    }
    {
        const Shape shape = {2, 3, 4};
        EXPECT_EQ(shape.ToString(), "(2, 3, 4)");
    }
}

TEST(StridesTest, SpanFromShape) {
    const Shape shape = {2, 3, 4};
    CheckSpanEqual({2, 3, 4}, gsl::make_span(shape));
}

TEST(ShapeTest, IsContiguous) {
    {
        Shape shape{2, 3, 4};
        Strides strides{shape, GetItemSize(Dtype::kFloat32)};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{2, 3, 4};
        Strides strides{shape, GetItemSize(Dtype::kFloat32)};
        EXPECT_FALSE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat64)));
    }
    {
        Shape shape{2, 3};
        Strides strides{12, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{2, 3, 4};
        Strides strides{48, 16, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{2, 3, 4};
        Strides strides{32, 16, 4};
        EXPECT_FALSE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{};
        Strides strides{};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{0};
        Strides strides{8};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{0, 0};
        Strides strides{24, 8};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 2};
        Strides strides{12, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 3};
        Strides strides{-12, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 2, 4};
        Strides strides{48, 16, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 3};
        Strides strides{0, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 1, 2, 3};
        Strides strides{0, 0, 12, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 1, 2, 3};
        Strides strides{240, 80, 20, 4};
        EXPECT_FALSE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 1, 2};
        Strides strides{48, 16, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{1, 2};
        Strides strides{0, 4};
        EXPECT_TRUE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{2, 2};
        Strides strides{16, 4};
        EXPECT_FALSE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
    {
        Shape shape{2, 1, 2};
        Strides strides{48, 16, 4};
        EXPECT_FALSE(internal::IsContiguous(shape, strides, GetItemSize(Dtype::kFloat32)));
    }
}

TEST(ShapeTest, IsValidReductionShape) {
    {
        Shape in_shape{};
        Shape out_shape{};
        Axes axes{};
        bool allow_keepdims = true;
        EXPECT_TRUE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{};
        Shape out_shape{};
        Axes axes{};
        bool allow_keepdims = false;
        EXPECT_TRUE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{2, 3, 4};
        Shape out_shape{2};
        Axes axes{1, 2};
        bool allow_keepdims = false;
        EXPECT_TRUE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{2, 3, 4};
        Shape out_shape{2};
        Axes axes{1, 2};
        bool allow_keepdims = false;
        EXPECT_TRUE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{2, 3, 4};
        Shape out_shape{2, 1, 1};
        Axes axes{1, 2};
        bool allow_keepdims = true;
        EXPECT_TRUE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
}

TEST(ShapeTest, IsValidReductionShapeInvalid) {
    {
        Shape in_shape{1};
        Shape out_shape{};
        Axes axes{};
        bool allow_keepdims = true;
        EXPECT_FALSE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{1};
        Shape out_shape{};
        Axes axes{};
        bool allow_keepdims = false;
        EXPECT_FALSE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{1};
        Shape out_shape{1, 2};
        Axes axes{1};
        bool allow_keepdims = true;
        EXPECT_FALSE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{1};
        Shape out_shape{1, 2};
        Axes axes{1};
        bool allow_keepdims = false;
        EXPECT_FALSE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
    {
        Shape in_shape{2, 3, 4};
        Shape out_shape{2, 1, 1};
        Axes axes{1, 2};
        bool allow_keepdims = false;
        EXPECT_FALSE(internal::IsValidReductionShape(in_shape, axes, out_shape, allow_keepdims));
    }
}

class BroadcastShapesCorrectTest : public ::testing::TestWithParam<std::tuple<Shape, Shape, Shape>> {
public:
    const Shape& shape0() const { return std::get<0>(GetParam()); }
    const Shape& shape1() const { return std::get<1>(GetParam()); }
    const Shape& expected() const { return std::get<2>(GetParam()); }
};

TEST_P(BroadcastShapesCorrectTest, Check) {
    Shape result = internal::BroadcastShapes(shape0(), shape1());
    EXPECT_EQ(expected(), result);
}

INSTANTIATE_TEST_CASE_P(
        Cases,
        BroadcastShapesCorrectTest,
        ::testing::Values(
                std::make_tuple(Shape{2, 3, 4}, Shape{2, 3, 4}, Shape{2, 3, 4}),
                std::make_tuple(Shape{1, 3, 1}, Shape{2, 3, 4}, Shape{2, 3, 4}),
                std::make_tuple(Shape{2, 3, 4}, Shape{1, 3, 1}, Shape{2, 3, 4}),
                std::make_tuple(Shape{1, 3, 4}, Shape{2, 3, 1}, Shape{2, 3, 4}),
                std::make_tuple(Shape{3, 4}, Shape{2, 3, 1}, Shape{2, 3, 4})));

TEST(BroadcastShapesInvalidTest, Check) {
    EXPECT_THROW(internal::BroadcastShapes({2, 3}, {3, 2}), DimensionError);
    EXPECT_THROW(internal::BroadcastShapes({2, 3}, {2, 3, 1}), DimensionError);
    EXPECT_THROW(internal::BroadcastShapes({3, 2, 1, 3}, {3, 2, 3}), DimensionError);
}

class ReduceShapesCorrectTest : public ::testing::TestWithParam<std::tuple<Shape, Axes, bool, Shape>> {
public:
    const Shape& shape() const { return std::get<0>(GetParam()); }
    const Axes& axes() const { return std::get<1>(GetParam()); }
    bool keepdims() const { return std::get<2>(GetParam()); }
    const Shape& expected() const { return std::get<3>(GetParam()); }
};

TEST_P(ReduceShapesCorrectTest, Check) {
    Shape result = internal::ReduceShape(shape(), axes(), keepdims());
    EXPECT_EQ(expected(), result);
}

INSTANTIATE_TEST_CASE_P(
        Cases,
        ReduceShapesCorrectTest,
        ::testing::Values(
                std::make_tuple(Shape{}, Axes{}, true, Shape{}),
                std::make_tuple(Shape{}, Axes{}, false, Shape{}),
                std::make_tuple(Shape{2, 3}, Axes{}, true, Shape{2, 3}),
                std::make_tuple(Shape{2, 3}, Axes{}, false, Shape{2, 3}),
                std::make_tuple(Shape{2, 3}, Axes{1}, true, Shape{2, 1}),
                std::make_tuple(Shape{2, 3}, Axes{1}, false, Shape{2}),
                std::make_tuple(Shape{2, 3, 4}, Axes{0, 2}, true, Shape{1, 3, 1}),
                std::make_tuple(Shape{2, 3, 4}, Axes{0, 2}, false, Shape{3})));

TEST(TransposeShapeTest, Normal) {
    Shape actual = internal::TransposeShape(Shape{2, 3, 4, 5}, {2, 0, 3, 1});
    Shape expected{4, 2, 5, 3};
    EXPECT_EQ(expected, actual);
}

TEST(TransposeShapeTest, LongAxis) {
    Shape actual = internal::TransposeShape(Shape{128, 256, 65536}, {1, 2, 0});
    Shape expected{256, 65536, 128};
    EXPECT_EQ(expected, actual);
}

}  // namespace
}  // namespace xchainer
