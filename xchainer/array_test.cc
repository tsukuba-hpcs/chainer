#include "xchainer/array.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <type_traits>

#ifdef XCHAINER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif  // XCHAINER_ENABLE_CUDA
#include <gtest/gtest.h>

#include "xchainer/array_node.h"
#ifdef XCHAINER_ENABLE_CUDA
#include "xchainer/cuda/cuda_runtime.h"
#endif  // XCHAINER_ENABLE_CUDA
#include "xchainer/device.h"
#include "xchainer/op_node.h"

namespace xchainer {
namespace {

class ArrayTest : public ::testing::TestWithParam<::testing::tuple<std::string>> {
protected:
    virtual void SetUp() {
        std::string device_name = ::testing::get<0>(GetParam());
        device_scope_ = std::make_unique<DeviceScope>(device_name);
    }

    virtual void TearDown() { device_scope_.reset(); }

public:
    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::shared_ptr<void> data) {
        return {shape, TypeToDtype<T>, data, true};
    }

    template <typename T>
    Array MakeArray(std::initializer_list<int64_t> shape, std::initializer_list<T> data) {
        auto a = std::make_unique<T[]>(data.size());
        std::copy(data.begin(), data.end(), a.get());
        return {shape, TypeToDtype<T>, std::move(a), true};
    }

    template <typename T>
    void AssertEqual(const Array& lhs, const Array& rhs) {
#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA
        ASSERT_NO_THROW(CheckEqual(lhs.dtype(), rhs.dtype()));
        ASSERT_NO_THROW(CheckEqual(lhs.shape(), rhs.shape()));
        auto total_size = lhs.shape().total_size();
        const T* ldata = static_cast<const T*>(lhs.data().get());
        const T* rdata = static_cast<const T*>(rhs.data().get());
        for (decltype(total_size) i = 0; i < total_size; i++) {
            ASSERT_EQ(ldata[i], rdata[i]) << "ldata[" << i << "] and rdata[" << i << "] do not match";
        }
    }

    bool IsPointerCudaManaged(const void* ptr) {
#ifdef XCHAINER_ENABLE_CUDA
        cudaPointerAttributes attr = {};
        cuda::CheckError(cudaPointerGetAttributes(&attr, ptr));
        return attr.isManaged != 0;
#else
        (void)ptr;
        return false;
#endif  // XCHAINER_ENABLE_CUDA
    }

    template <bool is_const>
    void CheckArray() {
        using TargetArray = std::conditional_t<is_const, const Array, Array>;

        std::shared_ptr<void> data = std::make_unique<float[]>(2 * 3 * 4);
        TargetArray x = MakeArray<float>({2, 3, 4}, data);

        // Basic attributes
        ASSERT_EQ(TypeToDtype<float>, x.dtype());
        ASSERT_EQ(3, x.ndim());
        ASSERT_EQ(2 * 3 * 4, x.total_size());
        ASSERT_EQ(4, x.element_bytes());
        ASSERT_EQ(2 * 3 * 4 * 4, x.total_bytes());
        ASSERT_EQ(0, x.offset());
        ASSERT_TRUE(x.is_contiguous());

        // Array::data
        std::shared_ptr<const void> x_data = x.data();
        if (GetCurrentDevice() == MakeDevice("cpu")) {
            ASSERT_EQ(data, x_data);
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_NE(data, x_data);
            ASSERT_TRUE(IsPointerCudaManaged(x_data.get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckEmpty() {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_EQ(x.shape(), Shape({3, 2}));
        ASSERT_EQ(x.dtype(), dtype);

        if (GetCurrentDevice() == MakeDevice("cpu")) {
            //
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_TRUE(IsPointerCudaManaged(x.data().get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckEmptyLike() {
        Dtype dtype = TypeToDtype<T>;
        Array x_orig = Array::Empty(Shape{3, 2}, dtype);
        Array x = Array::EmptyLike(x_orig);
        ASSERT_NE(x.data(), nullptr);
        ASSERT_NE(x.data(), x_orig.data());
        ASSERT_EQ(x.shape(), x_orig.shape());
        ASSERT_EQ(x.dtype(), x_orig.dtype());

        if (GetCurrentDevice() == MakeDevice("cpu")) {
            //
        } else if (GetCurrentDevice() == MakeDevice("cuda")) {
            ASSERT_TRUE(IsPointerCudaManaged(x.data().get()));
        } else {
            FAIL() << "invalid device";
        }
    }

    template <typename T>
    void CheckFill(T value) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(Scalar{value});

#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA

        int64_t size = x.total_size();
        T* data = static_cast<T*>(x.data().get());

        for (int64_t i = 0; i < size; ++i) {
            if (std::isnan(value)) {
                ASSERT_TRUE(std::isnan(data[i]));
            } else {
                ASSERT_EQ(data[i], value);
            }
        }
    }

    template <typename T, typename U>
    void CheckCastFill(U value) {
        Dtype dtype = TypeToDtype<T>;
        Array x = Array::Empty(Shape{3, 2}, dtype);
        x.Fill(Scalar{value});

#ifdef XCHAINER_ENABLE_CUDA
        std::string device_name = ::testing::get<0>(GetParam());
        if (device_name == "cuda") {
            cuda::CheckError(cudaDeviceSynchronize());
        }
#endif  // XCHAINER_ENABLE_CUDA

        ASSERT_EQ(dtype, x.dtype());

        int64_t size = x.total_size();
        T* data = static_cast<T*>(x.data().get());

        for (int64_t i = 0; i < size; ++i) {
            ASSERT_EQ(data[i], static_cast<T>(value));
        }
    }

private:
    std::unique_ptr<DeviceScope> device_scope_;
};

TEST_P(ArrayTest, ArrayCtor) { CheckArray<false>(); }

TEST_P(ArrayTest, ConstArrayCtor) { CheckArray<true>(); }

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
    CheckFill(static_cast<int8_t>(0));
    CheckFill(static_cast<int8_t>(-1));
    CheckFill(static_cast<int8_t>(5));
    CheckFill(static_cast<int8_t>(-128));
    CheckFill(static_cast<int8_t>(127));
    CheckFill(static_cast<int16_t>(0));
    CheckFill(static_cast<int16_t>(-3));
    CheckFill(static_cast<int32_t>(0));
    CheckFill(static_cast<int32_t>(-3));
    CheckFill(static_cast<int64_t>(0));
    CheckFill(static_cast<int64_t>(-3));
    CheckFill(static_cast<uint8_t>(0));
    CheckFill(static_cast<uint8_t>(255));
    CheckFill(static_cast<float>(0.f));
    CheckFill(static_cast<float>(std::numeric_limits<float>::infinity()));
    CheckFill(static_cast<float>(std::nanf("")));
    CheckFill(static_cast<double>(0.f));
    CheckFill(static_cast<double>(std::numeric_limits<double>::infinity()));
    CheckFill(static_cast<double>(std::nan("")));
}

TEST_P(ArrayTest, CastFill) {
    CheckCastFill<float>(static_cast<int32_t>(1));
    CheckCastFill<int32_t>(static_cast<float>(1));
}

TEST_P(ArrayTest, IAdd) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        a += b;
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        a += b;
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        a += b;
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, IMul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        a *= b;
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        a *= b;
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        a *= b;
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, Add) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, true, false});
        Array o = a + b;
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 4, 6});
        Array o = a + b;
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 4, 6});
        Array o = a + b;
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, Mul) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, false, false, false});
        Array o = a * b;
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {1, 4, 9});
        Array o = a * b;
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {1, 4, 9});
        Array o = a * b;
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array c = a * b;
        Array o = a + c;
        AssertEqual<bool>(e, o);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        AssertEqual<int8_t>(e, o);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        Array c = a * b;
        Array o = a + c;
        AssertEqual<float>(e, o);
    }
}

TEST_P(ArrayTest, ChainedInplaceMath) {
    {
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        Array e = MakeArray<bool>({4, 1}, {true, true, false, false});
        b *= a;
        a += b;
        AssertEqual<bool>(e, a);
    }
    {
        Array a = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array b = MakeArray<int8_t>({3, 1}, {1, 2, 3});
        Array e = MakeArray<int8_t>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        AssertEqual<int8_t>(e, a);
    }
    {
        Array a = MakeArray<float>({3, 1}, {1, 2, 3});
        Array b = MakeArray<float>({3, 1}, {1, 2, 3});
        Array e = MakeArray<float>({3, 1}, {2, 6, 12});
        b *= a;
        a += b;
        AssertEqual<float>(e, a);
    }
}

TEST_P(ArrayTest, ComputationalGraph) {
    {
        // c = a + b
        // o = a * c
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        {
            auto a_node = a.node();
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
        }

        Array c = a + b;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            ASSERT_NE(c_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
            ASSERT_NE(c_op_node, nullptr);
            ASSERT_EQ(c_op_node->name(), "add");
        }

        Array o = a * c;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            auto c_node = c.node();
            auto o_node = o.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            ASSERT_NE(c_node, nullptr);
            ASSERT_NE(o_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            auto c_op_node = c_node->next_node();
            auto o_op_node = o_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
            ASSERT_NE(c_op_node, nullptr);
            ASSERT_NE(o_op_node, nullptr);
            ASSERT_EQ(c_op_node->name(), "add");
            ASSERT_EQ(o_op_node->name(), "mul");
        }
    }
}

TEST_P(ArrayTest, ComputationalGraphInplace) {
    {
        // a += b
        // a *= b
        Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
        Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
        auto a_node_1 = a.node();
        {
            auto a_node = a_node_1;
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(b_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_EQ(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
        }

        a += b;
        auto a_node_2 = a.node();
        {
            auto a_node = a_node_2;
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(a_node, a_node_1) << "a's node is not renewed";
            ASSERT_NE(b_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_NE(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
            ASSERT_EQ(a_op_node->name(), "add");
        }

        a *= b;
        {
            auto a_node = a.node();
            auto b_node = b.node();
            ASSERT_NE(a_node, nullptr);
            ASSERT_NE(a_node, a_node_1) << "a's node is not renewed";
            ASSERT_NE(a_node, a_node_2) << "a's node is not renewed";
            ASSERT_NE(b_node, nullptr);
            auto a_op_node = a_node->next_node();
            auto b_op_node = b_node->next_node();
            ASSERT_NE(a_op_node, nullptr);
            ASSERT_EQ(b_op_node, nullptr);
            ASSERT_EQ(a_op_node->name(), "mul");
        }
    }
}

TEST_P(ArrayTest, DeepCopy) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = a.DeepCopy();
    AssertEqual<bool>(a, b);
}

TEST_P(ArrayTest, AddBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
    Array o = a + b;

    auto op_node = o.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    AssertEqual<bool>(ga, go);
    AssertEqual<bool>(gb, go);
}

TEST_P(ArrayTest, IAddBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
    a += b;

    auto op_node = a.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    AssertEqual<bool>(ga, go);
    AssertEqual<bool>(gb, go);
}

TEST_P(ArrayTest, MulBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
    Array o = a * b;

    auto op_node = o.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    AssertEqual<bool>(ga, go * b);
    AssertEqual<bool>(gb, go * a);
}

TEST_P(ArrayTest, IMulBackward) {
    Array a = MakeArray<bool>({4, 1}, {true, true, false, false});
    Array b = MakeArray<bool>({4, 1}, {true, false, true, false});
    Array orig_a = a.DeepCopy();
    a *= b;

    auto op_node = a.node()->next_node();
    Array go = MakeArray<bool>({4, 1}, {true, true, true, true});
    Array ga = op_node->backward_functions()[0](go);
    Array gb = op_node->backward_functions()[1](go);

    AssertEqual<bool>(ga, go * b);
    AssertEqual<bool>(gb, go * orig_a);
}

INSTANTIATE_TEST_CASE_P(ForEachDevice, ArrayTest, ::testing::Values(
#ifdef XCHAINER_ENABLE_CUDA
                                                      std::string{"cuda"},
#endif  // XCHAINER_ENABLE_CUDA
                                                      std::string{"cpu"}));

}  // namespace
}  // namespace xchainer
