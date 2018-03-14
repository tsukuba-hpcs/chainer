#include <cassert>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backend.h"
#include "xchainer/backward.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native_backend.h"
#include "xchainer/native_device.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/context_session.h"
#include "xchainer/testing/util.h"

namespace xchainer {
namespace {

// Test configuration class
class TestConfig {
public:
    // backend <0> can transfer from backend <1> to backend <2>
    using Key = std::tuple<int, int, int>;

    TestConfig() {
        set_.insert({
                Key{0, 0, 0},  // backend0 can transfer with itself
                Key{0, 0, 1},  // backend0 can transfer to backend1
                Key{0, 2, 0},  // backend0 can transfer from backend2
                               // backend0 and backend3 are incompatible
        });
    }

    // Returns true if the backend `who` can transfer data from backend `from` to backend `to`
    bool CanTransfer(int who, int from, int to) { return set_.find(std::make_tuple(who, from, to)) != set_.end(); }

    // Return the number of test backends
    int num_backends() { return 4; }

private:
    std::set<Key> set_;
};

// Instantiate the global test configuration
TestConfig g_config;

// Test backend class
class TestBackend : public NativeBackend {
public:
    TestBackend(Context& context, int num) : NativeBackend(context), num_(num) {}

    int num() const { return num_; }

    std::string GetName() const override { return "backend" + std::to_string(num_); }

    bool SupportsTransfer(Device& src_device, Device& dst_device) override {
        int src_num = dynamic_cast<TestBackend&>(src_device.backend()).num();
        int dst_num = dynamic_cast<TestBackend&>(dst_device.backend()).num();
        return g_config.CanTransfer(num_, src_num, dst_num);
    }

    int GetDeviceCount() const override { return 1; }

    std::unique_ptr<Device> CreateDevice(int index) override {
        assert(index == 0);
        return std::make_unique<NativeDevice>(*this, index);
    }

private:
    int num_;
};

// Test fixture for compatible transfer
class ArrayToDeviceCompatibleTest : public ::testing::TestWithParam<::testing::tuple<int, int, int>> {
protected:
    void SetUp() override {
        context_session_.emplace();

        default_backend_num_ = ::testing::get<0>(GetParam());
        src_backend_num_ = ::testing::get<1>(GetParam());
        dst_backend_num_ = ::testing::get<2>(GetParam());

        backends_.clear();
        for (int i = 0; i < g_config.num_backends(); ++i) {
            backends_.emplace_back(std::make_unique<TestBackend>(context_session_->context(), i));
        }

        // Set default backend (only if default_backend_num is non-negative)
        if (default_backend_num_ >= 0) {
            device_scope_ = std::make_unique<DeviceScope>(*GetDefaultDevicePtr());
        }
    }

    void TearDown() override {
        device_scope_.reset();
        context_session_.reset();
        backends_.clear();
    }

    Device* GetDefaultDevicePtr() {
        if (default_backend_num_ < 0) {
            return nullptr;
        }
        return &backends_[default_backend_num_]->GetDevice(0);
    }

    Device& GetSourceDevice() { return backends_[src_backend_num_]->GetDevice(0); }

    Device& GetDestinationDevice() { return backends_[dst_backend_num_]->GetDevice(0); }

private:
    nonstd::optional<testing::ContextSession> context_session_;
    std::unique_ptr<DeviceScope> device_scope_;
    std::vector<std::unique_ptr<TestBackend>> backends_;
    int default_backend_num_{};
    int src_backend_num_{};
    int dst_backend_num_{};
};

void ExpectArraysEqual(const Array& expected, const Array& actual) {
    EXPECT_EQ(expected.dtype(), actual.dtype());
    EXPECT_EQ(expected.shape(), actual.shape());
    VisitDtype(expected.dtype(), [&expected, &actual](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> expected_iarray{expected};
        IndexableArray<const T> actual_iarray{actual};
        Indexer<> indexer{expected.shape()};

        for (int64_t i = 0; i < indexer.total_size(); ++i) {
            indexer.Set(i);
            EXPECT_EQ(expected_iarray[indexer], actual_iarray[indexer]);
        }
    });
}

TEST_P(ArrayToDeviceCompatibleTest, ToDevice) {
    Device& src_dev = GetSourceDevice();
    Device& dst_dev = GetDestinationDevice();
    Device& default_device = GetDefaultDevice();

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = Array::FromBuffer({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), src_dev);

    // Transfer
    Array b = a.ToDevice(dst_dev);

    EXPECT_EQ(&b.device(), &dst_dev) << "Array::ToDevice must allocate an array on the specified device.";
    EXPECT_EQ(&a.device(), &src_dev) << "Array::ToDevice must not alter the device of the original array.";
    if (&dst_dev == &src_dev) {
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice must return an alias in same-device transfer.";
    }
    EXPECT_EQ(&GetDefaultDevice(), &default_device) << "Array::ToDevice must not alter the default device.";
    ExpectArraysEqual(a, b);
}

TEST_P(ArrayToDeviceCompatibleTest, ToDeviceNonContiguous) {
    Device& src_dev = GetSourceDevice();
    Device& dst_dev = GetDestinationDevice();
    Device& default_device = GetDefaultDevice();

    Array a = testing::MakeArray({2, 4})          //
                      .WithLinearData<int32_t>()  //
                      .WithPadding(1)             //
                      .WithDevice(src_dev);

    // Transfer
    Array b = a.ToDevice(dst_dev);

    EXPECT_EQ(&b.device(), &dst_dev) << "Array::ToDevice must allocate an array on the specified device.";
    EXPECT_EQ(&a.device(), &src_dev) << "Array::ToDevice must not alter the device of the original array.";
    if (&dst_dev == &src_dev) {
        EXPECT_EQ(a.data().get(), b.data().get()) << "Array::ToDevice must return an alias in same-device transfer.";
    }
    EXPECT_EQ(&GetDefaultDevice(), &default_device) << "Array::ToDevice must not alter the default device.";
    EXPECT_EQ(&src_dev != &dst_dev, b.IsContiguous()) << "Array::ToDevice must return a contiguous array if device transfer occurs.";
    ExpectArraysEqual(a, b);
}

INSTANTIATE_TEST_CASE_P(
        BackendCombination,
        ArrayToDeviceCompatibleTest,
        ::testing::Values(
                std::make_tuple(-1, 0, 0),   // transfer between same devices
                std::make_tuple(-1, 0, 1),   // transfer to 1
                std::make_tuple(-1, 2, 0),   // transfer from 2
                std::make_tuple(2, 0, 1)));  // checks default device does not change

// Test for incompatible transfer
TEST(ArrayToDeviceIncompatibleTest, ToDeviceIncompatible) {
    testing::ContextSession context_session;
    TestBackend src_backend{context_session.context(), 0};  // incompatible configuration
    TestBackend dst_backend{context_session.context(), 3};

    Device& src_dev = src_backend.GetDevice(0);
    Device& dst_dev = dst_backend.GetDevice(0);

    // Allocate the source array
    float data[] = {1.0f, 2.0f};
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    Array a = Array::FromBuffer({2, 1}, Dtype::kFloat32, std::shared_ptr<float>(data, nop), src_dev);

    // Transfer
    EXPECT_THROW(a.ToDevice(dst_dev), XchainerError) << "Array::ToDevice must throw if incompatible device is given.";
}

TEST(ArrayToDeviceArithmeticTest, Arithmetic) {
    testing::ContextSession context_session;
    NativeBackend backend{context_session.context()};

    XCHAINER_REQUIRE_DEVICE(backend, 3);

    Device& dev0 = backend.GetDevice(0);
    Device& dev1 = backend.GetDevice(1);
    Device& dev2 = backend.GetDevice(2);  // default device
    DeviceScope device_scope{dev2};

    // Allocate the source array
    auto nop = [](void* p) {
        (void)p;  // unused
    };
    float data0[]{1.0f, 2.0f};
    float data1[]{3.0f, 4.0f};
    float data2[]{5.0f, 6.0f};
    Shape shape{2, 1};
    Array a0 = Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(data0, nop), dev0);
    Array a1 = Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(data1, nop), dev0);
    Array a2 = Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(data2, nop), dev1);
    a0.RequireGrad();
    a1.RequireGrad();
    a2.RequireGrad();

    // Test preconditions
    ASSERT_EQ(&dev0, &a0.device());
    ASSERT_EQ(&dev0, &a1.device());
    ASSERT_EQ(&dev1, &a2.device());

    // Forward
    Array b = a0 * a1;
    Array b_dev1 = b.ToDevice(dev1);
    Array c = b_dev1 + a2;

    ASSERT_TRUE(c.IsGradRequired());
    ASSERT_TRUE(b_dev1.IsGradRequired());
    ASSERT_TRUE(b.IsGradRequired());

    // Check forward correctness
    EXPECT_EQ(&dev0, &b.device());
    EXPECT_EQ(&dev1, &b_dev1.device());
    EXPECT_EQ(&dev1, &c.device());
    EXPECT_TRUE(c.IsGradRequired());
    EXPECT_TRUE(b_dev1.IsGradRequired());
    EXPECT_TRUE(b.IsGradRequired());
    float datay[]{8.0f, 14.0f};  // d0 * d1 + d2
    ExpectArraysEqual(c, Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(datay, nop)));

    // Backward
    Backward(c);

    // Check backward correctness
    ASSERT_TRUE(a0.GetGrad().has_value());
    ASSERT_TRUE(a1.GetGrad().has_value());
    ASSERT_TRUE(a2.GetGrad().has_value());
    ASSERT_TRUE(c.GetGrad().has_value());
    EXPECT_EQ(&dev0, &a0.GetGrad()->device());
    EXPECT_EQ(&dev0, &a1.GetGrad()->device());
    EXPECT_EQ(&dev1, &a2.GetGrad()->device());
    EXPECT_EQ(&dev1, &c.GetGrad()->device());
    float data0_grad[]{3.0f, 4.0f};
    float data1_grad[]{1.0f, 2.0f};
    float data2_grad[]{1.0f, 1.0f};
    ExpectArraysEqual(*a0.GetGrad(), Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(data0_grad, nop)));
    ExpectArraysEqual(*a1.GetGrad(), Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(data1_grad, nop)));
    ExpectArraysEqual(*a2.GetGrad(), Array::FromBuffer(shape, Dtype::kFloat32, std::shared_ptr<float>(data2_grad, nop)));
}

}  // namespace
}  // namespace xchainer
