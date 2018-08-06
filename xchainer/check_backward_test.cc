#include "xchainer/numerical_gradient.h"

#include <algorithm>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <gsl/gsl>
#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/backprop_scope.h"
#include "xchainer/backward_builder.h"
#include "xchainer/backward_context.h"
#include "xchainer/check_backward.h"
#include "xchainer/context.h"
#include "xchainer/graph.h"
#include "xchainer/indexable_array.h"
#include "xchainer/indexer.h"
#include "xchainer/native/native_backend.h"
#include "xchainer/op_node.h"
#include "xchainer/shape.h"
#include "xchainer/testing/array.h"
#include "xchainer/testing/device_session.h"

namespace xchainer {
namespace {

using Arrays = std::vector<Array>;
using Fprop = std::function<std::vector<Array>(const std::vector<Array>&)>;

Arrays ForwardWithIncorrectBackward(const Arrays& inputs) {
    const Array& in = inputs[0];
    Array out = EmptyLike(in);

    BackwardBuilder bb{"incorrect_unary", in, out};
    if (BackwardBuilder::Target bt = bb.CreateTarget(0)) {
        bt.Define([](BackwardContext& bctx) {
            const Array& gout = bctx.output_grad();
            bctx.input_grad() = gout * gout;
        });
    }

    VisitDtype(in.dtype(), [&](auto pt) {
        using T = typename decltype(pt)::type;
        IndexableArray<const T> in_iarray{in};
        IndexableArray<T> out_iarray{out};
        Indexer<> indexer{out.shape()};

        for (auto it = indexer.It(0); it; ++it) {
            out_iarray[it] = in_iarray[it];
        }
    });

    return {out};
}

class CheckBackwardTest : public ::testing::TestWithParam<bool> {
protected:
    void SetUp() override {
        device_session_.emplace(DeviceId{native::NativeBackend::kDefaultName, 0});
        requires_grad_ = GetParam();
    }

    void TearDown() override { device_session_.reset(); }

protected:
    template <typename T>
    void CheckCheckBackward(
            bool expect_correct,
            const Fprop& fprop,
            const Shape& shape,
            const std::vector<T>& input_data,
            const std::vector<T>& grad_output_data,
            const std::vector<T>& eps_data,
            double atol,
            double rtol,
            const std::string& backprop_name) {
        Arrays inputs{testing::BuildArray(shape).WithData(input_data)};
        BackpropScope backprop_scope{backprop_name};

        if (requires_grad_) {
            inputs[0].RequireGrad(backprop_scope.backprop_id());
        }

        Arrays grad_outputs{testing::BuildArray(shape).WithData(grad_output_data)};
        Arrays eps{testing::BuildArray(shape).WithData(eps_data)};

        bool is_none_of_grad_required = std::none_of(inputs.begin(), inputs.end(), [&backprop_scope](const Array& input) {
            return input.IsGradRequired(backprop_scope.backprop_id());
        });

        if (expect_correct || is_none_of_grad_required) {
            // We cannot expect any failures in case none of the input std::vector<Array> require gradients
            CheckBackward(fprop, inputs, grad_outputs, eps, atol, rtol, backprop_scope.backprop_id());
        } else {
            // Catch the gtest failure expected to be generated by CheckBackward but without failing this test
            EXPECT_THROW(CheckBackward(fprop, inputs, grad_outputs, eps, atol, rtol, backprop_scope.backprop_id()), GradientCheckError);
        }
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
    bool requires_grad_{};
};

class CheckDoubleBackwardTest : public ::testing::Test {
protected:
    void SetUp() override { device_session_.emplace(DeviceId{native::NativeBackend::kDefaultName, 0}); }

    void TearDown() override { device_session_.reset(); }

protected:
    template <typename T>
    void CheckCheckDoubleBackward(
            const Fprop& fprop,
            const Shape& shape,
            const std::vector<T>& input_data,
            const std::vector<T>& grad_output_data,
            const std::vector<T>& grad_grad_input_data,
            const std::vector<T>& eps_input_data,
            const std::vector<T>& eps_grad_output_data,
            double atol,
            double rtol,
            const std::string& backprop_name) {
        Arrays inputs{testing::BuildArray(shape).WithData(input_data)};
        Arrays grad_outputs{testing::BuildArray(shape).WithData(grad_output_data)};
        Arrays grad_grad_inputs{testing::BuildArray(shape).WithData(grad_grad_input_data)};
        Arrays eps{testing::BuildArray(shape).WithData(eps_input_data), testing::BuildArray(shape).WithData(eps_grad_output_data)};
        BackpropScope backprop_scope{backprop_name};

        for (auto& input : inputs) {
            input.RequireGrad(backprop_scope.backprop_id());
        }
        for (auto& grad_output : grad_outputs) {
            grad_output.RequireGrad(backprop_scope.backprop_id());
        }

        // A failure occurs if backward computation and numerical gradients have differences
        CheckDoubleBackwardComputation(fprop, inputs, grad_outputs, grad_grad_inputs, eps, atol, rtol, backprop_scope.backprop_id());
    }

private:
    nonstd::optional<testing::DeviceSession> device_session_;
};

TEST_P(CheckBackwardTest, CorrectBackward) {
    using T = float;
    std::vector<T> input_data{1.f, 2.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    Fprop fprop = [](const Arrays& inputs) -> Arrays { return {inputs[0] * inputs[0]}; };
    CheckCheckBackward(true, fprop, {1, 3}, input_data, grad_output_data, eps_data, 1e-5, 1e-4, "graph_1");
}

TEST_P(CheckBackwardTest, CorrectBackwardWithNonDoubleDifferentiableFunction) {
    using T = float;
    std::vector<T> input_data{1.f, 2.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    Fprop fprop = [](const Arrays& inputs) -> Arrays { return {-inputs[0]}; };
    CheckCheckBackward(true, fprop, {1, 3}, input_data, grad_output_data, eps_data, 1e-5, 1e-4, "graph_1");
}

TEST_P(CheckBackwardTest, IncorrectBackward) {
    using T = float;
    std::vector<T> input_data{-2.f, 3.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    CheckCheckBackward(false, &ForwardWithIncorrectBackward, {1, 3}, input_data, grad_output_data, eps_data, 1e-5, 1e-4, "graph_1");
}

TEST_P(CheckBackwardTest, IncorrectBackwardIdenticalInputOutput) {
    using T = float;
    std::vector<T> input_data{-2.f, 3.f, 1.f};
    std::vector<T> grad_output_data{0.f, -2.f, 1.f};
    std::vector<T> eps_data{1e-3f, 1e-3f, 1e-3f};
    CheckCheckBackward(
            false,
            [](const std::vector<Array>& xs) -> std::vector<Array> { return {xs[0].AsType(xs[0].dtype(), false)}; },
            {1, 3},
            input_data,
            grad_output_data,
            eps_data,
            1e-4,
            1e-3,
            "graph_1");
}

TEST_F(CheckDoubleBackwardTest, CorrectBackward) {
    using T = float;
    std::vector<T> input_data{1.f, 2.f, 3.f};
    std::vector<T> grad_output_data{1.f, 1.f, 1.f};
    std::vector<T> grad_grad_input_data{1.f, 1.f, 1.f};
    std::vector<T> eps_input_data{1e-3f, 1e-3f, 1e-3f};
    std::vector<T> eps_grad_output_data{1e-3f, 1e-3f, 1e-3f};
    Fprop fprop = [](const Arrays& inputs) -> Arrays { return {inputs[0] * inputs[0]}; };
    CheckCheckDoubleBackward(
            fprop, {1, 3}, input_data, grad_output_data, grad_grad_input_data, eps_input_data, eps_grad_output_data, 1e-4, 1e-3, "graph_1");
}

INSTANTIATE_TEST_CASE_P(ForEachSingleSetRequiresGrad, CheckBackwardTest, ::testing::Bool());

}  // namespace
}  // namespace xchainer
