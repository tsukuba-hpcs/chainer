#include "xchainer/routines/math.h"

#include <cstdint>
#include <numeric>
#include <vector>

#include <nonstd/optional.hpp>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/dtype.h"
#include "xchainer/enum.h"
#include "xchainer/error.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {

Array Negative(const Array& x) {
    if (x.dtype() == Dtype::kBool) {
        throw DtypeError{"Cannot negative a boolean array."};
    }
    return Multiply(x, Scalar{-1, x.dtype()});
}

namespace {

// Called from Add, Subtract, Multiply, Divide, etc. to handle broadcasting.
template <typename Impl>
Array BroadcastBinary(Impl&& impl, const Array& x1, const Array& x2) {
    auto func = [&impl](const Array& x1, const Array& x2) -> Array {
        Array out = EmptyLike(x1, x1.device());
        impl(x1, x2, out);
        return out;
    };

    if (x1.shape() == x2.shape()) {
        return func(x1, x2);
    }
    Shape result_shape = internal::BroadcastShapes(x1.shape(), x2.shape());
    if (x1.shape() == result_shape) {
        return func(x1, x2.BroadcastTo(result_shape));
    }
    if (x2.shape() == result_shape) {
        return func(x1.BroadcastTo(result_shape), x2);
    }
    return func(x1.BroadcastTo(result_shape), x2.BroadcastTo(result_shape));
}

// Called from IAdd, ISubtract, IMultiply, IDivide, etc. to handle broadcasting.
template <typename Impl>
void BroadcastBinaryInPlace(Impl&& impl, const Array& x1, const Array& x2) {
    if (x1.shape() == x2.shape()) {
        impl(x1, x2, x1);
    } else {
        impl(x1, x2.BroadcastTo(x1.shape()), x1);
    }
}

template <typename Impl>
Array Binary(Impl&& impl, const Array& x1, Scalar x2) {
    Array out = EmptyLike(x1, x1.device());
    impl(x1, x2, out);
    return out;
}

template <typename Impl>
void BinaryInPlace(Impl&& impl, const Array& x1, Scalar x2) {
    impl(x1, x2, x1);
}

void AddImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) -> Array {
        return gout.AsConstant(graph_ids_to_stop_gradient, CopyKind::kCopy);
    };
    auto x2_backward_function = [](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) -> Array {
        return gout.AsConstant(graph_ids_to_stop_gradient, CopyKind::kCopy);
    };
    internal::SetUpOpNodes("add", {x1, x2}, out, {x1_backward_function, x2_backward_function});

    x1.device().Add(x1, x2, out);
}

void AddASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion
    auto x1_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    internal::SetUpOpNodes("add_scalar", {x1}, out, {x1_backward_function});

    x1.device().AddAS(x1, x2, out);
}

}  // namespace

namespace internal {

void IAdd(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&AddImpl, x1, x2); }

void IAdd(const Array& x1, Scalar x2) { BinaryInPlace(&AddASImpl, x1, x2); }

}  // namespace internal

Array Add(const Array& x1, const Array& x2) { return BroadcastBinary(&AddImpl, x1, x2); }

Array Add(const Array& x1, Scalar x2) { return Binary(&AddASImpl, x1, x2); }

Array Add(Scalar x1, const Array& x2) { return Add(x2, x1); }

namespace {

void SubtractImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(niboshi): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    auto x2_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return -gout; };
    internal::SetUpOpNodes("subtract", {x1, x2}, out, {x1_backward_function, x2_backward_function});

    x1.device().Subtract(x1, x2, out);
}

void SubtractASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion
    auto x1_backward_function = [](const Array& gout, const std::vector<GraphId>&) -> Array { return gout; };
    internal::SetUpOpNodes("subtract_scalar", {x1}, out, {x1_backward_function});

    x1.device().SubtractAS(x1, x2, out);
}

}  // namespace

namespace internal {

void ISubtract(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&SubtractImpl, x1, x2); }

void ISubtract(const Array& x1, Scalar x2) { BinaryInPlace(&SubtractASImpl, x1, x2); }

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2) { return BroadcastBinary(&SubtractImpl, x1, x2); }

Array Subtract(const Array& x1, Scalar x2) { return Binary(&SubtractASImpl, x1, x2); }

Array Subtract(Scalar x1, const Array& x2) { return Add(-x2, x1); }

namespace {

void MultiplyImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(sonots): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [other = x2](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)->Array {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    auto x2_backward_function = [other = x1](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient)->Array {
        return gout * other.AsConstant(graph_ids_to_stop_gradient);
    };
    internal::SetUpOpNodes("multiply", {x1, x2}, out, {x1_backward_function, x2_backward_function});

    x1.device().Multiply(x1, x2, out);
}

void MultiplyASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion
    auto x1_backward_function = [other = x2](const Array& gout, const std::vector<GraphId>&)->Array { return gout * other; };
    internal::SetUpOpNodes("multiply_scalar", {x1}, out, {x1_backward_function});

    x1.device().MultiplyAS(x1, x2, out);
}

}  // namespace

namespace internal {

void IMultiply(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&MultiplyImpl, x1, x2); }

void IMultiply(const Array& x1, Scalar x2) { BinaryInPlace(&MultiplyASImpl, x1, x2); }

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2) { return BroadcastBinary(&MultiplyImpl, x1, x2); }

Array Multiply(const Array& x1, Scalar x2) { return Binary(&MultiplyASImpl, x1, x2); }

Array Multiply(Scalar x1, const Array& x2) { return Multiply(x2, x1); }

namespace {

void DivideImpl(const Array& x1, const Array& x2, const Array& out) {
    // TODO(niboshi): The behavior should be true division for integral dtypes. Currently it's rounding towards zero.
    // TODO(niboshi): dtype conversion
    CheckEqual(x1.dtype(), x2.dtype());
    CheckEqual(x1.shape(), x2.shape());

    auto x1_backward_function = [x2](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) -> Array {
        return gout / x2.AsConstant(graph_ids_to_stop_gradient);
    };
    auto x2_backward_function = [x1, x2](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) -> Array {
        Array lhs_const = x1.AsConstant(graph_ids_to_stop_gradient);
        Array rhs_const = x2.AsConstant(graph_ids_to_stop_gradient);
        return -gout * lhs_const / (rhs_const * rhs_const);
    };
    internal::SetUpOpNodes("divide", {x1, x2}, out, {x1_backward_function, x2_backward_function});

    x1.device().Divide(x1, x2, out);
}

void DivideASImpl(const Array& x1, Scalar x2, const Array& out) {
    // TODO(hvy): dtype conversion
    auto x1_backward_function = [other = x2](const Array& gout, const std::vector<GraphId>&)->Array { return gout / other; };
    internal::SetUpOpNodes("divide_scalar", {x1}, out, {x1_backward_function});

    x1.device().DivideAS(x1, x2, out);
}

}  // namespace

namespace internal {

void IDivide(const Array& x1, const Array& x2) { BroadcastBinaryInPlace(&DivideImpl, x1, x2); }

void IDivide(const Array& x1, Scalar x2) { BinaryInPlace(&DivideASImpl, x1, x2); }

}  // namespace internal

Array Divide(const Array& x1, const Array& x2) { return BroadcastBinary(&DivideImpl, x1, x2); }

Array Divide(const Array& x1, Scalar x2) { return Binary(&DivideASImpl, x1, x2); }

Array Divide(Scalar /*x1*/, const Array& /*x2*/) { throw NotImplementedError{"Scalar / Array division is not yet supported."}; }

Array Sum(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::Reduced(a.shape(), a.dtype(), sorted_axis, keepdims, a.device());
    a.device().Sum(a, sorted_axis, out);

    auto backward_function = [ sorted_axis, in_shape = a.shape(), keepdims ](const Array& gout, const std::vector<GraphId>&) {
        assert(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

        if (!(in_shape.ndim() == 0 || sorted_axis.empty() || keepdims)) {
            Shape out_shape_broadcastable = gout.shape();
            for (auto axis : sorted_axis) {
                out_shape_broadcastable.insert(out_shape_broadcastable.begin() + axis, 1);
            }
            return gout.Reshape(out_shape_broadcastable).BroadcastTo(in_shape);
        }
        return gout.BroadcastTo(in_shape);
    };
    internal::SetUpOpNodes("sum", {a}, out, {backward_function});

    return out;
}

Array AMax(const Array& a, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, a.ndim());
    Array out = internal::Reduced(a.shape(), a.dtype(), sorted_axis, keepdims, a.device());

    for (int8_t i : sorted_axis) {
        if (a.shape()[i] == 0) {
            throw DimensionError{"cannot compute the maximum along zero-sized axis"};
        }
    }

    a.device().AMax(a, sorted_axis, out);

    auto backward_function = [sorted_axis, a, out](const Array& gout, const std::vector<GraphId>&) {
        assert(std::is_sorted(sorted_axis.begin(), sorted_axis.end()));

        // Add broadcastable dimensions to out and gout
        // for each one that was reduced in the forward operation
        Shape shape = internal::ReduceShape(a.shape(), sorted_axis, true);
        Array reshaped_gout = gout.Reshape(shape);
        Array reshaped_out = out.AsConstant(CopyKind::kView).Reshape(shape);

        // Compute the gradient
        Array cond = (a == reshaped_out);
        Array broadcasted_gout = reshaped_gout.BroadcastTo(cond.shape());
        // TODO(sonots): Use `where` if it becomes available.
        return broadcasted_gout * cond.AsType(gout.dtype(), false);
    };
    internal::SetUpOpNodes("amax", {a}, out, {backward_function});

    return out;
}

namespace {

// Calculates: x1 < x2 ? pos : neg
// Can only differentiate with respect to neg.
Array IfLessElse(const Array& x1, Scalar x2, Scalar pos, const Array& neg) {
    Array out = EmptyLike(x1, x1.device());
    x1.device().IfLessElseASSA(x1, x2, pos, neg, out);

    auto backward_function = [x1, x2](const Array& gout, const std::vector<GraphId>&) {
        return IfLessElse(x1, x2, Scalar{0, gout.dtype()}, gout);
    };
    internal::SetUpOpNodes("if-less-else", {neg}, out, {backward_function});

    return out;
}

}  // namespace

Array Maximum(const Array& x1, Scalar x2) {
    return IfLessElse(x1, x2, x2, x1);  // x1 < x2 ? x2 : x1
}

Array Maximum(Scalar x1, const Array& x2) { return Maximum(x2, x1); }

Array Exp(const Array& x) {
    Array out = EmptyLike(x, x.device());
    x.device().Exp(x, out);

    auto backward_function = [x](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return Exp(x.AsConstant(graph_ids_to_stop_gradient)) * gout;
    };
    internal::SetUpOpNodes("exp", {x}, out, {backward_function});

    return out;
}

Array Log(const Array& x) {
    Array out = EmptyLike(x, x.device());
    x.device().Log(x, out);

    auto backward_function = [x](const Array& gout, const std::vector<GraphId>& graph_ids_to_stop_gradient) {
        return gout / x.AsConstant(graph_ids_to_stop_gradient);
    };
    internal::SetUpOpNodes("log", {x}, out, {backward_function});

    return out;
}

Array LogSumExp(const Array& x, const OptionalAxes& axis, bool keepdims) {
    Axes sorted_axis = internal::GetSortedAxesOrAll(axis, x.ndim());
    Array xmax = AMax(x, sorted_axis, true);
    Array logs = Log(Sum(Exp(x - xmax), sorted_axis, keepdims));
    return (keepdims ? xmax : Squeeze(xmax, axis)) + logs;
}

Array LogSoftmax(const Array& x, const OptionalAxes& axis) { return x - LogSumExp(x, axis.has_value() ? axis : OptionalAxes{1}, true); }

}  // namespace xchainer
