#include "xchainer/routines/normalization.h"

#include <cstdint>
#include <memory>
#include <tuple>

#include "xchainer/array.h"
#include "xchainer/axes.h"
#include "xchainer/backward.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/error.h"
#include "xchainer/scalar.h"
#include "xchainer/shape.h"

namespace xchainer {
namespace {

struct PreprocessBatchNormResult {
    // Arrays are reshaped if necessary
    Array gamma;
    Array beta;
    Array mean;
    Array var;
    Axes sorted_axis;
};

// Reshapes the array. If the shape is unchanged, an array with identical array body is returned. Note that xchainer::Reshape() returns
// a view with different array body if the shape is unchanged.
Array ReshapeOrIdentity(const Array& a, const Shape& shape) {
    if (a.shape() == shape) {
        return a;
    }
    return a.Reshape(shape);
}

// Reshapes the input arrays (except x) as needed and makes them constant arrays.
// Sorted axes is also returned.
PreprocessBatchNormResult PreprocessBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, const OptionalAxes& axis) {
    Dtype dtype = x.dtype();
    CheckEqual(dtype, gamma.dtype());
    CheckEqual(dtype, beta.dtype());
    CheckEqual(dtype, mean.dtype());
    CheckEqual(dtype, var.dtype());

    Axes sorted_axis = axis.has_value() ? internal::GetSortedAxes(*axis, x.ndim()) : Axes{0};

    Shape reduced_shape = internal::ReduceShape(x.shape(), sorted_axis, true);
    int64_t reduced_size = reduced_shape.GetTotalSize();

    if (gamma.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Gamma must have the same size as the reduced input. Actual: ", gamma.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (beta.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Beta must have the same size as the reduced input. Actual: ", beta.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (mean.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Mean must have the same size as the reduced input. Actual: ", mean.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }
    if (var.GetTotalSize() != reduced_size) {
        throw DimensionError{
                "Variance must have the same size as the reduced input. Actual: ", var.GetTotalSize(), ". Expected: ", reduced_size, "."};
    }

    Array x_const = x.AsConstant();
    Array gamma_reshaped = ReshapeOrIdentity(gamma.AsConstant(), reduced_shape);
    Array beta_reshaped = ReshapeOrIdentity(beta.AsConstant(), reduced_shape);
    Array mean_reshaped = ReshapeOrIdentity(mean.AsConstant(), reduced_shape);
    Array var_reshaped = ReshapeOrIdentity(var.AsConstant(), reduced_shape);
    assert(gamma_reshaped.data() == gamma.data());  // No data copy should occur
    assert(beta_reshaped.data() == beta.data());
    assert(mean_reshaped.data() == mean.data());
    assert(var_reshaped.data() == var.data());

    return {std::move(gamma_reshaped), std::move(beta_reshaped), std::move(mean_reshaped), std::move(var_reshaped), sorted_axis};
}

}  // namespace

Array BatchNorm(
        const Array& x,
        const Array& gamma,
        const Array& beta,
        const Array& running_mean,
        const Array& running_var,
        Scalar eps,
        Scalar decay,
        const OptionalAxes& axis) {
    PreprocessBatchNormResult result = PreprocessBatchNorm(x, gamma, beta, running_mean, running_var, axis);
    std::shared_ptr<BatchNormForwardBackward> fb = x.device().GetBatchNormForwardBackward();

    Array out = fb->Forward(x.AsConstant(), result.gamma, result.beta, result.mean, result.var, eps, decay, result.sorted_axis);

    {
        BackwardBuilder bb{"batch_norm", {out}};
        if (!x.IsConstant() || !gamma.IsConstant() || !beta.IsConstant()) {
            bb.Define(
                    {x, gamma, beta},
                    [ fb = std::move(fb), x = x.AsConstant(), gamma_reshaped = result.gamma, eps, sorted_axis = result.sorted_axis ](
                            BackwardContext & bctx) {
                        const Array& gout = bctx.output_grad();
                        auto ginputs = fb->Backward(x, gamma_reshaped, gout, eps, sorted_axis);
                        // TODO(niboshi): Implement double backward

                        // TODO(niboshi): Implement a convenient function in BackwardContext to move arrays from a container
                        for (size_t i = 0; i < ginputs.size(); ++i) {
                            bctx.input_grad(i) = std::move(ginputs[i]);
                        }
                    });
        }
    }

    return out;
}

Array FixedBatchNorm(
        const Array& x, const Array& gamma, const Array& beta, const Array& mean, const Array& var, Scalar eps, const OptionalAxes& axis) {
    PreprocessBatchNormResult result = PreprocessBatchNorm(x, gamma, beta, mean, var, axis);
    return x.device().FixedBatchNorm(x.AsConstant(), result.gamma, result.beta, result.mean, result.var, eps, result.sorted_axis);
}

}  // namespace xchainer
