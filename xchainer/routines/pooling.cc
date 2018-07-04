#include "xchainer/routines/pooling.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xchainer/array.h"
#include "xchainer/backward.h"
#include "xchainer/constant.h"
#include "xchainer/device.h"
#include "xchainer/error.h"
#include "xchainer/routines/math.h"
#include "xchainer/stack_vector.h"

namespace xchainer {
namespace {

void CheckPoolInputs(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad) {
    int8_t ndim = x.ndim() - 2;  // Number of spacial dimensions.
    if (static_cast<int8_t>(kernel_size.size()) != ndim) {
        throw DimensionError{"Wrong numbers of kernel size dimensions ", kernel_size.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(stride.size()) != ndim) {
        throw DimensionError{"Wrong numbers of strides ", stride.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (static_cast<int8_t>(pad.size()) != ndim) {
        throw DimensionError{"Wrong numbers of paddings ", pad.size(), " for input with ", x.ndim(), " dimensions."};
    }
    if (ndim == 0) {
        throw DimensionError{"Pooling operation requires at least one spatial dimension."};
    }
}

}  // namespace

Array MaxPool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        bool cover_all) {
    CheckPoolInputs(x, kernel_size, stride, pad);
    std::unique_ptr<MaxPoolForwardBackward> fb = x.device().GetMaxPoolForwardBackward(kernel_size, stride, pad, cover_all);
    Array out = fb->Forward(x);

    // Supporting arbitrary number of backwards using a recursive definition.
    // TODO(hvy): Test backward of double backward.
    struct MaxPoolBwd {
        void operator()(BackwardContext& bctx1) {
            const Array& gout = bctx1.output_grad();
            Array gx = fb->Backward(gout);
            {
                BackwardBuilder bb2{"max_pooling_backward", gx};
                if (gout.IsBackpropRequired()) {
                    bb2.Define({gout}, [this, gout](BackwardContext& bctx2) {
                        const Array& ggx = bctx2.output_grad();
                        Array ggout = fb->DoubleBackward(ggx);
                        // Make ggout further backpropable.
                        {
                            BackwardBuilder bb3{"max_pooling_double_backward", ggout};
                            if (ggx.IsBackpropRequired()) {
                                bb3.Define({ggx}, *this);
                            }
                        }
                        bctx2.input_grad() = ggout;
                    });
                }
            }
            bctx1.input_grad() = gx;
        }

        Array x;
        StackVector<int64_t, kMaxNdim> kernel_size;
        StackVector<int64_t, kMaxNdim> stride;
        StackVector<int64_t, kMaxNdim> pad;
        bool cover_all;
        std::shared_ptr<MaxPoolForwardBackward> fb;
    };

    {
        BackwardBuilder bb1{"max_pooling", out};
        if (x.IsBackpropRequired()) {
            bb1.Define({x}, MaxPoolBwd{x, kernel_size, stride, pad, cover_all, std::move(fb)});
        }
    }
    return out;
}

Array AveragePool(
        const Array& x,
        const StackVector<int64_t, kMaxNdim>& kernel_size,
        const StackVector<int64_t, kMaxNdim>& stride,
        const StackVector<int64_t, kMaxNdim>& pad,
        AveragePoolPadMode pad_mode) {
    CheckPoolInputs(x, kernel_size, stride, pad);
    std::shared_ptr<AveragePoolForwardBackward> fb = x.device().GetAveragePoolForwardBackward(kernel_size, stride, pad, pad_mode);
    Array out = fb->Forward(x);
    {
        BackwardBuilder bb1{"average_pool", out};
        if (x.IsBackpropRequired()) {
            bb1.Define({x}, [ fb = std::move(fb), x, kernel_size, stride, pad, pad_mode ](BackwardContext & bctx) {
                const Array& gout = bctx.output_grad();
                Array gx = fb->Backward(gout);
                {
                    BackwardBuilder bb2{"average_pool_backward", gx};
                    if (gout.IsBackpropRequired()) {
                        bb2.Define({gout}, [kernel_size, stride, pad, pad_mode](BackwardContext& bctx2) {
                            const Array& ggx = bctx2.output_grad();
                            bctx2.input_grad() = AveragePool(ggx, kernel_size, stride, pad, pad_mode);
                        });
                    }
                }
                bctx.input_grad() = gx;
            });
        }
    }
    return out;
}

}  // namespace xchainer
