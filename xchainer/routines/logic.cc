#include "xchainer/routines/logic.h"

#include "xchainer/array.h"
#include "xchainer/device.h"
#include "xchainer/dtype.h"
#include "xchainer/routines/creation.h"
#include "xchainer/routines/manipulation.h"
#include "xchainer/shape.h"

namespace xchainer {

Array Equal(const Array& x1, const Array& x2) {
    CheckEqual(x1.dtype(), x2.dtype());

    auto func = [](const Array& x1, const Array& x2) {
        Array out = Empty(x1.shape(), Dtype::kBool, x1.device());
        x1.device().Equal(x1, x2, out);
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

}  // namespace xchainer
