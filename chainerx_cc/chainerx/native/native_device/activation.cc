#include "chainerx/native/native_device.h"

#include <cmath>
#include <cstdint>

#include "chainerx/array.h"
#include "chainerx/device.h"
#include "chainerx/dtype.h"
#include "chainerx/native/elementwise.h"
#include "chainerx/native/op_regist.h"
#include "chainerx/numeric.h"
#include "chainerx/routines/math.h"
#include "chainerx/routines/type_util.h"
#include "chainerx/scalar.h"

namespace chainerx {
namespace native {
namespace {

class NativeIfLessElseASSAOp : public IfLessElseASSAOp {
public:
    void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override {
        x1.device().CheckDevicesCompatible(x1, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        VisitDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            VisitDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                struct Impl {
                    void operator()(int64_t /*i*/, In x1, Out neg, Out& out) { out = x1 < x2 ? pos : neg; }
                    In x2;
                    Out pos;
                };
                Elementwise<const In, const Out, Out>(Impl{static_cast<In>(x2), static_cast<Out>(pos)}, x1_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(IfLessElseASSAOp, NativeIfLessElseASSAOp);

class NativeIfGreaterElseASSAOp : public IfGreaterElseASSAOp {
public:
    void Call(const Array& x1, Scalar x2, Scalar pos, const Array& neg, const Array& out) override {
        x1.device().CheckDevicesCompatible(x1, neg, out);
        Dtype x_dtype = ResultType(x1, x2);
        const Array& x1_cast = x1.dtype() == x_dtype ? x1 : x1.AsType(x_dtype);
        const Array& neg_cast = neg.dtype() == out.dtype() ? neg : neg.AsType(out.dtype());
        VisitDtype(x_dtype, [&](auto x_pt) {
            using In = typename decltype(x_pt)::type;
            VisitDtype(out.dtype(), [&](auto pt) {
                using Out = typename decltype(pt)::type;
                struct Impl {
                    void operator()(int64_t /*i*/, In x1, Out neg, Out& out) { out = x1 > x2 ? pos : neg; }
                    In x2;
                    Out pos;
                };
                Elementwise<const In, const Out, Out>(Impl{static_cast<In>(x2), static_cast<Out>(pos)}, x1_cast, neg_cast, out);
            });
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(IfGreaterElseASSAOp, NativeIfGreaterElseASSAOp);

class NativeIfGreaterElseAAAAOp : public IfGreaterElseAAAAOp {
public:
    void Call(const Array& x1, const Array& x2, const Array& pos, const Array& neg, const Array& out) override {
        x1.device().CheckDevicesCompatible(x1, x2, pos, neg, out);
        VisitDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x1, T x2, T pos, T neg, T& out) { out = x1 > x2 ? pos : neg; }
            };
            Elementwise<const T, const T, const T, const T, T>(Impl{}, x1, x2, pos, neg, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(IfGreaterElseAAAAOp, NativeIfGreaterElseAAAAOp);

class NativeTanhOp : public TanhOp {
public:
    void Call(const Array& x, const Array& out) override {
        x.device().CheckDevicesCompatible(x, out);
        const Array& x_cast = x.dtype() == out.dtype() ? x : x.AsType(out.dtype());
        VisitFloatingPointDtype(out.dtype(), [&](auto pt) {
            using T = typename decltype(pt)::type;
            struct Impl {
                void operator()(int64_t /*i*/, T x, T& out) { out = chainerx::Tanh(x); }
            };
            Elementwise<const T, T>(Impl{}, x_cast, out);
        });
    }
};

CHAINERX_REGISTER_OP_NATIVE(TanhOp, NativeTanhOp);

}  // namespace
}  // namespace native
}  // namespace chainerx
