#pragma once

#include <cstdint>

#include <nonstd/optional.hpp>

#include "chainerx/array.h"
#include "chainerx/axes.h"
#include "chainerx/scalar.h"

namespace chainerx {

class AddOp : public Op {
public:
    static const char* name() { return "Add"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class AddASOp : public Op {
public:
    static const char* name() { return "AddAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class SubtractOp : public Op {
public:
    static const char* name() { return "Subtract"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class SubtractASOp : public Op {
public:
    static const char* name() { return "SubtractAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class MultiplyOp : public Op {
public:
    static const char* name() { return "Multiply"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class MultiplyASOp : public Op {
public:
    static const char* name() { return "MultiplyAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class FloorDivideOp : public Op {
public:
    static const char* name() { return "FloorDivide"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class FloorDivideASOp : public Op {
public:
    static const char* name() { return "FloorDivideAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

class DivideOp : public Op {
public:
    static const char* name() { return "Divide"; }

    virtual void Call(const Array& x1, const Array& x2, const Array& out) = 0;
};

class DivideASOp : public Op {
public:
    static const char* name() { return "DivideAS"; }

    virtual void Call(const Array& x1, Scalar x2, const Array& out) = 0;
};

Array Negative(const Array& x);

namespace internal {

void IAdd(const Array& x1, const Array& x2);
void IAdd(const Array& x1, Scalar x2);

}  // namespace internal

Array Add(const Array& x1, const Array& x2);
Array Add(const Array& x1, Scalar x2);
Array Add(Scalar x1, const Array& x2);

namespace internal {

void ISubtract(const Array& x1, const Array& x2);
void ISubtract(const Array& x1, Scalar x2);

}  // namespace internal

Array Subtract(const Array& x1, const Array& x2);
Array Subtract(const Array& x1, Scalar x2);
Array Subtract(Scalar x1, const Array& x2);

namespace internal {

void IMultiply(const Array& x1, const Array& x2);
void IMultiply(const Array& x1, Scalar x2);

}  // namespace internal

Array Multiply(const Array& x1, const Array& x2);
Array Multiply(const Array& x1, Scalar x2);
Array Multiply(Scalar x1, const Array& x2);

namespace internal {

void IFloorDivide(const Array& x1, const Array& x2);
void IFloorDivide(const Array& x1, Scalar x2);
void ITrueDivide(const Array& x1, const Array& x2);
void ITrueDivide(const Array& x1, Scalar x2);

void IDivide(const Array& x1, const Array& x2);
void IDivide(const Array& x1, Scalar x2);

}  // namespace internal

Array FloorDivide(const Array& x1, const Array& x2);
Array FloorDivide(const Array& x1, Scalar x2);
Array FloorDivide(Scalar x1, const Array& x2);

Array Divide(const Array& x1, const Array& x2);
Array Divide(const Array& x1, Scalar x2);
Array Divide(Scalar x1, const Array& x2);

Array TrueDivide(const Array& x1, const Array& x2);
Array TrueDivide(const Array& x1, Scalar x2);
Array TrueDivide(Scalar x1, const Array& x2);

Array Reciprocal(const Array& x);

Array Sum(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);
// TODO(niboshi): Move to statistics routines
Array AMax(const Array& a, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

Array Maximum(const Array& x1, Scalar x2);
Array Maximum(Scalar x1, const Array& x2);

Array Minimum(const Array& x1, Scalar x2);
Array Minimum(Scalar x1, const Array& x2);
Array Minimum(const Array& x1, const Array& x2);

Array Exp(const Array& x);
Array Log(const Array& x);

// Returns the LogSumExp (LSE) of x, reduced along the specified axes.
// If no axes are specified, all axes will be reduced.
Array LogSumExp(const Array& x, const OptionalAxes& axis = nonstd::nullopt, bool keepdims = false);

// Returns the logarithm of the softmax of x along the specified axes.
// If no axes are specified, the softmax is applied on the second axis.
Array LogSoftmax(const Array& x, const OptionalAxes& axis = nonstd::nullopt);

Array Sigmoid(const Array& x);

Array Square(const Array& x);

Array SquaredDifference(const Array& x1, const Array& x2);

Array Sqrt(const Array& x);

Array IsNan(const Array& x);

Array IsInf(const Array& x);

Array Tanh(const Array& x);

class SinhOp : public Op {
public:
    static const char* name() { return "Sinh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class CoshOp : public Op {
public:
    static const char* name() { return "Cosh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArcsinhOp : public Op {
public:
    static const char* name() { return "Archsinh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArccoshOp : public Op {
public:
    static const char* name() { return "Arccosh"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class SinOp : public Op {
public:
    static const char* name() { return "Sin"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class CosOp : public Op {
public:
    static const char* name() { return "Cos"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class TanOp : public Op {
public:
    static const char* name() { return "Tan"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArcsinOp : public Op {
public:
    static const char* name() { return "Arcsin"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArccosOp : public Op {
public:
    static const char* name() { return "Arccos"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class ArctanOp : public Op {
public:
    static const char* name() { return "Arctan"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class CeilOp : public Op {
public:
    static const char* name() { return "Ceil"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

class FloorOp : public Op {
public:
    static const char* name() { return "Floor"; }

    virtual void Call(const Array& x, const Array& out) = 0;
};

Array Sin(const Array& x);

Array Cos(const Array& x);

Array Tan(const Array& x);

Array Arcsin(const Array& x);

Array Arccos(const Array& x);

Array Arctan(const Array& x);

Array Sinh(const Array& x);

Array Cosh(const Array& x);

Array Arcsinh(const Array& x);

Array Arccosh(const Array& x);

Array Ceil(const Array& x);

Array Floor(const Array& x);

}  // namespace chainerx
