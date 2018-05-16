import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


_shapes = [
    (),
    (0,),
    (1,),
    (2, 3),
    (1, 1, 1),
    (2, 0, 3),
]


@pytest.fixture(params=_shapes)
def shape(request):
    return request.param


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_negative(xp, device, shape, dtype, is_module):
    if dtype == 'bool_':  # Checked in test_invalid_bool_neg
        return xchainer.testing.ignore()
    x = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.negative(x)
    else:
        return -x


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DtypeError, TypeError))
def test_invalid_bool_negative(xp, device, is_module):
    x = xp.array([True, False], dtype='bool_')
    if is_module:
        xp.negative(x)
    else:
        -x


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_add(xp, device, shape, dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    if is_module:
        return xp.add(lhs, rhs)
    else:
        return lhs + rhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_iadd(xp, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    lhs += rhs
    return lhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_sub(xp, device, shape, numeric_dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=2)
    if is_module:
        return xp.subtract(lhs, rhs)
    else:
        return lhs - rhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_isub(xp, device, shape, numeric_dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype, pattern=2)
    lhs -= rhs
    return lhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_mul(xp, device, shape, dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    if is_module:
        return xp.multiply(lhs, rhs)
    else:
        return lhs * rhs


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_imul(xp, device, shape, dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=1)
    rhs = array_utils.create_dummy_ndarray(xp, shape, dtype, pattern=2)
    lhs *= rhs
    return lhs


@pytest.mark.parametrize('scalar', [0, -1, 1, 2])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_mul_scalar(scalar, device, shape, dtype):
    x_np = array_utils.create_dummy_ndarray(numpy, shape, dtype)
    # Implicit casting in NumPy's multiply depends on the 'casting' argument,
    # which is not yet supported (xChainer always casts).
    # Therefore, we explicitly cast the scalar to the dtype of the ndarray
    # before the multiplication for NumPy.
    expected = x_np * numpy.dtype(dtype).type(scalar)

    x = xchainer.array(x_np)
    scalar_xc = xchainer.Scalar(scalar, dtype)
    xchainer.testing.assert_array_equal_ex(x * scalar, expected)
    xchainer.testing.assert_array_equal_ex(x * scalar_xc, expected)
    xchainer.testing.assert_array_equal_ex(scalar * x, expected)
    xchainer.testing.assert_array_equal_ex(scalar_xc * x, expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(x, scalar), expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(x, scalar_xc), expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(scalar, x), expected)
    xchainer.testing.assert_array_equal_ex(xchainer.multiply(scalar_xc, x), expected)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_truediv(xp, device, shape, numeric_dtype, is_module):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype)
    rhs = xp.arange(1, lhs.size + 1, dtype=numeric_dtype).reshape(shape)
    # TODO(beam2d): Remove astype after supporting correct dtype promotion.
    if is_module:
        return xp.divide(lhs, rhs).astype(numeric_dtype)
    else:
        return (lhs / rhs).astype(numeric_dtype)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@xchainer.testing.numpy_xchainer_array_equal()
def test_itruediv(xp, device, shape, numeric_dtype):
    lhs = array_utils.create_dummy_ndarray(xp, shape, numeric_dtype)
    rhs = xp.arange(1, lhs.size + 1, dtype=numeric_dtype).reshape(shape)
    # TODO(beam2d): Fix after supporting correct dtype promotion.
    if xp is numpy and 'int' in numeric_dtype:
        # NumPy does not support itruediv to integer arrays.
        lhs = (lhs / rhs).astype(numeric_dtype)
    else:
        lhs /= rhs
    return lhs


# TODO(sonots): Fix type compatibility
# TODO(niboshi): Remove strides_check=False
@xchainer.testing.numpy_xchainer_array_equal(dtype_check=False, strides_check=False)
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('large_dtype', ['int32', 'int64', 'float32', 'float64'])
@pytest.mark.parametrize('shape,axis', [
    ((), None),
    ((), ()),
    ((2,), None),
    ((2,), ()),
    ((2,), 0),
    ((2,), (0,)),
    ((2,), (-1,)),
    ((2, 3), None),
    ((2, 3), ()),
    ((2, 3), 0),
    ((2, 3), (0,)),
    ((2, 3), (1,)),
    ((2, 3), (-1,)),
    ((2, 3), (-2,)),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, -1)),
    ((1, 3), None),  # sum over 1-dim axis
    ((0, 3), None),  # sum over 0-dim axis
    # Sum over axes that are in the middle or apart
    ((2, 3, 4), (1,)),
    ((2, 3, 4), (0, 2)),
    # Sum over axes that are apart and/or unsorted
    ((2, 3), (1, 0)),
    ((2, 3, 4), (2, 0)),
    ((2, 3, 4), (2, 0, 1)),
    ((2, 3, 4), (-2, 2, 0)),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_sum(is_module, xp, device, shape, axis, keepdims, large_dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, large_dtype)
    if is_module:
        return xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        return a.sum(axis=axis, keepdims=keepdims)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, ValueError))
@pytest.mark.parametrize('keepdims', [False, True])
@pytest.mark.parametrize('shape,axis', [
    # ((), 0), # TODO(sonots): Fix compatibility
    ((), 1),
    ((), (1,)),
    ((2,), 2),
    ((2,), (2,)),
    ((2,), (-2,)),
    ((2, 3,), (-3,)),
    ((2, 3,), (-3, -4)),
    ((2, 3,), (0, 0)),
    ((2, 3,), (-1, -1)),
    ((2, 3,), (0, 1, 1)),
    ((2, 3,), (0, -2)),
])
def test_sum_invalid(is_module, xp, shape, axis, keepdims, dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        xp.sum(a, axis=axis, keepdims=keepdims)
    else:
        a.sum(axis=axis, keepdims=keepdims)


# TODO(sonots): Fix type compatibility for when shape is ()
@xchainer.testing.numpy_xchainer_array_equal(dtype_check=False)
@pytest.mark.parametrize("shape,value", [
    ((), -1),
    ((), 1),
    ((1,), -1),
    ((1,), 1),
    ((2,), 1),
    ((2, 3), 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_maximum_with_scalar(xp, device, shape, value, signed_dtype):
    a = array_utils.create_dummy_ndarray(xp, shape, signed_dtype)
    return xp.maximum(a, value)


def _create_dummy_array_for_dot(xp, shape, dtype):
    x = numpy.arange(numpy.prod(shape)).reshape(shape)
    if dtype == 'bool_':
        x = numpy.asarray(x % 2 == 0)
    else:
        x = x.astype(dtype)
    return xp.array(x)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-4), numpy.asarray(4),
    numpy.asarray(-float('inf')), numpy.asarray(float('inf')), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
# TODO(niboshi): Dtype promotion is not supported yet.
@xchainer.testing.numpy_xchainer_array_equal()
def test_exp(xp, device, input, float_dtype):
    dtype = float_dtype
    a = xp.array(input.astype(dtype))
    return xp.exp(a)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('input', [
    numpy.asarray(0), numpy.asarray(-1), numpy.asarray(1), numpy.asarray(10), numpy.asarray(float('inf')), numpy.asarray(float('nan')),
    numpy.full((), 2), numpy.full((0,), 2), numpy.full((2, 3), 2)
])
# TODO(niboshi): Dtype promotion is not supported yet.
@xchainer.testing.numpy_xchainer_array_equal()
def test_log(xp, device, input, float_dtype):
    dtype = float_dtype
    a = xp.array(input.astype(dtype))
    return xp.log(a)


_logsumexp_params = [
    ((2,), 0),
    ((2,), -1),
    ((2, 3), None),
    ((2, 3), 0),
    ((2, 3), 1),
    ((2, 3), -2),
    ((2, 3), (0, 1)),
    ((2, 3), (-2, 1)),
    ((1, 2, 3), None),
    ((1, 2, 3), (1)),
    ((1, 2, 3), (1, 0)),
    ((1, 2, 3), (0, 1, 2)),
]


_invalid_logsumexp_params = [
    # Axis out of bounds
    ((2,), 1),
    ((2,), -2),
    ((2,), (0, 1)),
    ((2, 3), (0, 1, 2)),
    # Duplicate axes
    ((2,), (0, 0)),
    ((2, 3), (0, 0)),
]


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-7, atol=0, dtype_check=False)
# TODO(hvy): Dtype promotion is not supported yet.
def test_logsumexp(xp, device, a_shape, axis, float_dtype, keepdims):
    a = array_utils.create_dummy_ndarray(xp, a_shape, float_dtype)
    if xp is numpy:
        return xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=keepdims))
    return xp.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
@pytest.mark.parametrize('keepdims', [True, False])
# TODO(hvy): Dtype promotion is not supported yet.
# TODO(hvy): Should not overflow for large numbers, add tests
def test_logsumexp_invalid(device, a_shape, axis, float_dtype, keepdims):
    a = array_utils.create_dummy_ndarray(xchainer, a_shape, float_dtype)
    with pytest.raises(xchainer.DimensionError):
        xchainer.logsumexp(a, axis=axis, keepdims=keepdims)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _logsumexp_params)
@xchainer.testing.numpy_xchainer_allclose(rtol=1e-7, atol=1e-5, dtype_check=False)
# TODO(hvy): Dtype promotion is not supported yet.
def test_log_softmax(xp, device, a_shape, axis, float_dtype):
    a = array_utils.create_dummy_ndarray(xp, a_shape, float_dtype)
    if xp is numpy:
        # Default is the second axis
        axis = axis if axis is not None else 1
        return a - xp.log(xp.sum(xp.exp(a), axis=axis, keepdims=True))
    return xp.log_softmax(a, axis=axis)


@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
@pytest.mark.parametrize('a_shape,axis', _invalid_logsumexp_params)
# TODO(hvy): Dtype promotion is not supported yet.
def test_log_softmax_invalid(device, a_shape, axis, float_dtype):
    a = array_utils.create_dummy_ndarray(xchainer, a_shape, float_dtype)
    with pytest.raises(xchainer.DimensionError):
        return xchainer.log_softmax(a, axis=axis)
