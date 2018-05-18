import math

import numpy
import pytest

import xchainer
import xchainer.testing

from tests import array_utils


@pytest.mark.parametrize('value', [
    0, 1, -1, 0.1, 0.9, -0.1, -0.9, 1.1, -1.1, 1.9, -1.9, True, False, float('inf'), -float('inf'), float('nan'), -0.0
])
@pytest.mark.parametrize('shape', [
    (), (1,), (1, 1, 1)
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_asscalar(device, value, shape, dtype):
    np_dtype = numpy.dtype(dtype)
    try:
        np_value = np_dtype.type(value)
    except (ValueError, OverflowError):
        return

    a_np = numpy.asarray([np_value], dtype).reshape(shape)
    a_xc = xchainer.array(a_np)

    def should_cast_succeed(typ):
        try:
            typ(np_value)
            return True
        except (ValueError, OverflowError):
            return False

    # Cast to float
    if should_cast_succeed(float):
        assert type(float(a_xc)) is float
        if math.isnan(float(a_np)):
            assert math.isnan(float(a_xc))
        else:
            assert float(a_np) == float(a_xc)
    # Cast to int
    if should_cast_succeed(int):
        assert type(int(a_xc)) is int
        assert int(a_np) == int(a_xc)
    # Cast to bool
    if should_cast_succeed(bool):
        assert type(bool(a_xc)) is bool
        assert bool(a_np) == bool(a_xc)

    # xchainer.asscalar
    assert isinstance(xchainer.asscalar(a_xc), type(numpy.asscalar(a_np)))
    if math.isnan(numpy.asscalar(a_np)):
        assert math.isnan(xchainer.asscalar(a_xc))
    else:
        assert xchainer.asscalar(a_xc) == numpy.asscalar(a_np)


@pytest.mark.parametrize('shape', [
    (0,), (1, 0), (2,), (1, 2), (2, 3),
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_asscalar_invalid(device, shape):
    dtype = xchainer.float32

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        xchainer.asscalar(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        float(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        int(a)

    a = xchainer.ones(shape, dtype)
    with pytest.raises(xchainer.DimensionError):
        bool(a)


@xchainer.testing.numpy_xchainer_array_equal()
def test_transpose(is_module, xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.transpose(array)
    else:
        return array.transpose()


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('shape,axes', [
    ((1,), 0),
    ((1,), (0,)),
    ((2,), (0,)),
    ((2, 3), (1, 0)),
    ((2, 3, 1), (2, 0, 1)),
])
def test_transpose_axes(is_module, xp, shape, axes, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    if is_module:
        return xp.transpose(array, axes)
    else:
        return array.transpose(axes)


@pytest.mark.parametrize('shape,axes', [
    ((), (0,)),
    ((1,), (1,)),
    ((2, 3), (1,)),
    ((2, 3), (1, 0, 2)),
])
def test_transpose_invalid_axes(shape, axes):
    a = array_utils.create_dummy_ndarray(xchainer, shape, 'float32')
    with pytest.raises(xchainer.DimensionError):
        xchainer.transpose(a, axes)
    with pytest.raises(xchainer.DimensionError):
        a.transpose(axes)


@xchainer.testing.numpy_xchainer_array_equal()
def test_T(xp, shape, dtype):
    array = array_utils.create_dummy_ndarray(xp, shape, dtype)
    return array.T


_reshape_shape = [
    ((), ()),
    ((0,), (0,)),
    ((1,), (1,)),
    ((5,), (5,)),
    ((2, 3), (2, 3)),
    ((1,), ()),
    ((), (1,)),
    ((1, 1), ()),
    ((), (1, 1)),
    ((6,), (2, 3)),
    ((2, 3), (6,)),
    ((2, 0, 3), (5, 0, 7)),
    ((5,), (1, 1, 5, 1, 1)),
    ((1, 1, 5, 1, 1), (5,)),
    ((2, 3), (3, 2)),
    ((2, 3, 4), (3, 4, 2)),
]


# TODO(niboshi): Test with non-contiguous input array that requires copy to reshape
# TODO(niboshi): Test with non-contiguous input array that does not require copy to reshape
@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('a_shape,b_shape', _reshape_shape)
@pytest.mark.parametrize('shape_type', [tuple, list])
def test_reshape(is_module, xp, a_shape, b_shape, shape_type):
    a = array_utils.create_dummy_ndarray(xp, a_shape, 'int64')
    if is_module:
        b = xp.reshape(a, shape_type(b_shape))
    else:
        b = a.reshape(shape_type(b_shape))

    if xp is xchainer:
        assert b.is_contiguous
        assert a._debug_data_memory_address == b._debug_data_memory_address, 'Reshape must be done without copy'
        assert numpy.arange(a.size).reshape(b_shape).strides == b.strides, 'Strides after reshape must match NumPy behavior'

    return b


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(TypeError, xchainer.XchainerError))
@pytest.mark.parametrize('a_shape,b_shape', _reshape_shape)
def test_reshape_args(is_module, xp, a_shape, b_shape):
    a = array_utils.create_dummy_ndarray(xp, a_shape, 'int64')
    if is_module:
        if len(b_shape) > 1:
            # Skipping tests where the 'order' argument is unintentionally given a shape value, since numpy won't raise any errors in
            # this case which you might expect at first.
            return xp.array([])
        b = xp.reshape(a, *b_shape)  # TypeError/xchainer.XchainerError in case b_shape is empty.
    else:
        b = a.reshape(*b_shape)  # TypeError/xchainer.XchainerError in case b_shape is empty.

    if xp is xchainer:
        assert b.is_contiguous
        assert a._debug_data_memory_address == b._debug_data_memory_address, 'Reshape must be done without copy'
        assert numpy.arange(a.size).reshape(b_shape).strides == b.strides, 'Strides after reshape must match NumPy behavior'

    return b


@pytest.mark.parametrize('shape1,shape2', [
    ((), (0,)),
    ((), (2,)),
    ((), (1, 2,)),
    ((0,), (1,)),
    ((0,), (1, 1, 1)),
    ((2, 3), (2, 3, 2)),
    ((2, 3, 4), (2, 3, 5)),
])
def test_reshape_invalid(shape1, shape2):
    def check(a_shape, b_shape):
        a = array_utils.create_dummy_ndarray(xchainer, a_shape, 'float32')
        with pytest.raises(xchainer.DimensionError):
            a.reshape(b_shape)

    check(shape1, shape2)
    check(shape2, shape1)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('shape,axis', [
    ((), None),
    ((0,), None),
    ((1,), None),
    ((1, 1), None),
    ((1, 0, 1), None),
    ((3,), None),
    ((3, 1), None),
    ((1, 3), None),
    ((2, 0, 3), None),
    ((2, 4, 3), None),
    ((2, 1, 3), 1),
    ((2, 1, 3), -2),
    ((1, 2, 1, 3, 1, 1, 4), None),
    ((1, 2, 1, 3, 1, 1, 4), (2, 0, 4)),
    ((1, 2, 1, 3, 1, 1, 4), (-2, 0, 4)),
])
def test_squeeze(is_module, xp, shape, axis):
    a = array_utils.create_dummy_ndarray(xp, shape, 'float32')
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, ValueError))
@pytest.mark.parametrize('shape,axis', [
    ((2, 1, 3), 0),
    ((2, 1, 3), -1),
    ((2, 1, 3), (1, 2)),
    ((2, 1, 3), (1, -1)),
    ((2, 1, 3), (1, 1)),
])
def test_squeeze_invalid(is_module, xp, shape, axis):
    a = xp.ones(shape, 'float32')
    if is_module:
        return xp.squeeze(a, axis)
    else:
        return a.squeeze(axis)


@xchainer.testing.numpy_xchainer_array_equal()
@pytest.mark.parametrize('src_shape,dst_shape', [
    ((), ()),
    ((1,), (2,)),
    ((1, 1), (2, 2)),
    ((1, 1), (1, 2)),
])
def test_broadcast_to(xp, src_shape, dst_shape):
    a = array_utils.create_dummy_ndarray(xp, src_shape, 'float32')
    return xp.broadcast_to(a, dst_shape)


@xchainer.testing.numpy_xchainer_array_equal()
def test_broadcast_to_auto_prefix(xp):
    a = xp.arange(2, dtype='float32')
    return xp.broadcast_to(a, (3, 2))


@xchainer.testing.numpy_xchainer_array_equal(accept_error=(xchainer.DimensionError, ValueError))
@pytest.mark.parametrize(('src_shape,dst_shape'), [
    ((3,), (2,)),
    ((3,), (3, 2)),
    ((1, 3), (3, 2)),
])
def test_broadcast_to_invalid(xp, src_shape, dst_shape):
    a = xp.ones(src_shape, 'float32')
    return xp.broadcast_to(a, dst_shape)
