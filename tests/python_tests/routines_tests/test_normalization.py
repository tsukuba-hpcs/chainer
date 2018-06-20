import chainer
import numpy
import pytest

import xchainer

from tests import array_utils


def _create_batch_norm_ndarray_args(xp, device, x_shape, gamma_shape, beta_shape, running_mean_shape, running_var_shape, float_dtype):
    x = array_utils.create_dummy_ndarray(xp, x_shape, float_dtype)

    # Non-contiguous gamma and beta is not supported by CUDA.
    # TODO(hvy): Support non-contiguous gamma and beta with CUDA. Create a contiguous copy in the cuDNN wrapper.
    pad_gamma_beta = device.backend.name != 'cuda'
    gamma = array_utils.create_dummy_ndarray(xp, gamma_shape, float_dtype, padding=pad_gamma_beta)
    beta = array_utils.create_dummy_ndarray(xp, beta_shape, float_dtype, padding=pad_gamma_beta)

    # Non-contiguous running values which are updated in-place are not supported by CUDA, so we only pad for other devices.
    pad_running = device.backend.name != 'cuda'
    running_mean = array_utils.create_dummy_ndarray(xp, running_mean_shape, float_dtype, padding=pad_running)
    running_var = array_utils.create_dummy_ndarray(xp, running_var_shape, float_dtype, padding=pad_running, start=0)

    return x, gamma, beta, running_mean, running_var


# Note that CUDA (cuDNN) only supports batch normalization with 4 or 5-dimenisional data.
@pytest.mark.parametrize('x_shape,reduced_shape,axis', [
    ((2, 3, 4, 5), (3, 4, 5), None),
    ((2, 3, 4, 5), (3, 4, 5), (0,)),
    ((2, 3, 4, 5), (3,), (0, 2, 3)),
    ((2, 3, 4, 5, 2), (3, 4, 5, 2), None),
    ((2, 3, 4, 5, 2), (3, 4, 5, 2), (0,)),
    ((2, 3, 4, 5, 2), (3,), (0, 2, 3, 4))
])
@pytest.mark.parametrize('eps', [None, 3e-5, 1.2])
@pytest.mark.parametrize('decay', [None, 0.5])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_batch_norm(device, x_shape, reduced_shape, eps, decay, axis, float_dtype):
    def create_args(xp):
        return _create_batch_norm_ndarray_args(
            xp, device, x_shape, reduced_shape, reduced_shape, reduced_shape, reduced_shape, float_dtype)

    x_xc, gamma_xc, beta_xc, running_mean_xc, running_var_xc = create_args(xchainer)
    x_np, gamma_np, beta_np, running_mean_np, running_var_np = create_args(numpy)

    # Save copies of running values before updating to later check that they are updated.
    initial_running_mean = running_mean_xc.copy()
    initial_running_var = running_var_xc.copy()

    optional_args = {}
    if eps is not None:
        optional_args['eps'] = eps
    if decay is not None:
        optional_args['decay'] = decay
    if axis is not None:
        optional_args['axis'] = axis

    y_xc = xchainer.batch_norm(x_xc, gamma_xc, beta_xc, running_mean=running_mean_xc, running_var=running_var_xc, **optional_args)
    y_np = chainer.functions.batch_normalization(
        x_np, gamma_np, beta_np, running_mean=running_mean_np, running_var=running_var_np, **optional_args).data

    # Check that the running values are updated.
    assert not numpy.allclose(xchainer.tonumpy(initial_running_mean), xchainer.tonumpy(running_mean_xc))
    assert not numpy.allclose(xchainer.tonumpy(initial_running_var), xchainer.tonumpy(running_var_xc))

    xchainer.testing.assert_allclose_ex(y_xc, y_np, rtol=1e-6, atol=1e-5)
    xchainer.testing.assert_allclose_ex(running_mean_xc, running_mean_np, rtol=1e-6, atol=1e-6)
    xchainer.testing.assert_allclose_ex(running_var_xc, running_var_np, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('x_shape,gamma_shape,beta_shape,running_mean_shape,running_var_shape,axis', [
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), None),  # Bad reduction, axis defaults to (0,) but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), ()),  # Bad reduction, axis is () but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), (2, 3)),  # Bad reduction, axis is (2, 3) but should be (0, 2, 3).
    ((2, 3, 4, 5), (3, 4), (3,), (3,), (3,), (0, 2, 3)),  # Bad gamma shape.
    ((2, 3, 4, 5), (3,), (3, 4), (3,), (3,), (0, 2, 3)),  # Bad beta shape.
    ((2, 3, 4, 5), (3,), (3,), (3, 4), (3,), (0, 2, 3)),  # Bad running_mean shape.
    ((2, 3, 4, 5), (3,), (3,), (3,), (3, 4), (0, 2, 3)),  # Bad running_var shape.
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_batch_norm_invalid_dimensions(device, x_shape, gamma_shape, beta_shape, running_mean_shape, running_var_shape, axis, float_dtype):
    x, gamma, beta, running_mean, running_var = _create_batch_norm_ndarray_args(
        xchainer, device, x_shape, gamma_shape, beta_shape, running_mean_shape, running_var_shape, float_dtype)

    with pytest.raises(xchainer.DimensionError):
        xchainer.batch_norm(x, gamma, beta, running_mean=running_mean, running_var=running_var, eps=1e-2, decay=0.9, axis=axis)


# Note that CUDA (cuDNN) only supports batch normalization with 4 or 5-dimenisional data.
@pytest.mark.parametrize('x_shape,reduced_shape,axis', [
    ((2, 3, 4, 5), (3, 4, 5), None),
    ((2, 3, 4, 5), (3, 4, 5), (0,)),
    ((2, 3, 4, 5), (3,), (0, 2, 3)),
    ((2, 3, 4, 5, 2), (3, 4, 5, 2), None),
    ((2, 3, 4, 5, 2), (3, 4, 5, 2), (0,)),
    ((2, 3, 4, 5, 2), (3,), (0, 2, 3, 4))
])
@pytest.mark.parametrize('eps', [None]) # , 3e-5, 1.2])
@pytest.mark.parametrize_device(['native:0']) # , 'cuda:0'])
def test_fixed_batch_norm(device, x_shape, reduced_shape, eps, axis, float_dtype):
    def create_args(xp):
        return _create_batch_norm_ndarray_args(
            xp, device, x_shape, reduced_shape, reduced_shape, reduced_shape, reduced_shape, float_dtype)

    x_xc, gamma_xc, beta_xc, mean_xc, var_xc = create_args(xchainer)
    x_np, gamma_np, beta_np, mean_np, var_np = create_args(numpy)

    optional_args = {}
    if eps is not None:
        optional_args['eps'] = eps
    if axis is not None:
        optional_args['axis'] = axis

    y_xc = xchainer.fixed_batch_norm(x_xc, gamma_xc, beta_xc, mean=mean_xc, var=var_xc, **optional_args)
    y_np = chainer.functions.fixed_batch_normalization(
        x_np, gamma_np, beta_np, mean=mean_np, var=var_np, **optional_args).data

    xchainer.testing.assert_allclose_ex(y_xc, y_np, rtol=1e-6, atol=1e-5)


@pytest.mark.parametrize('x_shape,gamma_shape,beta_shape,mean_shape,var_shape,axis', [
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), None),  # Bad reduction, axis defaults to (0,) but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), ()),  # Bad reduction, axis is () but should be (0, 2, 3).
    ((2, 3, 4, 5), (3,), (3,), (3,), (3,), (2, 3)),  # Bad reduction, axis is (2, 3) but should be (0, 2, 3).
    ((2, 3, 4, 5), (3, 4), (3,), (3,), (3,), (0, 2, 3)),  # Bad gamma shape.
    ((2, 3, 4, 5), (3,), (3, 4), (3,), (3,), (0, 2, 3)),  # Bad beta shape.
    ((2, 3, 4, 5), (3,), (3,), (3, 4), (3,), (0, 2, 3)),  # Bad mean shape.
    ((2, 3, 4, 5), (3,), (3,), (3,), (3, 4), (0, 2, 3)),  # Bad var shape.
])
@pytest.mark.parametrize_device(['native:0', 'cuda:0'])
def test_fixed_batch_norm_invalid_dimensions(device, x_shape, gamma_shape, beta_shape, mean_shape, var_shape, axis, float_dtype):
    x, gamma, beta, mean, var = _create_batch_norm_ndarray_args(
        xchainer, device, x_shape, gamma_shape, beta_shape, mean_shape, var_shape, float_dtype)

    with pytest.raises(xchainer.DimensionError):
        xchainer.fixed_batch_norm(x, gamma, beta, mean=mean, var=var, eps=1e-2, axis=axis)
