import numpy.testing

import xchainer

# NumPy-like assertion functions that accept both NumPy and xChainer arrays


def _check_xchainer_array(x):
    # Checks basic conditions that are assumed to hold true for any given xChainer array passed to assert_array_close and
    # assert_array_equal.
    assert isinstance(x, xchainer.ndarray)
    assert not x.is_grad_required()


def _as_numpy(x):
    if isinstance(x, xchainer.ndarray):
        return xchainer.tonumpy(x)
    assert isinstance(x, numpy.ndarray) or numpy.isscalar(x)
    return x


def _check_dtype_and_strides(x, y, dtype_check, strides_check):
    if strides_check is not None and dtype_check is not None and strides_check and not dtype_check:
        raise ValueError()
    if dtype_check is None:
        dtype_check = True
    if strides_check is None:
        strides_check = dtype_check

    if isinstance(x, (numpy.ndarray, xchainer.ndarray)) and isinstance(y, (numpy.ndarray, xchainer.ndarray)):
        if strides_check:
            assert x.strides == y.strides, f'Strides mismatch: x: {x.strides}, y: {y.strides}'
        if dtype_check:
            assert x.dtype.name == y.dtype.name, f'Dtype mismatch: x: {x.dtype}, y: {y.dtype}'


def _preprocess_input(a):
    # Convert xchainer.Scalar to Python scalar
    if isinstance(a, xchainer.Scalar):
        a = a.tolist()

    # Check conditions for xchainer.ndarray
    if isinstance(a, xchainer.ndarray):
        _check_xchainer_array(a)

    # Convert to something NumPy can handle
    a = _as_numpy(a)
    return a


def assert_allclose(x, y, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal up to a tolerance.

    Args:
         x(numpy.ndarray or xchainer.ndarray): The actual object to check.
         y(numpy.ndarray or xchainer.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    x = _preprocess_input(x)
    y = _preprocess_input(y)

    numpy.testing.assert_allclose(x, y, rtol=rtol, atol=atol, equal_nan=equal_nan, err_msg=err_msg, verbose=verbose)


def assert_array_equal(x, y, err_msg='', verbose=True):
    """Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or xchainer.ndarray): The actual object to check.
         y(numpy.ndarray or xchainer.ndarray): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
    .. seealso:: :func:`numpy.testing.assert_array_equal`
    """
    x = _preprocess_input(x)
    y = _preprocess_input(y)

    numpy.testing.assert_array_equal(x, y, err_msg=err_msg, verbose=verbose)


def assert_allclose_ex(x, y, *args, dtype_check=None, strides_check=None, **kwargs):
    """assert_allclose_ex(x, y, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True, *, dtype_check=True, strides_check=True):

    Raises an AssertionError if two array_like objects are not equal up to a tolerance.

    Args:
         x(numpy.ndarray or xchainer.ndarray): The actual object to check.
         y(numpy.ndarray or xchainer.ndarray): The desired, expected object.
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         dtype_check(bool): If ``True``, consistency of dtype is also checked. Disabling ``dtype_check`` also implies ``strides_check=False``.
         strides_check(bool): If ``True``, consistency of strides is also checked.
    .. seealso:: :func:`numpy.testing.assert_allclose`
    """
    assert_allclose(x, y, *args, **kwargs)
    _check_dtype_and_strides(x, y, dtype_check, strides_check)


def assert_array_equal_ex(x, y, *args, dtype_check=None, strides_check=None, **kwargs):
    """assert_array_equal_ex(x, y, err_msg='', verbose=True, *, dtype_check=True, strides_check=True)

    Raises an AssertionError if two array_like objects are not equal.

    Args:
         x(numpy.ndarray or xchainer.ndarray): The actual object to check.
         y(numpy.ndarray or xchainer.ndarray): The desired, expected object.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values
             are appended to the error message.
         dtype_check(bool): If ``True``, consistency of dtype is also checked. Disabling ``dtype_check`` also implies ``strides_check=False``.
         strides_check(bool): If ``True``, consistency of strides is also checked.
    .. seealso::
       :func:`numpy.testing.assert_array_equal`
    """
    assert_array_equal(x, y, *args, **kwargs)
    _check_dtype_and_strides(x, y, dtype_check, strides_check)
