import functools
import traceback
import warnings

import numpy
import pytest

import xchainer
from xchainer.testing import array


# A test returning an instance of this class will have its return value ignored.
#
# This is e.g. useful when a combination of parametrizations and operations unintentionally cover non-supported function calls.
# For instance, you might parametrize over shapes (tuples) which are unpacked and passed to a function.
# While you might want to test empty tuples for module functions, they should maybe be ignored for ndarray functions.
#
# If either xchainer or numpy returns an instance of this class, the other module should too.
# Otherwise, the test will be considered inconsistent and be treated as a failure.
class NumPyXchainerIgnoredResult(object):
    pass


# A wrapper for instantiating the ignore object.
def ignore():
    return NumPyXchainerIgnoredResult()


def _call_func(impl, args, kw):
    try:
        result = impl(*args, **kw)
        assert isinstance(result, NumPyXchainerIgnoredResult) or result is not None
        error = None
        tb = None
    except Exception as e:
        result = None
        error = e
        tb = traceback.format_exc()

    return result, error, tb


def _check_xchainer_numpy_error(xchainer_error, xchainer_tb, numpy_error,
                                numpy_tb, accept_error=()):
    # TODO(sonots): Change error class names of xChainer to be similar with NumPy, and check names.
    if xchainer_error is None and numpy_error is None:
        pytest.fail('Both xchainer and numpy are expected to raise errors, but not')
    elif xchainer_error is None:
        pytest.fail('Only numpy raises error\n\n' + numpy_tb)
    elif numpy_error is None:
        pytest.fail('Only xchainer raises error\n\n' + xchainer_tb)
    elif not (isinstance(xchainer_error, accept_error) and
              isinstance(numpy_error, accept_error)):
        msg = '''Both xchainer and numpy raise exceptions

xchainer
%s
numpy
%s
''' % (xchainer_tb, numpy_tb)
        pytest.fail(msg)


def _check_xchainer_numpy_result(check_result_func, xchainer_result, numpy_result, type_check):
    xchainer_ignored = isinstance(xchainer_result, NumPyXchainerIgnoredResult)
    numpy_ignored = isinstance(numpy_result, NumPyXchainerIgnoredResult)
    if xchainer_ignored and numpy_ignored:
        return  # Ignore without failing.
    elif numpy_ignored:
        pytest.fail("Only numpy result ignored.")
    elif xchainer_ignored:
        pytest.fail("Only xchainer result ignored.")

    assert isinstance(xchainer_result, xchainer.ndarray), type(xchainer_result)
    assert isinstance(numpy_result, numpy.ndarray) or numpy.isscalar(numpy_result), type(numpy_result)
    assert xchainer_result.shape == numpy_result.shape
    assert xchainer_result.device is xchainer.get_default_device()
    if type_check:
        assert numpy.dtype(xchainer_result.dtype.name) == numpy_result.dtype

    check_result_func(xchainer_result, numpy_result)


def _make_decorator(check_result_func, name, type_check, accept_error):
    def decorator(impl):
        @functools.wraps(impl)
        def test_func(*args, **kw):
            kw[name] = xchainer
            xchainer_result, xchainer_error, xchainer_tb = _call_func(impl, args, kw)

            kw[name] = numpy
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                numpy_result, numpy_error, numpy_tb = _call_func(impl, args, kw)

            if xchainer_error or numpy_error:
                _check_xchainer_numpy_error(xchainer_error, xchainer_tb,
                                            numpy_error, numpy_tb,
                                            accept_error=accept_error)
                return

            if xchainer_result is not None or numpy_result is not None:
                _check_xchainer_numpy_result(check_result_func, xchainer_result, numpy_result, type_check)
                return

            raise AssertionError('Using decorator without returning ndarrays.', xchainer_result, numpy_result)
        # Apply dummy parametrization on `name` (e.g. 'xp') to avoid pytest error when collecting test functions.
        return pytest.mark.parametrize(name, [None])(test_func)
    return decorator


def numpy_xchainer_allclose(*, rtol=1e-7, atol=0, equal_nan=True, err_msg='', verbose=True, name='xp', type_check=True, accept_error=()):
    """Decorator that checks that NumPy and xChainer results are equal up to a tolerance.

    Args:
         rtol(float): Relative tolerance.
         atol(float): Absolute tolerance.
         equal_nan(bool): Allow NaN values if True. Otherwise, fail the assertion if any NaN is found.
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or ``xchainer`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and xChainer test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_xchainer_allclose`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``xchainer``.

    .. seealso:: :func:`xchainer.testing.assert_allclose`
    """
    def check_result_func(x, y):
        array.assert_allclose(x, y, rtol, atol, equal_nan, err_msg, verbose)

    return _make_decorator(check_result_func, name, type_check, accept_error)


def numpy_xchainer_array_equal(*, err_msg='', verbose=True, name='xp', type_check=True, accept_error=()):
    """Decorator that checks that NumPy and xChainer results are equal.

    Args:
         err_msg(str): The error message to be printed in case of failure.
         verbose(bool): If ``True``, the conflicting values are
             appended to the error message.
         name(str): Argument name whose value is either ``numpy`` or ``xchainer`` module.
         type_check(bool): If ``True``, consistency of dtype is also checked.
         accept_error(Exception or tuple of Exception): Specify
             acceptable errors. When both NumPy test and xChainer test raises the
             same type of errors, and the type of the errors is specified with
             this argument, the errors are ignored and not raised.

    Decorated test fixture is required to return the same arrays
    in the sense of :func:`numpy_xchainer_array_equal`
    (except the type of array module) even if ``xp`` is ``numpy`` or ``xchainer``.

    .. seealso:: :func:`xchainer.testing.assert_array_equal`
    """
    def check_result_func(x, y):
        array.assert_array_equal(x, y, err_msg, verbose)

    return _make_decorator(check_result_func, name, type_check, accept_error)
