import numpy
import pytest

import xchainer


all_dtypes = (
    'bool_',
    'int8',
    'int16',
    'int32',
    'int64',
    'uint8',
    'float32',
    'float64',
)


float_dtypes = (
    'float32',
    'float64',
)


signed_dtypes = (
    'int8',
    'int16',
    'int32',
    'int64',
    'float32',
    'float64',
)


def parametrize_dtype_specifier(argname, dtypes=None, with_xchainer_dtypes=True, additional_args=None):
    """Parametrizes a test with various arguments that can be used as dtypes.

    Args:
         argname(str): Argument name to pass the value that can be used as a dtype.
         dtypes(list of strs): List of dtype names.
         with_xchainer_dtypes(bool): If ``True``, :class:`xchainer.dtype`\ s are included as the values. ``True`` by default.
         additional_args(tuple of list): Additional values to be included.
    """
    if dtypes is None:
        dtypes = all_dtypes
    assert isinstance(argname, str)
    assert isinstance(dtypes, (tuple, list))
    assert all(isinstance(dt, str) for dt in dtypes)
    lst = []

    # dtype names
    lst += list(dtypes)
    # numpy dtypes
    lst += [numpy.dtype(dt) for dt in dtypes]
    # char codes
    lst += [xchainer.dtype(dt).char for dt in dtypes]
    # xchainer dtypes
    if with_xchainer_dtypes:
        lst += [xchainer.dtype(dt) for dt in dtypes]
    # User-specified args
    if additional_args is not None:
        assert isinstance(additional_args, (tuple, list))
        lst += list(additional_args)

    return pytest.mark.parametrize(argname, lst)
