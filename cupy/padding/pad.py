import numpy

import cupy
import six


def _prepend_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    if value == 0:
        return cupy.concatenate((cupy.zeros(padshape, dtype=narray.dtype),
                                 narray), axis=axis)
    else:
        return cupy.concatenate(((cupy.zeros(padshape) + value).astype(
                                 narray.dtype), narray), axis=axis)


def _append_const(narray, pad_amount, value, axis=-1):
    if pad_amount == 0:
        return narray
    padshape = tuple(x if i != axis else pad_amount
                     for i, x in enumerate(narray.shape))
    if value == 0:
        return cupy.concatenate((narray,
                                cupy.zeros(padshape, dtype=narray.dtype)),
                                axis=axis)
    else:
        return cupy.concatenate(
            (narray, (cupy.zeros(padshape) + value).astype(narray.dtype)),
            axis=axis)


def _normalize_shape(ndarray, shape, cast_to_int=True):
    ndims = ndarray.ndim
    if shape is None:
        return ((None, None), ) * ndims
    arr = numpy.asarray(shape)
    if arr.ndim <= 1:
        if arr.shape == () or arr.shape == (1,):
            arr = numpy.ones((ndims, 2), dtype=ndarray.dtype) * arr
        elif arr.shape == (2,):
            arr = arr[numpy.newaxis, :].repeat(ndims, axis=0)
        else:
            fmt = "Unable to create correctly shaped tuple from %s"
            raise ValueError(fmt % (shape,))
    elif arr.ndim == 2:
        if arr.shape[1] == 1 and arr.shape[0] == ndims:
            arr = arr.repeat(2, axis=1)
        elif arr.shape[0] == ndims:
            pass
        else:
            fmt = "Unable to create correctly shaped tuple from %s"
            raise ValueError(fmt % (shape,))
    else:
        fmt = "Unable to create correctly shaped tuple from %s"
        raise ValueError(fmt % (shape,))
    if cast_to_int:
        arr = numpy.rint(arr).astype(int)
    return tuple(tuple(axis) for axis in arr.tolist())


def _validate_lengths(narray, number_elements):
    shape = _normalize_shape(narray, number_elements)
    for axis_shape in shape:
        axis_shape = [1 if x is None else x for x in axis_shape]
        axis_shape = [1 if x >= 0 else -1 for x in axis_shape]
        if axis_shape[0] < 0 or axis_shape[1] < 0:
            message = '%s cannot contain negative values.' % number_elements
            raise ValueError(message)
    return shape


def pad(array, pad_width, mode, **keywords):
    if not cupy.asarray(pad_width).dtype.kind == 'i':
        raise TypeError('pad_width must be of integral type.')
    narray = cupy.array(array)
    pad_width = _validate_lengths(narray, pad_width)
    allowed_keywords = {
        'constant': ['constant_values'],
    }
    keyword_defaults = {
        'constant_values': 0,
    }
    if mode != 'constant':
        raise NotImplementedError
    for key in keywords:
        if key not in allowed_keywords[mode]:
            raise ValueError('%s keyword not in allowed keywords %s' %
                             (key, allowed_keywords[mode]))
    for allowed_keyword in allowed_keywords[mode]:
        keywords.setdefault(allowed_keyword, keyword_defaults[allowed_keyword])
    for key in keywords:
        if key in ['constant_values']:
            keywords[key] = _normalize_shape(narray, keywords[key],
                                             cast_to_int=False)
    newmatrix = narray.copy()
    for axis, ((pad_before, pad_after), (before_value, after_value)) \
            in enumerate(six.moves.zip(pad_width,
                                       keywords['constant_values'])):
        newmatrix = _prepend_const(newmatrix, pad_before, before_value, axis)
        newmatrix = _append_const(newmatrix, pad_after, after_value, axis)
    return newmatrix
