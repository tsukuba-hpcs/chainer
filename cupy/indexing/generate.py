# flake8: NOQA
# "flake8: NOQA" to suppress warning "H104  File contains nothing but comments"
import cupy

import six


class AxisConcatenator(object):
    """Translates slice objects to concatenation along an axis.

    For detailed documentation on usage, see `r_`.

    """
    def _retval(self, res):
        if self.matrix:
            oldndim = res.ndim
            res = makemat(res)
            if oldndim == 1 and self.col:
                res = res.T
        self.axis = self._axis
        self.matrix = self._matrix
        self.col = 0
        return res

    def _output_obj(self, newobj, tempobj, ndmin, trans1d):
        k2 = ndmin - tempobj.ndim
        if (trans1d < 0):
            trans1d += k2 + 1
        defaxes = list(six.moves.range(ndmin))
        k1 = trans1d
        axes = defaxes[:k1] + defaxes[k2:] + \
               defaxes[k1:k2]
        newobj = newobj.transpose(axes)
        return newobj

    def __init__(self, axis=0, matrix=False, ndmin=1, trans1d=-1):
        self._axis = axis
        self._matrix = matrix
        self.axis = axis
        self.matrix = matrix
        self.col = 0
        self.trans1d = trans1d
        self.ndmin = ndmin

    def __getitem__(self, key):
        trans1d = self.trans1d
        ndmin = self.ndmin
        objs = []
        if isinstance(key, str):
            return NotImplemented
        if not isinstance(key, tuple):
            key = (key,)

        for k in six.moves.range(len(key)):
            if isinstance(key[k], slice):
                return NotImplemented
            elif isinstance(key[k], str):
                if k != 0:
                    raise ValueError(
                    'special directives must be the first entry.')
                return NotImplemented
            else:
                newobj = key[k]
                if ndmin > 1:
                    tempobj = cupy.array(newobj, copy=False)
                    newobj = cupy.array(newobj, copy=False, ndmin=ndmin)
                    if trans1d != -1 and tempobj.ndim < ndmin:
                        newobj = self._output_obj(newobj, ndmin, trans1d)
                    del tempobj
                elif ndmin == 1:
                    tempobj = cupy.array(newobj, copy=False)
                    newobj = cupy.array(newobj, copy=False, ndmin=ndmin)
                    if tempobj.ndim < ndmin:
                        newobj = self._output_obj(newobj, tempobj, ndmin, trans1d)
                    del tempobj

            objs.append(newobj)

        res = cupy.concatenate(tuple(objs), axis=self.axis)
        return self._retval(res)

    def __len__(self):
        return 0

class CClass(AxisConcatenator):
    def __init__(self):
        """Translates slice objects to concatenation along the second axis.

        This is short-hand for ``np.r_['-1,2,0', index expression]``, which is
        useful because of its common occurrence. In particular, arrays will be
        stacked along their last axis after being upgraded to at least 2-D with
        1's post-pended to the shape (column vectors made out of 1-D arrays).

        For detailed documentation, see `r_`.
        """
        AxisConcatenator.__init__(self, -1, ndmin=2, trans1d=0)

c_ = CClass()


class RClass(AxisConcatenator):
    def __init__(self):
        """Translates slice objects to concatenation along the first axis.

        This is a simple way to build up arrays quickly. There are two use cases.

        1. If the index expression contains comma separated arrays, then stack
           them along their first axis.
        2. If the index expression contains slice notation or scalars then create
           a 1-D array with a range indicated by the slice notation.(Not Implemented)

        If slice notation is used, the syntax ``start:stop:step`` is equivalent
        to ``np.arange(start, stop, step)`` inside of the brackets. However, if
        ``step`` is an imaginary number (i.e. 100j) then its integer portion is
        interpreted as a number-of-points desired and the start and stop are
        inclusive. In other words ``start:stop:stepj`` is interpreted as
        ``np.linspace(start, stop, step, endpoint=1)`` inside of the brackets.
        After expansion of slice notation, all comma separated sequences are
        concatenated together.

        Optional character strings placed as the first element of the index
        expression can be used to change the output. The strings 'r' or 'c' result
        in matrix output. If the result is 1-D and 'r' is specified a 1 x N (row)
        matrix is produced. If the result is 1-D and 'c' is specified, then a N x 1
        (column) matrix is produced. If the result is 2-D then both provide the
        same matrix result.

        A string integer specifies which axis to stack multiple comma separated
        arrays along. A string of two comma-separated integers allows indication
        of the minimum number of dimensions to force each entry into as the
        second integer (the axis to concatenate along is still the first integer).

        A string with three comma-separated integers allows specification of the
        axis to concatenate along, the minimum number of dimensions to force the
        entries to, and which axis should contain the start of the arrays which
        are less than the specified number of dimensions. In other words the third
        integer allows you to specify where the 1's should be placed in the shape
        of the arrays that have their shapes upgraded. By default, they are placed
        in the front of the shape tuple. The third argument allows you to specify
        where the start of the array should be instead. Thus, a third argument of
        '0' would place the 1's at the end of the array shape. Negative integers
        specify where in the new shape tuple the last dimension of upgraded arrays
        should be placed, so the default is '-1'.

        Args:
            Not a function, so takes no parameters

        Returns:
            cupy.ndarray: Joined array.

        .. seealso:: :func:`numpy.r_`

        """
        AxisConcatenator.__init__(self, 0)

r_ = RClass()

# class s_(object):


# TODO(okuta): Implement indices


# TODO(okuta): Implement ix_


# TODO(okuta): Implement ravel_multi_index


# TODO(okuta): Implement unravel_index


# TODO(okuta): Implement diag_indices


# TODO(okuta): Implement diag_indices_from


# TODO(okuta): Implement mask_indices


# TODO(okuta): Implement tril_indices


# TODO(okuta): Implement tril_indices_from


# TODO(okuta): Implement triu_indices


# TODO(okuta): Implement triu_indices_from
