import numpy

import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import type_check


class Sum(function_node.FunctionNode):
    """Sum of array elements over a given axis."""

    keepdims = False

    def __init__(self, axis=None, keepdims=False):
        if axis is None:
            self.axis = None
        elif isinstance(axis, int):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all(isinstance(a, int) for a in axis):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(
                    ', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

        self.keepdims = keepdims

    def check_type_forward(self, in_types):
        type_check.expect(
            in_types.size() == 1,
            in_types[0].dtype.kind == 'f',
        )

        if self.axis is not None:
            for axis in self.axis:
                if axis >= 0:
                    type_check.expect(
                        axis < in_types[0].ndim,
                    )
                else:
                    type_check.expect(
                        -axis - 1 < in_types[0].ndim,
                    )

    def forward(self, inputs):
        x, = inputs
        ret = x.sum(axis=self.axis, keepdims=self.keepdims)
        if cuda.get_array_module(x) is numpy:
            ret = numpy.asarray(ret)
        return ret,

    def backward(self, indexes, grad_outputs):
        gy, = grad_outputs
        ndim = len(self.inputs[0].shape)
        if not (ndim == 0 or self.axis is None or self.keepdims):
            actual_axis = [
                axis if axis >= 0 else axis + ndim
                for axis in self.axis]
            shape = list(gy.shape)
            for axis in sorted(actual_axis):
                shape.insert(axis, 1)
            gy = chainer.functions.reshape(gy, shape)
        return chainer.functions.broadcast_to(gy, self.inputs[0].shape),


def sum(x, axis=None, keepdims=False):
    """Sum of array elements over a given axis.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Elements to sum.
            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.
        axis (None, int, or tuple of int): Axis along which a sum is performed.
            The default (axis = None) is perform a sum over all the dimensions
            of the input array.
        keepdims (bool): If ``True``, the specified axes are remained as axes
            of length one.

    Returns:
        ~chainer.Variable: Output variable.

    .. admonition:: Example

        >>> x = np.arange(6).reshape(2,3).astype('f')
        >>> x
        array([[0., 1., 2.],
               [3., 4., 5.]], dtype=float32)
        >>> y = F.sum(x)
        >>> y.shape
        ()
        >>> y.data
        array(15., dtype=float32)
        >>> y = F.sum(x, axis=1)
        >>> y.shape
        (2,)
        >>> y.data
        array([ 3., 12.], dtype=float32)
        >>> y = F.sum(x, keepdims=True)
        >>> y.shape
        (1, 1)
        >>> y.data
        array([[15.]], dtype=float32)

    """
    y, = Sum(axis, keepdims).apply((x,))
    return y
