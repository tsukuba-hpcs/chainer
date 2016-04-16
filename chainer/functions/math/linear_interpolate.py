import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check


class LinearInterpolate(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 3)
        p_type, x_type, y_type = in_types

        type_check.expect(
            p_type.dtype == numpy.float32,
            x_type.dtype == numpy.float32,
            y_type.dtype == numpy.float32,
            p_type.shape == x_type.shape,
            p_type.shape == y_type.shape,
        )

    def forward_cpu(self, inputs):
        p, x, y = inputs
        return p * x + (1 - p) * y,

    def forward_gpu(self, inputs):
        p, x, y = inputs
        return cuda.elementwise(
            'T p, T x, T y', 'T z',
            'z = p * x + (1 - p) * y',
            'linear_interpolate_fwd',
        )(p, x, y),

    def backward_cpu(self, inputs, grads):
        p, x, y = inputs
        g = grads[0]
        pg = p * g
        return (x - y) * g, pg, g - pg

    def backward_gpu(self, inputs, grads):
        p, x, y = inputs
        g = grads[0]
        return cuda.elementwise(
            'T p, T x, T y, T g', 'T gp, T gx, T gy',
            '''
            gp = (x - y) * g;
            gx = g * p;
            gy = g * (1 - p);
            ''',
            'linear_interpolate_bwd'
        )(p, x, y, g)


def linear_interpolate(p, x, y):
    """Elementwise linear-interpolation function.

    This function is defined as

    .. math::

        f(p, x, y) = p x + (1 - p) y.

    Args:
        p (~chainer.Variable): Input variable.
        x (~chainer.Variable): Input variable.
        y (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """

    return LinearInterpolate()(p, x, y)
