from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _kern():
    return cuda.elementwise(
        'T cond, T x, T slope', 'T y',
        'y = cond >= 0 ? x : (T)(slope * x)', 'lrelu')


class LeakyReLU(function.Function):

    """Leaky rectifier unit."""

    def __init__(self, slope=0.2):
        self.slope = slope

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 1)
        x_type, = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, x):
        y = x[0].copy()
        y[x[0] < 0] *= self.slope
        if self.slope >= 0:
            self.retain_inputs(())
            self.retain_outputs((0,))
        return y,

    def forward_gpu(self, x):
        y = _kern()(x[0], x[0], self.slope)
        if self.slope >= 0:
            self.retain_inputs(())
            self.retain_outputs((0,))
        return y,

    def backward_cpu(self, x, gy):
        gx = gy[0].copy()
        if self.slope >= 0:
            y = self.output_data
            gx[y[0] < 0] *= self.slope
        else:
            gx[x[0] < 0] *= self.slope
        return gx,

    def backward_gpu(self, x, gy):
        if self.slope >= 0:
            y = self.output_data
            gx = _kern()(y[0], gy[0], self.slope)
        else:
            gx = _kern()(x[0], gy[0], self.slope)
        return gx,


def leaky_relu(x, slope=0.2):
    """Leaky Rectified Linear Unit function.

    This function is expressed as

    .. math:: f(x)=\\max(x, ax),

    where :math:`a` is a configurable slope value.

    Args:
        x (:class:`~chainer.Variable` or :class:`numpy.ndarray` or \
        :class:`cupy.ndarray`):
            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.
        slope (float): Slope value :math:`a`.

    Returns:
        ~chainer.Variable: Output variable. A
        :math:`(s_1, s_2, ..., s_N)`-shaped float array.

    .. admonition:: Example

        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], 'f')
        >>> x
        array([[-1.,  0.],
               [ 2., -3.],
               [-2.,  1.]], dtype=float32)
        >>> F.leaky_relu(x, slope=0.2).data
        array([[-0.2       ,  0.        ],
               [ 2.        , -0.60000002],
               [-0.40000001,  1.        ]], dtype=float32)

    """
    return LeakyReLU(slope)(x)
