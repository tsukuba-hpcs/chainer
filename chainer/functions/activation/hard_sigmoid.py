import numpy

from chainer import cuda
from chainer import function
from chainer import utils


class HardSigmoid(function.Function):

    """Hard-sigmoid funciton."""

    def forward_cpu(self, inputs):
        x = inputs[0]
        y = numpy.clip(x * 0.2 + 0.5, 0.0, 1.0)
        return utils.force_array(y).astype(x.dtype),

    def forward_gpu(self, inputs):
        x = inputs[0]
        return cuda.elementwise(
            'T x', 'T y',
            'y = min(1.0, max(0.0, x * 0.2 + 0.5))',
            'hard_sigmoid_fwd'
        )(x),

    def backward_cpu(self, inputs, grads):
        x = inputs[0]
        g = grads[0]
        gx = ((-2.5 < x) & (x < 2.5)) * g * 0.2
        return utils.force_array(gx).astype(x.dtype),

    def backward_gpu(self, inputs, grads):
        x = inputs[0]
        g = grads[0]
        return cuda.elementwise(
            'T x, T g', 'T gx',
            'gx = fabs(x) < 2.5 ? 0.2 * g : 0',
            'hard_sigmoid_bwd'
        )(x, g),


def hard_sigmoid(x):
    """Elementwise hard-sigmoid function.

    This function is defined as

    .. math::

        f(x) = \\left \\{ \\begin{array}{ll}
        0 & {\\rm if}~ x < -0.25 \\\\
        0.2 x + 0.5 & {\\rm if}~ -0.25 < x < 0.25 \\\\
        1 & {\\rm if}~ 0.25 < x.
        \\end{array} \\right.

    Args:
        x (~chainer.Variable): Input variable.

    Returns:
        ~chainer.Variable: Output variable.

    """
    return HardSigmoid()(x)
