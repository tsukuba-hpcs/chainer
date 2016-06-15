import chainer
from chainer.functions.math import bias
from chainer import link


class Bias(link.Link):
    """Broadcasted elementwise summation with learnable parameter.

    Computes a elementwise summation as :func:`~chainer.functions.bias`
    function does except that its second input is a learnable bias parameter
    :math:`b` the link has.

    Args:
        axis (int): The first axis of the first input of
            :func:`~chainer.functions.bias` function along which its second
            input is applied.
        shape (tuple of ints): Shape of the learnable bias parameter. If
            ``None``, a bias parameter needs to be given explicitly to its
            ``__call__`` method's second input.

    .. seealso:: See :func:`chainer.functions.bias` for details.

    Attributes:
        b (~chainer.Variable): Bias parameter.

    """
    def __init__(self, axis=1, shape=None):
        super(Bias, self).__init__()

        # Add b parameter if given.
        if shape is not None:
            self.add_param('b', shape)
            self.b.data.fill(0)

        self.axis = axis

    def __call__(self, *xs):
        """Applies broadcasted elementwise summation.

        Args:
            xs (list of ~chainer.Variable): Input variables whose length should
                be one if the link has a learnable bias parameter, otherwise
                should be two.
        """
        axis = self.axis

        # Case of only one bottom where b is learnt parameter.
        if hasattr(self, 'b'):
            if chainer.is_debug():
                assert len(xs) == 1
            x, = xs
            b = self.b
            return bias.bias(x, b, axis)
        # Case of two bottoms where b is given as a bottom.
        else:
            if chainer.is_debug():
                assert len(xs) == 2
            x, y = xs
            return bias.bias(x, y, axis)
