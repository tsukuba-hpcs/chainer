import math

from chainer import cuda
from chainer.functions.noise import dropconnect
from chainer import initializers
from chainer import link


class Dropconnect(link.Link):

    """Fully-connected layer with dropconnect regularization.

    Args:
        in_size (int): Dimension of input vectors. If ``None``, parameter
            initialization will be deferred until the first forward data pass
            at which time the size will be determined.
        out_size (int): Dimension of output vectors.
        wscale (float): Scaling factor of the weight matrix.
        initialW (3-D array or None): Initial weight value.
            If ``None``, then this function uses ``wscale`` to initialize.
        initial_bias (2-D array, float or None): Initial bias value.
            If it is float, initial bias is filled with this value.
            If ``None``, bias is omitted.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    .. seealso:: :func:`~chainer.functions.dropconnect`

    .. seealso::
        Li, W., Matthew Z., Sixin Z., Yann L., Rob F. (2013).
        Regularization of Neural Network using DropConnect.
        International Conference on Machine Learning.
        `URL <http://cs.nyu.edu/~wanli/dropc/>`_
    """

    def __init__(self, in_size, out_size, wscale=1,
                 ratio=.5, initialW=None, initial_bias=0):
        super(Dropconnect, self).__init__()

        self.out_size = out_size
        self.ratio = ratio

        # Square root is for the compatibility with Linear Function.
        self._W_initializer = initializers._get_initializer(
            initialW, math.sqrt(wscale))

        if in_size is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_size)

        if initial_bias is not None:
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_size, initializer=bias_initializer)

    def _initialize_params(self, in_size):
        self.add_param('W', (self.out_size, in_size),
                       initializer=self._W_initializer)

    def __call__(self, x, train=True, mask=None):
        """Applies the dropconnect layer.

        Args:
            x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
                Batch of input vectors. Its first dimension ``n`` is assumed
                to be the *minibatch dimension*.
            train (bool):
                If ``True``, executes dropconnect.
                Otherwise, dropconnect link works as a linear unit.
            mask (None or chainer.Variable or :class:`numpy.ndarray` or
                cupy.ndarray):
                If ``None``, randomized dropconnect mask is generated.
                Otherwise, The mask must be ``(n, M, N)`` shaped array.
                Main purpose of this option is debugging.
                `mask` array will be used as a dropconnect mask.

        Returns:
            ~chainer.Variable: Output of the dropconnect layer.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.size // len(x.data))
        if mask is not None and 'mask' not in self.__dict__:
            self.add_persistent('mask', mask)
        return dropconnect.dropconnect(x, self.W, self.b,
                                       self.ratio, train, mask)
