from chainer.functions.connection import deconvolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd
from chainer import variable


class DeconvolutionND(link.Link):
    """N-dimensional deconvolution function.

    This link wraps :func:`~chainer.functions.deconvolution_nd` function and
    holds the filter weight and bias vector as its parameters.

    Deconvolution links can use a feature of cuDNN called autotuning, which
    selects the most efficient CNN algorithm for images of fixed-size,
    can provide a significant performance boost for fixed neural nets.
    To enable, set `chainer.using_config('autotune', True)`

    Args:
        ndim (int): Number of spatial dimensions.
        in_channels (int): Number of channels of input arrays.
        out_channels (int): Number of channels of output arrays.
        ksize (int or tuple of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k, ..., k)`` are equivalent.
        stride (int or tuple of ints): Stride of filter application.
            ``stride=s`` and ``stride=(s, s, ..., s)`` are equivalent.
        pad (int or tuple of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.
        nobias (bool): If ``True``, then this function does not use the bias.
        outsize (tuple of ints): Expected output size of deconvolutional
            operation. It should be a tuple of ints that represents the output
            size of each dimension. Default value is ``None`` and the outsize
            is estimated with input size, stride and pad.
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be :math:`n+2` where :math:`n` is
            the number of spatial dimensions.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should 1.
        dilate (:class:`int` or :class:`tuple` of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (:class:`int`):
            The number of groups to use grouped convolution.
            The default is one, where grouped convolution is not used.

    .. seealso::
       :func:`~chainer.functions.deconvolution_nd`

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize=None, stride=1,
                 pad=0, nobias=False, outsize=None, initialW=None,
                 initial_bias=None, dilate=1, groups=1):
        super(DeconvolutionND, self).__init__()

        if ksize is None:
            ndim, out_channels, ksize, in_channels = \
                ndim, in_channels, out_channels, None

        self.out_channels = out_channels
        self.ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.outsize = outsize
        self.dilate = conv_nd.as_tuple(dilate, ndim)
        self.groups = int(groups)

        with self.init_scope():
            W_initializer = initializers._get_initializer(initialW)
            self.W = variable.Parameter(W_initializer)
            if in_channels is not None:
                self._initialize_params(in_channels)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                initial_bias = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(initial_bias, out_channels)

    def _initialize_params(self, in_channels):
        W_shape = (in_channels, self.out_channels) + self.ksize
        self.W.initialize(W_shape)

    def forward(self, x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return deconvolution_nd.deconvolution_nd(
            x, self.W, b=self.b, stride=self.stride, pad=self.pad,
            outsize=self.outsize, dilate=self.dilate, groups=self.groups)
