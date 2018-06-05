from chainer.functions.connection import convolution_nd
from chainer import initializers
from chainer import link
from chainer.utils import conv_nd
from chainer import variable


class ConvolutionND(link.Link):
    """N-dimensional convolution layer.

    This link wraps the :func:`~chainer.functions.convolution_nd` function and
    holds the filter weight and bias vector as parameters.

    Convolution links can use a feature of cuDNN called autotuning, which
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
        initialW (:ref:`initializer <initializer>`): Initializer to
            initialize the weight. When it is :class:`numpy.ndarray`,
            its ``ndim`` should be :math:`n+2` where :math:`n` is
            the number of spatial dimensions.
        initial_bias (:ref:`initializer <initializer>`): Initializer to
            initialize the bias. If ``None``, the bias will be initialized to
            zero. When it is :class:`numpy.ndarray`, its ``ndim`` should 1.
        cover_all (bool): If ``True``, all spatial locations are convoluted
            into some output pixels. It may make the output size larger.
            ``cover_all`` needs to be ``False`` if you want to use cuDNN.
        dilate (:class:`int` or :class:`tuple` of :class:`int` s):
            Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        groups (:class:`int`):
            The number of groups to use grouped convolution.
            The default is one, where grouped convolution is not used.

    .. seealso::
        See :func:`~chainer.functions.convolution_nd` for the definition of
        N-dimensional convolution. See
        :func:`~chainer.functions.convolution_2d` for the definition of
        two-dimensional convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter. If ``initial_bias`` is ``None``,
            set to ``None``.

    """

    def __init__(self, ndim, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):
        super(ConvolutionND, self).__init__()

        ksize = conv_nd.as_tuple(ksize, ndim)
        self.stride = stride
        self.pad = pad
        self.cover_all = cover_all
        self.dilate = conv_nd.as_tuple(dilate, ndim)
        self.groups = int(groups)

        with self.init_scope():
            W_shape = (out_channels, in_channels) + ksize
            self.W = variable.Parameter(
                initializers._get_initializer(initialW), W_shape)

            if nobias:
                self.b = None
            else:
                if initial_bias is None:
                    initial_bias = 0
                initial_bias = initializers._get_initializer(initial_bias)
                self.b = variable.Parameter(initial_bias, out_channels)

    def __call__(self, x):
        """Applies N-dimensional convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of convolution.

        """
        return convolution_nd.convolution_nd(
            x, self.W, self.b, self.stride, self.pad, cover_all=self.cover_all,
            dilate=self.dilate, groups=self.groups)


class Convolution1D(ConvolutionND):
    """1-dimensional convolution layer.

    .. note::

        This link wraps :class:`~chainer.links.ConvolutionND` by giving 1 to
        the first argument ``ndim``, so see the details of the behavior in
        the documentation of :class:`~chainer.links.ConvolutionND`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):
        super(Convolution1D, self).__init__(
            1, in_channels, out_channels, ksize, stride, pad, nobias, initialW,
            initial_bias, cover_all, dilate, groups)


class Convolution3D(ConvolutionND):
    """3-dimensional convolution layer.

    .. note::

        This link wraps :class:`~chainer.links.ConvolutionND` by giving 3 to
        the first argument ``ndim``, so see the details of the behavior in
        the documentation of :class:`~chainer.links.ConvolutionND`.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None,
                 cover_all=False, dilate=1, groups=1):
        super(Convolution3D, self).__init__(
            3, in_channels, out_channels, ksize, stride, pad, nobias, initialW,
            initial_bias, cover_all, dilate, groups)
