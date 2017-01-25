import math

import numpy

from chainer import cuda
from chainer.functions.connection import dilated_convolution_2d
from chainer import initializers
from chainer import link


class DilatedConvolution2D(link.Link):

    """Two-dimensional dilated convolutional layer.

    This link wraps the :func:`~chainer.functions.dilated_convolution_2d`
    function and holds the filter weight and bias vector as parameters.

    Args:
        in_channels (int): Number of channels of input arrays. If ``None``,
            parameter initialization will be deferred until the first forward
            data pass at which time the size will be determined.
        out_channels (int): Number of channels of output arrays.
        ksize (int or pair of ints): Size of filters (a.k.a. kernels).
            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.
        stride (int or pair of ints): Stride of filter applications.
            ``stride=s`` and ``stride=(s, s)`` are equivalent.
        pad (int or pair of ints): Spatial padding width for input arrays.
            ``pad=p`` and ``pad=(p, p)`` are equivalent.
        dilate (int or pair of ints): Dilation factor of filter applications.
            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.
        use_cudnn (bool): If ``True``, then this link uses cuDNN if available.
        initialW (4-D array): Initial weight value. If ``None``, :func:`HeNormal`
            initializer is used to initialize weight matrix.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.
        initial_bias (1-D array): Initial bias value. If ``None``, then this
            function uses to initialize ``bias``.
            May also be a callable that takes ``numpy.ndarray`` or
            ``cupy.ndarray`` and edits its value.

    .. seealso::
       See :func:`chainer.functions.dilated_convolution_2d`
       for the definition of two-dimensional dilated convolution.

    Attributes:
        W (~chainer.Variable): Weight parameter.
        b (~chainer.Variable): Bias parameter.

    """

    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0,
                 dilate=1, use_cudnn=True,
                 initialW=initializers.HeNormal(1.0 / numpy.sqrt(2)),
                 initial_bias=initializers.Constant(0)):
        super(DilatedConvolution2D, self).__init__()
        self.ksize = ksize
        self.stride = _pair(stride)
        self.pad = _pair(pad)
        self.dilate = _pair(dilate)
        self.use_cudnn = use_cudnn
        self.out_channels = out_channels
        self.initialW = initialW

        if in_channels is None:
            self.add_uninitialized_param('W')
        else:
            self._initialize_params(in_channels)

        if initial_bias is None:
            self.b = None
        else:
            bias_initializer = initializers._get_initializer(initial_bias)
            self.add_param('b', out_channels, initializer=bias_initializer)

    def _initialize_params(self, in_channels):
        kh, kw = _pair(self.ksize)
        W_shape = (self.out_channels, in_channels, kh, kw)
        self.add_param('W', W_shape)
        initializers.init_weight(self.W.data, self.initialW)

    def __call__(self, x):
        """Applies the convolution layer.

        Args:
            x (~chainer.Variable): Input image.

        Returns:
            ~chainer.Variable: Output of the convolution.

        """
        if self.has_uninitialized_params:
            with cuda.get_device(self._device_id):
                self._initialize_params(x.shape[1])
        return dilated_convolution_2d.dilated_convolution_2d(
            x, self.W, self.b, self.stride,
            self.pad, self.dilate, self.use_cudnn)


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x
