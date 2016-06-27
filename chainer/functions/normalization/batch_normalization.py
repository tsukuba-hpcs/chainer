import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check

if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cudnn.cudnn
    _cudnn_version = libcudnn.getVersion()


def _as4darray(arr):
    if arr.ndim == 0:
        return arr.reshape(1, 1, 1, 1)
    elif arr.ndim == 4:
        return arr
    else:
        return arr.reshape(arr.shape[0], -1, 1, 1)


class BatchNormalizationFunction(function.Function):

    def __init__(self, mean, var, train, decay=0.9, eps=2e-5, use_cudnn=True):
        if train:
            self.running_mean = mean
            self.running_var = var
        else:
            # Test/evaluation mode
            self.running_mean = None
            self.running_var = None
            self.fixed_mean = mean
            self.fixed_var = var
        # If train is true, use batch statistics (training mode). Otherwise, if
        # false, use the supplied mean and variance.
        self.train = train
        # Note: cuDNN v5 requires that eps be greater than 1e-5. Otherwise, an
        # error will occur.
        # See CUDNN_BN_MIN_EPSILON value in cudnn.h to verify minimum allowable
        # value.
        self.eps = eps
        if cuda.cudnn_enabled and use_cudnn:
            if eps <= 1e-5:
                msg = 'cuDNN does not allow an eps value less than 1e-5.'
                raise RuntimeError(msg)
        self.use_cudnn = use_cudnn
        self.mean_cache = None
        self.x_hat = None
        self.decay = decay

    def check_type_forward(self, in_types):
        n_in = in_types.size().eval()
        if n_in != 3:
            raise type_check.InvalidType(
                '%s or %s' % (in_types.size() == 3, in_types.size() == 5),
                '%s == %s' % (in_types.size(), n_in))

        x_type, gamma_type, beta_type = in_types[:3]
        type_check.expect(
            x_type.dtype.kind == 'f',
            x_type.ndim >= gamma_type.ndim + 1,
            # TODO(beam2d): Check shape
            gamma_type.dtype == x_type.dtype,
            beta_type.dtype == x_type.dtype,
            gamma_type.shape == beta_type.shape,
        )

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        x, gamma, beta = inputs
        if x.ndim == 4:
            # for convolutional layer
            self.mode = libcudnn.CUDNN_BATCHNORM_SPATIAL
        else:
            # for linear layer
            self.mode = libcudnn.CUDNN_BATCHNORM_PER_ACTIVATION

        if x[0].dtype == numpy.float16:
            # cuDNN v5 batch normalization does not seem to support float16.
            self.use_cudnn = False

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        gamma = gamma[expander]
        beta = beta[expander]

        if self.train:
            self.running_mean = xp.array(self.running_mean)
            self.running_var = xp.array(self.running_var)
        else:
            # Make sure the attribute arrays are the same type (numpy/cupy)
            # as the inputs.
            self.fixed_mean = xp.array(self.fixed_mean)
            self.fixed_var = xp.array(self.fixed_var)

        if (x.ndim == 2) or (x.ndim == 4):
            # cuDNN only supports these tensor dimensions because they are
            # the most commonly used. If there is a need to support other
            # dimensions with cuDNN, we could consider reshaping the input
            # into a 2-dim array with channels as second dim and m=<product
            # of all dimensions except the 2nd dimension> as the first
            # dimension.
            self.cudnn_dim_ok = True
        else:
            self.cudnn_dim_ok = False

        cudnn_updated_running_stats = False
        if xp is numpy:
            if self.train:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mean = x.mean(axis=axis)
                var = x.var(axis=axis)
                var += self.eps
            else:
                mean = self.fixed_mean
                var = self.fixed_var
            self.std = xp.sqrt(var, dtype=var.dtype)
            x_mu = x - mean[expander]
            self.x_hat = x_mu / self.std[expander]
            y = gamma * self.x_hat
            y += beta
        elif cuda.cudnn_enabled and self.use_cudnn and self.cudnn_dim_ok:
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            y = cuda.cupy.empty_like(x)
            # Factor used in the moving average
            factor = 1 - self.decay
            # computation runningMean
            if self.mean_cache is None:
                # Output cache to speed up bacward pass (recommended to enable)
                self.mean_cache = xp.zeros_like(gamma)
                # Output cache to speed up bacward pass (recommended to enable)
                self.var_cache = xp.zeros_like(gamma)

            if self.running_mean is None:
                self.running_mean = xp.zeros_like(gamma)
                self.running_var = xp.zeros_like(gamma)

            if self.train:
                # Note: cuDNN computes the mini-batch mean and variance
                # internally. We can simply (optionally) pass
                # it the running-average mean and variance arrays.
                libcudnn.batchNormalizationForwardTraining(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value,
                    y.data.ptr, derivedBnDesc.value, gamma.data.ptr,
                    beta.data.ptr, factor, self.running_mean.data.ptr,
                    self.running_var.data.ptr, self.eps,
                    self.mean_cache.data.ptr, self.var_cache.data.ptr)
                cudnn_updated_running_stats = True
            else:
                libcudnn.batchNormalizationForwardInference(
                    handle, self.mode, one.data, zero.data,
                    x_desc.value, x.data.ptr, x_desc.value, y.data.ptr,
                    derivedBnDesc.value, gamma.data.ptr, beta.data.ptr,
                    self.fixed_mean.data.ptr, self.fixed_var.data.ptr,
                    self.eps)

        else:
            if self.train:
                axis = (0,) + tuple(range(head_ndim, x.ndim))
                mean = x.mean(axis=axis)
                var = x.var(axis=axis)
                var += self.eps
            else:
                mean = self.fixed_mean
                var = self.fixed_var
            self.std = xp.sqrt(var, dtype=var.dtype)
            self.x_hat, y = cuda.elementwise(
                'T x, T mean, T std, T gamma, T beta', 'T x_hat, T y',
                '''
                   x_hat = (x - mean) / std;
                   y = gamma * x_hat + beta;
                ''',
                'bn_fwd')(x, mean[expander], self.std[expander], gamma, beta)

        if self.train and (not cudnn_updated_running_stats):
            # Note: If in training mode, the cuDNN forward training function
            # will do this for us, so
            # only run following code if cuDNN was not used.
            # Update running statistics:
            m = x.size // gamma.size
            adjust = m / max(m - 1., 1.)  # unbiased estimation
            self.running_mean *= self.decay
            temp_ar = xp.array(mean)
            temp_ar *= (1 - self.decay)
            self.running_mean += temp_ar
            del temp_ar
            self.running_var *= self.decay
            temp_ar = xp.array(var)
            temp_ar *= (1 - self.decay) * adjust
            self.running_var += temp_ar
            del temp_ar
        return y,

    def backward(self, inputs, grad_outputs):
        x, gamma = inputs[:2]
        gy = grad_outputs[0]

        if x[0].dtype == numpy.float16:
            # cuDNN v5 batch normalization does not seem to support float16.
            self.use_cudnn = False

        head_ndim = gamma.ndim + 1
        expander = (None, Ellipsis) + (None,) * (x.ndim - head_ndim)
        m = gamma.dtype.type(x.size // gamma.size)
        axis = (0,) + tuple(range(head_ndim, x.ndim))
        xp = cuda.get_array_module(x)

        if xp is numpy:
            gbeta = gy.sum(axis=axis)
            ggamma = (gy * self.x_hat).sum(axis=axis)
            if self.train:
                gx = (gamma / self.std)[expander] * (
                    gy - (self.x_hat * ggamma[expander] + gbeta[expander]) / m)
            else:
                # Note: Under normal conditions, this will never be executed
                # because fixed-mean-variance mode is normally
                # only used at test/evaluation time.
                gx = (gamma / self.std)[expander] * gy
        elif cuda.cudnn_enabled and self.use_cudnn and self.cudnn_dim_ok and \
                self.train:
            # Note: cuDNN batch normalization backward only works in
            # "training mode." That is, it does not support
            # computing gradients in fixed-mean-variance mode, because there
            # is normally no reason to call backward()
            # while in test/evaluation mode.
            dtype = x.dtype
            handle = cudnn.get_handle()
            x_desc = cudnn.create_tensor_descriptor(_as4darray(x))
            derivedBnDesc = cudnn.create_uninitialized_tensor_descriptor()
            libcudnn.deriveBNTensorDescriptor(derivedBnDesc.value,
                                              x_desc.value, self.mode)
            one = numpy.array(1, dtype=dtype).ctypes
            zero = numpy.array(0, dtype=dtype).ctypes
            gx = cuda.cupy.empty_like(x)
            ggamma = cuda.cupy.empty_like(gamma)
            gbeta = cuda.cupy.empty_like(gamma)
            libcudnn.batchNormalizationBackward(
                handle, self.mode, one.data, zero.data,
                one.data, zero.data, x_desc.value, x.data.ptr,
                x_desc.value, gy.data.ptr, x_desc.value, gx.data.ptr,
                derivedBnDesc.value, gamma.data.ptr,
                ggamma.data.ptr, gbeta.data.ptr,
                self.eps, self.mean_cache.data.ptr, self.var_cache.data.ptr)
        else:
            gbeta = gy.sum(axis=axis)
            if self.train:
                ggamma = (gy * self.x_hat).sum(axis=axis)
                inv_m = numpy.float32(1) / m
                gx = cuda.elementwise(
                    'T gy, T x_hat, T gamma, T std, T ggamma, T gbeta, \
                    T inv_m',
                    'T gx',
                    'gx = (gamma / std) * (gy - (x_hat * ggamma + gbeta) * \
                    inv_m)',
                    'bn_bwd')(gy, self.x_hat, gamma[expander],
                              self.std[expander],
                              ggamma[expander], gbeta[expander], inv_m)
            else:
                # Note: Under normal conditions, this will never be executed
                # because fixed-mean-variance mode is normally
                # only used at test/evaluation time.
                # If cuDNN was called on the forward call, self.x_hat will
                # be None.
                if self.x_hat is None:
                    self.std = xp.sqrt(self.fixed_var,
                                       dtype=self.fixed_var.dtype)
                    x_mu = x - self.fixed_mean[expander]
                    self.x_hat = x_mu / self.std[expander]
                ggamma = (gy * self.x_hat).sum(axis=axis)
                gx = cuda.elementwise(
                    'T gy, T x_hat, T gamma, T std',
                    'T gx',
                    'gx = (gamma / std) * gy',
                    'bn_bwd')(gy, self.x_hat, gamma[expander],
                              self.std[expander])
        return gx, ggamma, gbeta


def batch_normalization(x, gamma, beta, running_mean=None,
                        running_var=None, decay=0.9, eps=2e-5):
    """Batch normalization function.

    It takes the input variable ``x`` and two parameter variables ``gamma`` and
    ``beta``. The input must have the batch size and the features (or channels)
    as the first two dimensions of its shape. The input can have more than two
    dimensions, where the remained dimensions are considered as spatial
    dimensions, which are considered as a part of the batch size.

    Note: If this function is called, it will not be possible to access the
    updated running mean and variance statistics, because they are members
    of the function object, which cannot be accessed by the caller.
    If it is desired to access the updated running statistics, it is necessary
    to get a new instance of the function object, call the object, and then
    access the running_mean and/or running_var attributes. See the
    corresponding Link class for an example of how to do this.

    Args:
        x (Variable): The input variable.
        gamma (Variable): The scaling parameter of normalized data.
        beta (Variable): The shifting parameter of scaled normalized data.
        running_mean (array): The running average of the mean. This is a
        running average of the mean over several
            mini-batches using the decay parameter. If None, the running
            average is not computed. If this is None,
            then runnng_var must also be None.
        running_var (array): The running average of the variance. This is a
        running average of the variance
            over several mini-batches using the decay parameter. If None, the
            running average is not computed. If
            this is None, then running_mean must also be None.
        decay (float): Decay rate of moving average. It is used during
            training.
        eps (float): Epsilon value for numerical stability.

    See: `Batch Normalization: Accelerating Deep Network Training by Reducing\
          Internal Covariate Shift <http://arxiv.org/abs/1502.03167>`_

    .. seealso:: :class:`links.BatchNormalization`

    """
    return BatchNormalizationFunction(running_mean, running_var, True,
                                      decay, eps)(x, gamma, beta)


def fixed_batch_normalization(x, gamma, beta, fixed_mean, fixed_var, eps=2e-5):
    """Batch normalization function with fixed statistics.

    This is a variant of batch normalization, where the mean and variance
    statistics are given by the caller as fixed parameters. This is
    used on testing mode of the batch normalization layer, where batch
    statistics cannot be used for prediction consistency.

    Args:
        x (Variable): The input variable.
        gamma (Variable): The scaling parameter of normalized data.
        beta (Variable): The shifting parameter of scaled normalized data.
        fixed_mean (array): The shifting parameter of input.
        fixed_var (array): The square of scaling parameter of input.
        eps (float): Epsilon value for numerical stability.

    .. seealso::
       :func:`functions.batch_normalization`,
       :class:`links.BatchNormalization`

    """
    return BatchNormalizationFunction(fixed_mean, fixed_var, False, 0.0,
                                      eps)(x, gamma, beta)
