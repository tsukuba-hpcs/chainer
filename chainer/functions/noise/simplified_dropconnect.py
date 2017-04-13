import numpy

from chainer import cuda
from chainer import function
from chainer.utils import type_check
from chainer import variable


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


def _matmul(a, b, xp):
    if xp is numpy:
        # numpy 1.9 does not support matmul.
        # So we use numpy.einsum instead of numpy.matmul.
        return xp.einsum('...jk,...kl->...jl', a, b)
    else:
        return xp.matmul(a, b)


class SimplifiedDropconnect(function.Function):

    """Linear unit regularized by simplified dropconnect."""

    def __init__(self, ratio, batchwise_mask, mask=None):
        self.ratio = ratio
        self.mask = mask
        self.batchwise_mask = batchwise_mask

    def check_type_forward(self, in_types):
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type, w_type = in_types[:2]

        type_check.expect(
            x_type.dtype.kind == 'f',
            w_type.dtype.kind == 'f',
            x_type.ndim >= 2,
            w_type.ndim == 2,
            type_check.prod(x_type.shape[1:]) == w_type.shape[1],
        )
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(
                b_type.dtype == x_type.dtype,
                b_type.ndim == 1,
                b_type.shape[0] == w_type.shape[0],
            )

    def forward(self, inputs):
        scale = inputs[1].dtype.type(1. / (1 - self.ratio))
        xp = cuda.get_array_module(*inputs)
        if self.batchwise_mask:
            mask_shape = (inputs[0].shape[0], inputs[1].shape[0],
                          inputs[1].shape[1])
        else:
            mask_shape = (inputs[1].shape[0], inputs[1].shape[1])

        if self.mask is None:
            if xp == numpy:
                self.mask = xp.random.rand(*mask_shape) >= self.ratio
            else:
                self.mask = xp.random.rand(*mask_shape,
                                           dtype=numpy.float32) >= self.ratio
        elif isinstance(self.mask, variable.Variable):
            self.mask = self.mask.data

        x = _as_mat(inputs[0])
        W = inputs[1] * scale * self.mask

        # (i)jk,ik->ij
        y = _matmul(W, x[:, :, None], xp)
        y = y.reshape(y.shape[0], y.shape[1]).astype(x.dtype, copy=False)

        if len(inputs) == 3:
            b = inputs[2]
            y += b
        return y,

    def backward(self, inputs, grad_outputs):
        scale = inputs[1].dtype.type(1. / (1 - self.ratio))
        x = _as_mat(inputs[0])
        W = inputs[1] * scale * self.mask
        gy = grad_outputs[0]
        xp = cuda.get_array_module(*inputs)

        # ij,(i)jk->ik
        gx = _matmul(gy[:, None, :], W, xp).reshape(inputs[0].shape)
        gx = gx.astype(x.dtype, copy=False)

        # ij,ik,ijk->jk
        gW = (gy[:, :, None] * x[:, None, :] * self.mask).sum(0) * scale
        gW = gW.astype(W.dtype, copy=False)

        if len(inputs) == 3:
            gb = gy.sum(0)
            return gx, gW, gb
        else:
            return gx, gW


def simplified_dropconnect(x, W, b=None, ratio=.5, batchwise_mask=True,
                           train=True, mask=None):
    """Linear unit regularized by simplified dropconnect.

    Simplified dropconnect drops weight matrix elements randomly with
    probability ``ratio`` and scales the remaining elements by factor
    ``1 / (1 - ratio)``.
    It accepts two or three arguments: an input minibatch ``x``, a weight
    matrix ``W``, and optionally a bias vector ``b``. It computes
    :math:`Y = xW^\\top + b`.

    In testing mode, zero will be used as simplified dropconnect ratio instead
    of ``ratio``.

    Notice:
    This implementation cannot be used for reproduction of the paper.
    There is a difference between the current implementation and the
    original one.
    The original version uses sampling with gaussian distribution before
    passing activation function, whereas the current implementation averages
    before activation.

    Args:
        x (chainer.Variable or :class:`numpy.ndarray` or cupy.ndarray):
            Input variable. Its first dimension ``n`` is assumed
            to be the *minibatch dimension*. The other dimensions are treated
            as concatenated one dimension whose size must be ``N``.
        W (~chainer.Variable): Weight variable of shape ``(M, N)``.
        b (~chainer.Variable): Bias variable (optional) of shape ``(M,)``.
        ratio (float):
            Dropconnect ratio.
        batchwise_mask (bool):
            If ``True``, dropped connections depend on each sample in
            mini-batch.
        train (bool):
            If ``True``, executes simplified dropconnect.
            Otherwise, simplified dropconnect function works as a linear
            function.
        mask (None or chainer.Variable or numpy.ndarray or cupy.ndarray):
            If ``None``, randomized dropconnect mask is generated.
            Otherwise, The mask must be ``(n, M, N)`` shaped array.
            Main purpose of this option is debugging.
            `mask` array will be used as a dropconnect mask.

    Returns:
        ~chainer.Variable: Output variable.

    .. seealso:: :class:`~chainer.links.Dropconnect`

    .. seealso::
        Li, W., Matthew Z., Sixin Z., Yann L., Rob F. (2013).
        Regularization of Neural Network using DropConnect.
        International Conference on Machine Learning.
        `URL <http://cs.nyu.edu/~wanli/dropc/>`_
    """
    if not train:
        ratio = 0
    if b is None:
        return SimplifiedDropconnect(ratio, batchwise_mask, mask)(x, W)
    else:
        return SimplifiedDropconnect(ratio, batchwise_mask, mask)(x, W, b)
