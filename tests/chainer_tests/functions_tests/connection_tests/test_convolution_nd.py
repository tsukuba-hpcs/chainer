import unittest

import functools
import mock
import numpy
from operator import mul

import chainer
from chainer import cuda
from chainer import functions
from chainer.functions.connection import convolution_nd
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv


@testing.parameterize(*testing.product({
    'dims': [(10,), (10, 8), (10, 8, 6)],
    'c_contiguous': [True, False],
    'cover_all': [True, False],
    'x_dtype': [numpy.float16, numpy.float32, numpy.float64],
    'W_dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestConvolutionND(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim

        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (out_channels, in_channels) + ksize
        self.W = numpy.random.normal(0, W_scale, W_shape).astype(self.W_dtype)
        self.b = numpy.random.uniform(-1, 1, out_channels).astype(self.x_dtype)

        x_shape = (2, 3) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.x_dtype)
        gy_shape = (2, 2) + tuple(
            conv.get_conv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p) in zip(self.dims, ksize, self.stride, self.pad))
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.x_dtype)

        self.check_forward_options = {}
        self.check_backward_options = {
            'eps': 1e-2, 'atol': 5e-5, 'rtol': 5e-4}
        if self.x_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'eps': 2 ** -3, 'atol': 1e-2, 'rtol': 1e-1}
        elif self.W_dtype == numpy.float16:
            self.check_forward_options = {'atol': 5e-4, 'rtol': 5e-3}
            self.check_backward_options = {
                'eps': 2 ** -3, 'atol': 1e-3, 'rtol': 1e-2}

    def check_forward_consistency(self, nobias=False, use_cudnn=False):
        x_cpu = chainer.Variable(self.x)
        W_cpu = chainer.Variable(self.W)
        b_cpu = None if nobias else chainer.Variable(self.b)
        y_cpu = functions.convolution_nd(
            x_cpu, W_cpu, b_cpu, stride=self.stride, pad=self.pad,
            use_cudnn=use_cudnn, cover_all=self.cover_all)

        x_gpu = chainer.Variable(cuda.to_gpu(self.x))
        W_gpu = chainer.Variable(cuda.to_gpu(self.W))
        b_gpu = None if nobias else chainer.Variable(cuda.to_gpu(self.b))
        y_gpu = functions.convolution_nd(
            x_gpu, W_gpu, b_gpu, stride=self.stride, pad=self.pad,
            use_cudnn=use_cudnn, cover_all=self.cover_all)

        gradient_check.assert_allclose(
            y_cpu.data, y_gpu.data.get(), **self.check_forward_options)

    @attr.cudnn
    def test_forward_consistency(self):
        self.check_forward_consistency(nobias=False, use_cudnn=True)

    @attr.cudnn
    def test_forward_consistency_nobias(self):
        self.check_forward_consistency(nobias=True, use_cudnn=True)

    @attr.gpu
    def test_forward_consistency_im2col(self):
        self.check_forward_consistency(nobias=False, use_cudnn=False)

    @attr.gpu
    def test_forward_consistency_im2col_nobias(self):
        self.check_forward_consistency(nobias=True, use_cudnn=False)

    def check_backward(self, x_data, W_data, b_data, y_grad, use_cudnn=False):
        xp = cuda.get_array_module(x_data)
        if not self.c_contiguous:
            x_data = xp.asfortranarray(x_data)
            W_data = xp.asfortranarray(W_data)
            y_grad = xp.asfortranarray(y_grad)
            self.assertFalse(x_data.flags.c_contiguous)
            self.assertFalse(W_data.flags.c_contiguous)
            self.assertFalse(y_grad.flags.c_contiguous)
            if b_data is not None:
                b = xp.empty((len(b_data) * 2,), dtype=b_data.dtype)
                b[::2] = b_data
                b_data = b[::2]
                self.assertFalse(b_data.flags.c_contiguous)

        args = (x_data, W_data)
        if b_data is not None:
            args = args + (b_data,)

        ndim = len(self.dims)
        gradient_check.check_backward(
            convolution_nd.ConvolutionND(
                ndim, self.stride, self.pad, use_cudnn, self.cover_all),
            args, y_grad, **self.check_backward_options)

    @condition.retry(3)
    def test_backward_cpu(self):
        self.check_backward(self.x, self.W, self.b, self.gy)

    @condition.retry(3)
    def test_backward_cpu_nobias(self):
        self.check_backward(self.x, self.W, None, self.gy)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy),
                            use_cudnn=True)

    @attr.cudnn
    @condition.retry(3)
    def test_backward_gpu_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy),
                            use_cudnn=True)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            cuda.to_gpu(self.b), cuda.to_gpu(self.gy),
                            use_cudnn=False)

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu_im2col_nobias(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.W),
                            None, cuda.to_gpu(self.gy),
                            use_cudnn=False)


@testing.parameterize(*testing.product({
    'dims': [(10,), (10, 8), (10, 8, 6)],
    'use_cudnn': [True, False],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
@attr.cudnn
class TestConvolutionNDCudnnCall(unittest.TestCase):

    def setUp(self):
        in_channels = 3
        out_channels = 2
        ndim = len(self.dims)
        ksize = (3,) * ndim
        self.stride = (2,) * ndim
        self.pad = (1,) * ndim
        x_shape = (2, 3) + self.dims
        self.x = cuda.cupy.random.uniform(-1, 1, x_shape).astype(self.dtype)
        W_scale = numpy.sqrt(1. / functools.reduce(mul, ksize, in_channels))
        W_shape = (out_channels, in_channels) + ksize
        self.W = cuda.cupy.random.normal(
            0, W_scale, W_shape).astype(self.dtype)
        gy_shape = (2, 2) + tuple(
            conv.get_conv_outsize(d, k, s, p) for (d, k, s, p) in zip(
                self.dims, ksize, self.stride, self.pad))
        self.gy = cuda.cupy.random.uniform(-1, 1, gy_shape).astype(self.dtype)
        self.expect = self.use_cudnn and ndim > 1 and (
            cuda.cudnn.cudnn.getVersion() >= 3000 or
            self.dtype != numpy.float16)

    def forward(self):
        x = chainer.Variable(cuda.to_gpu(self.x))
        W = chainer.Variable(cuda.to_gpu(self.W))
        return functions.convolution_nd(
            x, W, None, stride=self.stride, pad=self.pad,
            use_cudnn=self.use_cudnn)

    def test_call_cudnn_forward(self):
        with mock.patch('cupy.cudnn.cudnn.convolutionForward') as func:
            self.forward()
            self.assertEqual(func.called, self.expect)

    def test_call_cudnn_backward(self):
        y = self.forward()
        y.grad = self.gy
        if cuda.cudnn.cudnn.getVersion() >= 4000:
            name = 'cupy.cudnn.cudnn.convolutionBackwardData_v3'
        else:
            name = 'cupy.cudnn.cudnn.convolutionBackwardData_v2'
        with mock.patch(name) as func:
            y.backward()
            self.assertEqual(func.called, self.expect)

testing.run_module(__name__, __file__)
