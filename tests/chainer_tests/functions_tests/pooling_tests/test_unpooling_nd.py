import unittest

import itertools
import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition
from chainer.utils import conv
from chainer.utils import type_check


def xs_iter(dims):
    return itertools.product(*[range(d) for d in dims])


def kxs_iter(x, outs, ksize, stride, pad):
    return itertools.product(
        *[range(max(0, -p + s * _x), min(-p + s * _x + k, out))
          for (_x, out, k, s, p) in zip(x, outs, ksize, stride, pad)])


def expected_unpooling_nd(x_data, outs, ksize, stride, pad):
    N, c = x_data.shape[:2]
    dims = x_data.shape[2:]
    y_expected_shape = (N, c) + outs
    y_expected = numpy.zeros(y_expected_shape, dtype=x_data.dtype)
    for i in six.moves.range(N):
        for _c in six.moves.range(c):
            for x in xs_iter(dims):
                x_idx = (i, _c) + x
                for kx in kxs_iter(x, outs, ksize, stride, pad):
                    y_idx = (i, _c) + kx
                    y_expected[y_idx] += x_data[x_idx]
    return y_expected


@testing.parameterize(*testing.product({
    'dims': [(4, 3)],
    '_ksize': [1, 2, 3],
    '_stride': [1, 2, 3],
    '_pad': [0, 1],
    'dtype': [numpy.float32],
    'cover_all': [True, False],
}))
class TestUnpoolingND(unittest.TestCase):

    def setUp(self):
        N = 2
        c = 3
        ndim = len(self.dims)
        self.ksize = (self._ksize,) * ndim
        self.stride = (self._stride,) * ndim
        self.pad = (self._pad,) * ndim

        x_shape = (N, c) + self.dims
        self.x = numpy.random.uniform(-1, 1, x_shape).astype(self.dtype)

        outs = tuple(
            conv.get_deconv_outsize(d, k, s, p, cover_all=self.cover_all)
            for (d, k, s, p)
            in zip(self.dims, self.ksize, self.stride, self.pad))
        gy_shape = (N, c) + outs
        self.gy = numpy.random.uniform(-1, 1, gy_shape).astype(self.dtype)

    def check_forward(self, x_data):
        ksize = self.ksize
        stride = self.stride
        pad = self.pad

        # Compute unpooling.
        x = chainer.Variable(x_data)
        y = functions.unpooling_nd(
            x, ksize, stride, pad, cover_all=self.cover_all)

        # Test output's dtype and shape.
        self.assertEqual(y.data.dtype, self.dtype)
        self.assertEqual(y.data.shape, self.gy.shape)

        # Test output's value.
        outs = self.gy.shape[2:]
        y_expected = expected_unpooling_nd(self.x, outs, ksize, stride, pad)
        testing.assert_allclose(y_expected, y.data)

    @condition.retry(3)
    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, y_grad):
        pass

    @condition.retry(3)
    def test_backward_cpu(self):
        pass

    @attr.gpu
    @condition.retry(3)
    def test_backward_gpu(self):
        pass


@testing.parameterize(*testing.product({
    'outsize': [(10,), (10, 9), (10, 9, 8)],
    '_ksize': [1, 2, 3],
    '_stride': [1, 2, 3],
    '_pad': [0, 1],
    'cover_all': [True, False],
}))
class TestUnpoolingNDOutsize(unittest.TestCase):

    def setUp(self):
        self.N = 2
        self.c = 3
        ndim = len(self.outsize)
        self.ksize = (self._ksize,) * ndim
        self.stride = (self._stride,) * ndim
        self.pad = (self._pad,) * ndim

    def test_valid_insize(self):
        N = self.N
        c = self.c
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        outs = self.outsize
        cover_all = self.cover_all

        # Make input.
        dims = tuple(conv.get_conv_outsize(out, k, s, p, cover_all=cover_all)
                     for (out, k, s, p) in zip(outs, ksize, stride, pad))
        x_shape = (N, c) + dims
        x_data = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        x = chainer.Variable(x_data)

        # Compute unpooling.
        y = functions.unpooling_nd(
            x, ksize, stride, pad, outsize=outs, cover_all=cover_all)

        # Test output's value.
        y_expected = expected_unpooling_nd(x_data, outs, ksize, stride, pad)
        testing.assert_allclose(y_expected, y.data)

    def test_invalid_insize(self):
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        outs = self.outsize
        cover_all = self.cover_all

        # Make input with invalid shape.
        dims = tuple(conv.get_conv_outsize(out, k, s, p, cover_all=cover_all)
                     for (out, k, s, p) in zip(outs, ksize, stride, pad))
        dims = tuple(d + 1 for d in dims)  # Make invalid input shape.
        x_shape = (self.N, self.c) + dims
        x_data = numpy.random.uniform(-1, 1, x_shape).astype(numpy.float32)
        x = chainer.Variable(x_data)

        # Computing unpooling raises exception.
        with self.assertRaises(type_check.InvalidType):
            functions.unpooling_nd(
                x, ksize, stride, pad, outsize=outs, cover_all=cover_all)


testing.run_module(__name__, __file__)
