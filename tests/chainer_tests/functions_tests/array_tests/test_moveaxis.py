import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.utils import type_check


@testing.parameterize(
    {'source': 0, 'destination': 2, 'out_shape': (3, 4, 2)},
    {'source': -1, 'destination': 1, 'out_shape': (2, 4, 3)},
    {'source': (0, 2), 'destination': (1, 0), 'out_shape': (4, 2, 3)},
    {'source': (0, -1), 'destination': (-1, 1), 'out_shape': (3, 4, 2)},
)
class TestMoveaxis(unittest.TestCase):

    dtype = numpy.float32

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(self.dtype)
        self.g = numpy.random.uniform(-1, 1, self.out_shape).astype(self.dtype)
        self.gg = numpy.random.uniform(-1, 1, (2, 3, 4)).astype(self.dtype)
        self.check_backward_options = {}
        self.check_double_backward_options = {'atol': 1e-3, 'rtol': 1e-2}

    def check_forward(self, x_data):
        x = chainer.Variable(x_data)
        y = functions.moveaxis(x, self.source, self.destination)

        expect = numpy.moveaxis(self.x, self.source, self.destination)
        testing.assert_allclose(y.data, expect)

    def test_forward_cpu(self):
        self.check_forward(self.x)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x_data, g_data):
        def f(x):
            return functions.moveaxis(x, self.source, self.destination)

        gradient_check.check_backward(
            f, x_data, g_data, dtype='d', **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g))

    def check_double_backward(self, x_data, g_data, gg_data):
        def f(x):
            y = functions.moveaxis(x, self.source, self.destination)
            return y * y

        gradient_check.check_double_backward(
            f, x_data, g_data, gg_data, dtype='d',
            **self.check_double_backward_options)

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.g, self.gg)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.g),
                                   cuda.to_gpu(self.gg))


@testing.parameterize(
    {'source': 4, 'destination': 0},
    {'source': 0, 'destination': -4},
)
class TestMoveaxisInvalidType(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(type_check.InvalidType):
            functions.moveaxis(x, self.source, self.destination)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


@testing.parameterize(
    {'source': 0, 'destination': (2,)},
    {'source': (1, 2), 'destination': (1, 2, 0)},
    {'source': (0, 0), 'destination': (1, 2)},
    {'source': (0, 1), 'destination': (2, 2)},
)
class TestMoveaxisInvalidValue(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (2, 3, 4)).astype('f')

    def check_type_error(self, x):
        with self.assertRaises(ValueError):
            functions.moveaxis(x, self.source, self.destination)

    def test_type_error_cpu(self):
        self.check_type_error(self.x)

    @attr.gpu
    def test_type_error_gpu(self):
        self.check_type_error(cuda.to_gpu(self.x))


testing.run_module(__name__, __file__)
