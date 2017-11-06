import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


class TestForget(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.y = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.gz = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggx = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)
        self.ggy = numpy.random.uniform(-1, 1, (3, 4)).astype(numpy.float32)

    def check_forward(self, x_data, y_data):
        x = chainer.Variable(x_data)
        y = chainer.Variable(y_data)
        z = functions.forget(lambda x, y: (x + y + x,), x, y)
        testing.assert_allclose(x_data + y_data + x_data, z.data)

    def test_forward_cpu(self):
        self.check_forward(self.x, self.y)

    def check_backward(self, x_data, y_data, gz_data):
        def f(x, y):
            return functions.forget(lambda x, y: (x + y + x), x, y)

        gradient_check.check_backward(f, (x_data, y_data), gz_data)

    def test_backward_cpu(self):
        self.check_backward(self.x, self.y, self.gz)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.y),
                            cuda.to_gpu(self.gz))

    def check_double_backward(self, x_data, y_data, gz_data, ggx_data,
                              ggy_data):
        def f(x, y):
            return functions.forget(lambda x, y: (x * x * 3 + y * x,), x, y)

        gradient_check.check_double_backward(
            f, (x_data, y_data), gz_data, (ggx_data, ggy_data))

    def test_double_backward_cpu(self):
        self.check_double_backward(self.x, self.y, self.gz, self.ggx, self.ggy)

    @attr.gpu
    def test_double_backward_gpu(self):
        self.check_double_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.y),
                                   cuda.to_gpu(self.gz), cuda.to_gpu(self.ggx),
                                   cuda.to_gpu(self.ggy))


class TestForgetError(unittest.TestCase):

    def setUp(self):
        self.v = chainer.Variable(numpy.zeros(1))

    def test_not_callable(self):
        with self.assertRaises(TypeError):
            functions.forget(1)

    def test_invalid_type(self):
        with six.assertRaisesRegex(self, RuntimeError, 'int'):
            functions.forget(lambda: 1)

    def test_invalid_tuple_type_1st(self):
        with six.assertRaisesRegex(self, RuntimeError, '1st.*int'):
            functions.forget(lambda: (1,))

    def test_invalid_tuple_type_2nd(self):
        with six.assertRaisesRegex(self, RuntimeError, '2nd.*int'):
            functions.forget(lambda: (self.v, 1))

    def test_invalid_tuple_type_3rd(self):
        with six.assertRaisesRegex(self, RuntimeError, '3rd.*int'):
            functions.forget(lambda: (self.v, self.v, 1))

    def test_invalid_tuple_type_4th(self):
        with six.assertRaisesRegex(self, RuntimeError, '4th.*int'):
            functions.forget(lambda: (self.v,) * 3 + (1,))

    def test_invalid_tuple_type_11th(self):
        with six.assertRaisesRegex(self, RuntimeError, '11th.*int'):
            functions.forget(lambda: (self.v,) * 10 + (1,))

    def test_invalid_tuple_type_12th(self):
        with six.assertRaisesRegex(self, RuntimeError, '12th.*int'):
            functions.forget(lambda: (self.v,) * 11 + (1,))

    def test_invalid_tuple_type_13th(self):
        with six.assertRaisesRegex(self, RuntimeError, '13th.*int'):
            functions.forget(lambda: (self.v,) * 12 + (1,))


testing.run_module(__name__, __file__)
