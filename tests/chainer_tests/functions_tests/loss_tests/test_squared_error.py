import unittest

import numpy

import chainer
from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(
    {'shape': (4, 3)},
    {'shape': (4, 3, 2)},
    {'shape': (4,)},
    {'shape': ()},
    {'shape': (1,)},
    {'shape': (1, 1)},
)
class TestSquaredError(unittest.TestCase):

    def setUp(self):
        self.x0 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.x1 = numpy.random.uniform(-1, 1, self.shape).astype(numpy.float32)
        self.gy = numpy.random.random(self.shape).astype(numpy.float32)

        self.check_backward_options = {}

    def check_forward(self, x0_data, x1_data):
        x0 = chainer.Variable(x0_data)
        x1 = chainer.Variable(x1_data)
        loss = functions.squared_error(x0, x1)
        loss_value = cuda.to_cpu(loss.data)
        self.assertEqual(loss_value.dtype, numpy.float32)
        self.assertEqual(loss_value.shape, x0_data.shape)

        for i in numpy.ndindex(self.x0.shape):
            # Compute expected value
            loss_expect = (self.x0[i] - self.x1[i]) ** 2
            self.assertAlmostEqual(loss_value[i], loss_expect, places=5)

    def test_forward_cpu(self):
        self.check_forward(self.x0, self.x1)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x0), cuda.to_gpu(self.x1))

    def check_backward(self, x0_data, x1_data, y_grad):
        gradient_check.check_backward(
            functions.SquaredError(),
            (x0_data, x1_data), y_grad, eps=1e-2,
            dtype=numpy.float64,
            **self.check_backward_options)

    def test_backward_cpu(self):
        self.check_backward(self.x0, self.x1, self.gy)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            cuda.to_gpu(self.x0), cuda.to_gpu(self.x1), cuda.to_gpu(self.gy))


testing.run_module(__name__, __file__)
