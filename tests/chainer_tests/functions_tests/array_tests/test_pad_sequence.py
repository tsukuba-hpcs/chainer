import unittest

import numpy
import six

from chainer import cuda
from chainer import functions
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr


@testing.parameterize(*testing.product({
    'lengths': [[2, 1, 5, 3], [2]],
    'length': [None, 6],
    'shape': [(3, 4), ()],
    'pad': [0, -1],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestPadSequence(unittest.TestCase):

    def setUp(self):
        self.xs = [
            numpy.random.uniform(-1, 1, (l,) + self.shape).astype(self.dtype)
            for l in self.lengths]
        if self.length:
            max_length = self.length
        else:
            max_length = max(self.lengths)
        self.y_shape = (len(self.lengths), max_length,) + self.shape
        self.g = numpy.random.uniform(-1, 1, self.y_shape).astype(self.dtype)

    def check_forward(self, xs):
        y = functions.pad_sequence(xs, length=self.length, padding=self.pad)

        self.assertEqual(y.shape, self.y_shape)
        for i, (length, x) in enumerate(six.moves.zip(self.lengths, self.xs)):
            testing.assert_allclose(y.data[i, 0:length], x)
            testing.assert_allclose(y.data[i, length:], self.pad)

    def test_forward_cpu(self):
        self.check_forward(self.xs)

    @attr.gpu
    def test_forward_gpu(self):
        self.check_forward([cuda.to_gpu(x) for x in self.xs])

    def check_backward(self, xs, g):
        gradient_check.check_backward(
            functions.PadSequence(self.length, self.pad), xs, g,
            dtype=numpy.float64)

    def test_backward_cpu(self):
        self.check_backward(self.xs, self.g)

    @attr.gpu
    def test_backward_gpu(self):
        self.check_backward(
            [cuda.to_gpu(x) for x in self.xs], cuda.to_gpu(self.g))


testing.run_module(__name__, __file__)
