import operator
import sys
import unittest

import numpy

from cupy import testing


@testing.parameterize(
    *testing.product({
        'shape_pair': [
            ((3, 2), (2, 4)),
            ((2,), (2, 4)),
            ((3, 2), (2,)),
            ((2,), (2,)),
            ((5, 3, 2), (5, 2, 4)),
            ((5, 3, 2), (2, 4)),
            ((3, 2), (5, 2, 4)),
            ((5, 3, 2), (1, 2, 4)),
            ((1, 3, 2), (5, 2, 4)),
            ((5, 3, 2), (2,)),
            ((2,), (5, 2, 4)),
            ((6, 5, 3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (6, 1, 2, 4)),
            ((6, 5, 3, 2), (1, 5, 2, 4)),
            ((6, 5, 3, 2), (1, 1, 2, 4)),
            ((6, 1, 3, 2), (6, 5, 2, 4)),
            ((1, 5, 3, 2), (6, 5, 2, 4)),
            ((1, 1, 3, 2), (6, 5, 2, 4)),
            ((3, 2), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2, 4)),
            ((2,), (6, 5, 2, 4)),
            ((6, 5, 3, 2), (2,)),
        ],
    }))
@testing.gpu
class TestMatmul(unittest.TestCase):

    # _multiprocess_can_split_ = True

    @unittest.skipUnless(sys.version_info >= (3, 5),
                         'Only for Python3.5 or higher')
    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1', no_float16=True)
    @testing.for_all_dtypes(name='dtype2', no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_operator_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return operator.matmul(x1, x2)

    # Since calculation accuracy is bad for (float16, uint8).
    @unittest.skipUnless(sys.version_info >= (3, 5),
                         'Only for Python3.5 or higher')
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_operator_matmul_float16(self, xp):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, numpy.float16)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, numpy.float16)
        return operator.matmul(x1, x2)

    @testing.with_requires('numpy>=1.10')
    @testing.for_all_dtypes(name='dtype1', no_float16=True)
    @testing.for_all_dtypes(name='dtype2', no_float16=True)
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)  # required for uint8
    def test_cupy_matmul(self, xp, dtype1, dtype2):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, dtype1)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, dtype2)
        return xp.matmul(x1, x2)

    # Since calculation accuracy is bad for (float16, uint8).
    @testing.with_requires('numpy>=1.10')
    @testing.numpy_cupy_allclose(rtol=1e-3, atol=1e-3)
    def test_cupy_matmul_float16(self, xp):
        x1 = testing.shaped_arange(self.shape_pair[0], xp, numpy.float16)
        x2 = testing.shaped_arange(self.shape_pair[1], xp, numpy.float16)
        return xp.matmul(x1, x2)
