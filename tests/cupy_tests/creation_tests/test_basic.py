import unittest

import numpy

import cupy
from cupy import testing


@testing.gpu
class TestBasic(unittest.TestCase):

    _multiprocess_can_split_ = True

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty(self, xp, dtype):
        a = xp.empty((2, 3, 4), dtype=dtype)
        a.fill(0)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_scalar(self, xp, dtype):
        a = xp.empty(None, dtype=dtype)
        a.fill(0)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_int(self, xp, dtype):
        a = xp.empty(3, dtype=dtype)
        a.fill(0)
        return a

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_empty_like(self, xp, dtype):
        a = testing.shaped_arange((2, 3, 4), xp, dtype)
        b = xp.empty_like(a)
        b.fill(0)
        return b

    def test_empty_zero_sized_array_strides(self):
        a = numpy.empty((1, 0, 2), dtype='d')
        b = cupy.empty((1, 0, 2), dtype='d')
        self.assertEqual(b.strides, a.strides)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_eye(self, xp, dtype):
        return xp.eye(5, 4, 1, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_identity(self, xp, dtype):
        return xp.identity(4, dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros(self, xp, dtype):
        return xp.zeros((2, 3, 4), dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_scalar(self, xp, dtype):
        return xp.zeros(None, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_int(self, xp, dtype):
        return xp.zeros(3, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_zeros_like(self, xp, dtype):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.zeros_like(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones(self, xp, dtype):
        return xp.ones((2, 3, 4), dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_ones_like(self, xp, dtype):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.ones_like(a)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full(self, xp, dtype):
        return xp.full((2, 3, 4), 1, dtype=dtype)

    @testing.for_all_dtypes()
    @testing.numpy_cupy_array_equal()
    def test_full_like(self, xp, dtype):
        a = xp.ndarray((2, 3, 4), dtype=dtype)
        return xp.full_like(a, 1)
