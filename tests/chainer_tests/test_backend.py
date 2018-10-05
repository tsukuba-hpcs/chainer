import unittest

import numpy

import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import testing
from chainer.testing import attr
import chainerx


class _TestCopyToBase(object):

    src_data = numpy.arange(1, 5, dtype=numpy.float32)
    dst_data = numpy.zeros_like(src_data)

    def _get_dst(self):
        raise NotImplementedError

    def test_from_cpu(self):
        src = self.src_data
        dst = self._get_dst()
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)

    @attr.gpu
    def test_from_gpu(self):
        src = cuda.cupy.array(self.src_data)
        dst = self._get_dst()
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)

    @attr.ideep
    def test_from_ideep(self):
        src = intel64.ideep.array(self.src_data)
        dst = self._get_dst()
        assert isinstance(src, intel64.mdarray)
        backend.copyto(dst, src)
        numpy.testing.assert_array_equal(cuda.to_cpu(dst), self.src_data)


class TestCopyToCPU(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        return self.dst_data


@attr.gpu
class TestCopyToGPU(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        return cuda.cupy.array(self.dst_data)

    @attr.multi_gpu(2)
    def test_gpu_to_another_gpu(self):
        src = cuda.cupy.array(self.src_data)
        with cuda.get_device_from_id(1):
            dst = self._get_dst()
        backend.copyto(dst, src)
        cuda.cupy.testing.assert_array_equal(dst, src)


@attr.ideep
class TestCopyToIDeep(_TestCopyToBase, unittest.TestCase):
    def _get_dst(self):
        dst = intel64.ideep.array(self.src_data)
        assert isinstance(dst, intel64.mdarray)
        return dst


class TestCopyToError(unittest.TestCase):
    def test_fail_on_invalid_src(self):
        src = None
        dst = numpy.zeros(1)
        with self.assertRaises(TypeError):
            backend.copyto(dst, src)

    def test_fail_on_invalid_dst(self):
        src = numpy.zeros(1)
        dst = None
        with self.assertRaises(TypeError):
            backend.copyto(dst, src)


class TestGetArrayModule(unittest.TestCase):

    def test_get_array_module_for_numpy_array(self):
        xp = backend.get_array_module(numpy.array([]))
        self.assertIs(xp, numpy)
        assert xp is not cuda.cupy
        assert xp is not chainerx

    def test_get_array_module_for_numpy_variable(self):
        xp = backend.get_array_module(chainer.Variable(numpy.array([])))
        assert xp is numpy
        assert xp is not cuda.cupy
        assert xp is not chainerx

    @attr.gpu
    def test_get_array_module_for_cupy_array(self):
        xp = backend.get_array_module(cuda.cupy.array([]))
        assert xp is cuda.cupy
        assert xp is not numpy
        assert xp is not chainerx

    @attr.gpu
    def test_get_array_module_for_cupy_variable(self):
        xp = backend.get_array_module(chainer.Variable(cuda.cupy.array([])))
        assert xp is cuda.cupy
        assert xp is not numpy
        assert xp is not chainerx

    @attr.chainerx
    def test_get_array_module_for_chainerx_array(self):
        xp = backend.get_array_module(chainerx.array([]))
        assert xp is chainerx
        assert xp is not numpy
        assert xp is not cuda.cupy

    @attr.chainerx
    def test_get_array_module_for_chainerx_variable(self):
        xp = backend.get_array_module(chainer.Variable(chainerx.array([])))
        assert xp is chainerx
        assert xp is not numpy
        assert xp is not cuda.cupy


class TestGetDeviceFromArray(unittest.TestCase):

    def test_numpy_int(self):
        device = chainer.backend.get_device_from_array(numpy.int64(0))
        assert device is cuda.DummyDevice

    def test_numpy_array(self):
        device = chainer.backend.get_device_from_array(numpy.array([0]))
        assert device is cuda.DummyDevice

    @attr.gpu
    def test_empty_cupy_array(self):
        arr = cuda.cupy.array([]).reshape((0, 10))
        device = chainer.backend.get_device_from_array(arr)
        assert device == cuda.Device(0)

    @attr.gpu
    def test_cupy_array(self):
        device = chainer.backend.get_device_from_array(cuda.cupy.array([0]))
        assert device == cuda.Device(0)

    @attr.chainerx
    def test_chainerx_cpu_array(self):
        arr = chainer.backend.to_chainerx(numpy.array([0]))
        device = chainer.backend.get_device_from_array(arr)
        assert isinstance(device, chainerx.DeviceScope)
        with device:
            chainerx.get_default_device() is arr.device
            assert chainerx.get_default_device().name == 'native:0'

    @attr.chainerx
    @attr.gpu
    def test_chainerx_gpu_array(self):
        arr = chainer.backend.to_chainerx(cuda.cupy.array([0]))
        device = chainer.backend.get_device_from_array(arr)
        assert isinstance(device, chainerx.DeviceScope)
        with device:
            assert chainerx.get_default_device() is arr.device
            assert chainerx.get_default_device().name == 'cuda:0'


class TestToDevice(unittest.TestCase):

    def orig_numpy(self):
        return numpy.ones((2, 3), numpy.float32)

    def orig_cupy(self):
        return cuda.to_gpu(numpy.ones((2, 3), numpy.float32))

    def orig_chainerx(self, device_name):
        return chainerx.ones((2, 3), numpy.float32, device=device_name)

    def to_device_check_equal(self, orig, device):
        converted = backend.to_device(orig, device)
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig),
            backend.to_numpy(converted))
        return converted

    def test_numpy_to_numpy(self):
        orig = self.orig_numpy()
        converted = self.to_device_check_equal(orig, cuda.DummyDevice)
        assert converted is orig

    @attr.gpu
    def test_numpy_to_cupy(self):
        orig = self.orig_numpy()
        converted = self.to_device_check_equal(orig, cuda.Device(0))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == cuda.Device(0)

    @attr.chainerx
    def test_numpy_to_chainerx_native(self):
        orig = self.orig_numpy()
        device = chainerx.get_device('native:0')
        converted = self.to_device_check_equal(orig, device)
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device is device

        # memory must be shared
        orig[:] *= 2
        numpy.testing.assert_array_equal(
            orig, backend.to_numpy(converted))

    @attr.chainerx
    @attr.gpu
    def test_numpy_to_chainerx_cuda(self):
        orig = self.orig_numpy()
        device = chainerx.get_device('cuda:0')
        converted = self.to_device_check_equal(orig, device)
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device is device

    @attr.gpu
    def test_cupy_to_cupy(self):
        orig = self.orig_cupy()
        converted = self.to_device_check_equal(orig, cuda.Device(0))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == cuda.Device(0)

    @attr.multi_gpu(2)
    def test_cupy_to_cupy_multigpu(self):
        orig = self.orig_cupy()
        converted = self.to_device_check_equal(orig, cuda.Device(1))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == cuda.Device(1)

    @attr.gpu
    def test_cupy_to_numpy(self):
        orig = self.orig_cupy()
        converted = self.to_device_check_equal(orig, cuda.DummyDevice)
        assert isinstance(converted, numpy.ndarray)

    @attr.chainerx
    @attr.gpu
    def test_cupy_to_chainerx_native(self):
        orig = self.orig_cupy()
        device = chainerx.get_device('native:0')
        converted = self.to_device_check_equal(orig, device)
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device is device

    @attr.chainerx
    @attr.gpu
    def test_cupy_to_chainerx_cuda(self):
        orig = self.orig_cupy()
        device = chainerx.get_device('cuda:0')
        converted = self.to_device_check_equal(orig, device)
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device is device

        # memory must be shared
        orig[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.chainerx
    @attr.multi_gpu(2)
    def test_cupy_to_chainerx_cuda_multigpu(self):
        orig = self.orig_cupy()
        device = chainerx.get_device('cuda:1')
        converted = self.to_device_check_equal(orig, device)
        assert isinstance(converted, chainerx.ndarray)
        assert converted.device is device

        # memory must not be shared
        orig_copy = orig.copy()
        orig[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig_copy), backend.to_numpy(converted))

    @attr.chainerx
    def test_chainerx_native_to_numpy(self):
        orig = self.orig_chainerx('native:0')
        converted = self.to_device_check_equal(orig, cuda.DummyDevice)
        assert isinstance(converted, numpy.ndarray)

        # memory must be shared
        converted[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.chainerx
    @attr.gpu
    def test_chainerx_cuda_to_cupy(self):
        orig = self.orig_chainerx('cuda:0')
        converted = self.to_device_check_equal(orig, cuda.Device(0))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == cuda.Device(0)

        # memory must be shared
        converted[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted))

    @attr.chainerx
    @attr.multi_gpu(2)
    def test_chainerx_cuda_to_cupy_multigpu(self):
        orig = self.orig_chainerx('cuda:0')
        converted = self.to_device_check_equal(orig, cuda.Device(1))
        assert isinstance(converted, cuda.ndarray)
        assert converted.device == cuda.Device(1)

        # memory must not be shared
        converted_copy = converted.copy()
        with cuda.Device(1):
            converted[:] *= 2
        numpy.testing.assert_array_equal(
            backend.to_numpy(orig), backend.to_numpy(converted_copy))

    @attr.chainerx
    @attr.gpu
    def test_chainerx_cuda_to_numpy(self):
        orig = self.orig_chainerx('cuda:0')
        converted = self.to_device_check_equal(orig, cuda.DummyDevice)
        assert isinstance(converted, numpy.ndarray)


testing.run_module(__name__, __file__)
