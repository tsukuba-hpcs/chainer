from __future__ import print_function
import threading
import unittest

import mock
import numpy
import pytest
import six

import chainer
from chainer.backends import cuda
from chainer import testing
from chainer.testing import attr
from chainer.testing import backend
from chainer.utils import type_check


def make_array(start, shape, dtype):
    size = numpy.product(shape, dtype='i')
    a = numpy.arange(start, start + size)
    a = a.reshape(shape)
    a = a.astype(dtype, copy=False)
    return a


@testing.parameterize(*testing.product({
    'y_shape': [(4,), (0,), (2, 3), ()],
    'x_shape': [(3,), (0,), (4, 1), ()],
}))
class TestFunctionNode(unittest.TestCase):

    def _get_method(self, prefix, gpu):
        suffix = 'gpu' if gpu else 'cpu'
        return getattr(self.f, prefix + '_' + suffix)

    def setUp(self):
        y_shape = self.y_shape
        x_shape = self.x_shape
        y1 = make_array(1, y_shape, numpy.float32)
        y2 = make_array(2, y_shape, numpy.float32)
        gx1 = chainer.Variable(
            make_array(1, x_shape, numpy.float32))
        gx2 = None
        gy1 = make_array(1, y_shape, numpy.float32)
        gy2 = make_array(1, y_shape, numpy.float32)

        f = chainer.FunctionNode()
        f.check_type_forward = mock.MagicMock()
        f.forward_cpu = mock.MagicMock(return_value=(y1, y2))
        f.forward_gpu = mock.MagicMock()
        f.backward = mock.MagicMock(return_value=(gx1, gx2))
        self.f = f

        self.x1 = make_array(0, x_shape, numpy.float32)
        self.x2 = make_array(0, x_shape, numpy.int32)
        self.y1 = y1
        self.y2 = y2
        self.gx1 = gx1
        self.gx2 = gx2
        self.gx1_orig = chainer.Variable(
            make_array(3, x_shape, numpy.float32))
        self.gx2_orig = chainer.Variable(
            make_array(2, x_shape, numpy.float32))
        self.gx1_accum = gx1 + self.gx1_orig
        self.gy1 = gy1
        self.gy2 = gy2

    def tearDown(self):
        # Set None to delete cuda array
        self.f = None
        self.y1 = None
        self.y2 = None
        self.gx1 = None

    def setup_gpu(self):
        self.x1 = cuda.to_gpu(self.x1)
        self.x2 = cuda.to_gpu(self.x2)
        self.y1 = cuda.to_gpu(self.y1)
        self.y2 = cuda.to_gpu(self.y2)
        self.gx1.to_gpu()
        self.gx1_orig.to_gpu()
        self.gx2_orig.to_gpu()
        self.gx1_accum.to_gpu()
        self.gy1 = cuda.to_gpu(self.gy1)
        self.gy2 = cuda.to_gpu(self.gy2)
        self.f.forward_gpu = mock.MagicMock(return_value=(self.y1, self.y2))
        self.f.backward = mock.MagicMock(return_value=(self.gx1, self.gx2))

    def check_forward(self, gpu):
        y1, y2 = self.f.forward((self.x1, self.x2))
        self.assertEqual(self.f.check_type_forward.call_count, 0)
        self.assertEqual(self._get_method('forward', not gpu).call_count, 0)
        self._get_method('forward', gpu).assert_called_once_with(
            (self.x1, self.x2))
        self.assertTrue((cuda.to_cpu(y1) == cuda.to_cpu(self.y1)).all())
        self.assertTrue((cuda.to_cpu(y2) == cuda.to_cpu(self.y2)).all())

    def test_forward_cpu(self):
        self.check_forward(False)

    @attr.gpu
    def test_forward_gpu(self):
        self.setup_gpu()
        self.check_forward(True)

    def check_check_type_forward(self):
        self.assertEqual(self.f.check_type_forward.call_count, 1)
        ts = self.f.check_type_forward.call_args[0][0]
        self.assertIsInstance(ts, type_check.LightTypeInfoTuple)
        self.assertEqual(len(ts), 2)

        t1 = ts[0]
        assert t1.shape == self.x_shape
        assert t1.dtype == numpy.float32

        t2 = ts[1]
        assert t2.shape == self.x_shape
        assert t2.dtype == numpy.int32

    def check_apply(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        x1._node._rank = 1
        x2._node._rank = 3
        ys = self.f.apply((x1, x2))

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            # rank is (maximum rank in xs) + 1
            self.assertEqual(y.rank, 4)
            self.assertIs(y.creator_node, self.f)
            self.assertTrue(y.requires_grad)

        self.assertIsInstance(y.creator_node.outputs, tuple)

    def test_apply_cpu(self):
        self.check_apply()

    @attr.gpu
    def test_apply_gpu(self):
        self.setup_gpu()
        self.check_apply()

    def check_apply_all_ndarray(self):
        x1 = self.x1
        x2 = self.x2
        ys = self.f.apply((x1, x2))

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        xp = chainer.backend.get_array_module(x1)

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            self.assertIsInstance(y.data, xp.ndarray)
            self.assertFalse(y.requires_grad)

    def test_apply_all_ndarray_cpu(self):
        self.check_apply_all_ndarray()

    @attr.gpu
    def test_apply_all_ndarray_gpu(self):
        self.setup_gpu()
        self.check_apply_all_ndarray()

    def check_apply_ndarray(self):
        x1 = chainer.Variable(self.x1)
        x2 = self.x2
        x1._node._rank = 1
        ys = self.f.apply((x1, x2))

        self.assertEqual(len(ys), 2)
        self.check_check_type_forward()

        for y in ys:
            self.assertIsInstance(y, chainer.Variable)
            # rank is (maximum rank in xs) + 1
            self.assertEqual(y.rank, 2)
            self.assertIs(y.creator_node, self.f)
            self.assertTrue(y.requires_grad)

        self.assertIsInstance(y.creator_node.outputs, tuple)

    def test_apply_ndarray_cpu(self):
        self.check_apply_ndarray()

    @attr.gpu
    def test_apply_ndarray_gpu(self):
        self.setup_gpu()
        self.check_apply_ndarray()

    def check_apply_single_return_value(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        ret, = self.f.apply((x1, x2))
        self.assertIsInstance(ret, chainer.Variable)

    def test_apply_single_return_value_cpu(self):
        self.f.forward_cpu.return_value = (cuda.to_cpu(self.y1),)
        self.check_apply_single_return_value()

    @attr.gpu
    def test_apply_single_return_value_gpu(self):
        self.setup_gpu()
        self.f.forward_gpu.return_value = (cuda.to_gpu(self.y1),)
        self.check_apply_single_return_value()

    def _get_f(self):
        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)
        y1, y2 = self.f.apply((x1, x2))

        f = y1.creator_node
        # To test weak refernece, return only x1 and y1.
        # x2 and y2 are deleted by the garbage collector
        return f, x1, y1

    def test_unchain(self):
        f, _x1, _y1 = self._get_f()
        y1, y2 = f.outputs
        f.unchain()

        # As _y1 is alive, this weak ref is also alive
        y1_ref = y1()
        self.assertIsNotNone(y1_ref)
        self.assertIsNone(y1_ref.creator)
        # This weak ref is dead by unchain
        y2_ref = y2()
        self.assertIsNone(y2_ref)

        self.assertIsNone(f.inputs)

    def test_label(self):
        self.assertEqual(self.f.label, 'FunctionNode')


class TestFunctionNodeInvalidType(unittest.TestCase):

    def test_forward_invalid1(self):
        class FunctionNode(chainer.FunctionNode):

            def check_type_forward(self, in_types):
                x_type, = in_types
                type_check.expect(
                    x_type.dtype == numpy.float32,
                    x_type.ndim >= 2,
                )

            def forward(self, inputs):
                return inputs

        f = FunctionNode()

        # OK
        v = chainer.Variable(numpy.random.randn(1, 5).astype(numpy.float32))
        result, = f.apply((v,))
        assert isinstance(result, chainer.Variable)

        # Incorrect dtype
        # in py3, numpy dtypes are represented as class
        msg = """\
Invalid operation is performed in: FunctionNode \\(Forward\\)

Expect: in_types\\[0\\]\\.dtype == <(type|class) 'numpy\\.float32'>
Actual: float64 \\!= <(type|class) 'numpy\\.float32'>"""

        v = chainer.Variable(numpy.random.randn(1, 5))
        with six.assertRaisesRegex(self, chainer.utils.type_check.InvalidType,
                                   msg):
            f.apply((v,))

        # Incorrect dim
        msg = """\
Invalid operation is performed in: FunctionNode \\(Forward\\)

Expect: in_types\\[0\\]\\.ndim >= 2
Actual: 1 < 2"""

        v = chainer.Variable(numpy.random.randn(5).astype(numpy.float32))
        with six.assertRaisesRegex(self, chainer.utils.type_check.InvalidType,
                                   msg):
            f.apply((v,))


class TestFunctionNodeInconsistentBackends(unittest.TestCase):

    def setUp(self):
        self.x1 = numpy.random.rand(2, 3).astype(numpy.float32)
        self.x2 = numpy.random.rand(2, 3).astype(numpy.float32)

    @attr.gpu
    def test_inconsistent_inputs(self):
        class FunctionNode(chainer.FunctionNode):

            def forward(self, inputs):
                return inputs

        f = FunctionNode()

        # Cause inconsistency between inputs
        x1 = cuda.to_gpu(self.x1)

        x1 = chainer.Variable(x1)
        x2 = chainer.Variable(self.x2)

        with self.assertRaises(TypeError):
            f.apply((x1, x2))

    @attr.gpu
    def test_inconsistent_outputs(self):
        class FunctionNode(chainer.FunctionNode):

            def forward(self, inputs):
                # Cause inconsistency between outputs
                return inputs[0], cuda.to_gpu(inputs[1])

        f = FunctionNode()

        x1 = chainer.Variable(self.x1)
        x2 = chainer.Variable(self.x2)

        with self.assertRaises(TypeError):
            f.apply((x1, x2))


@testing.parameterize(
    {'return_value': (numpy.array([float('nan')], numpy.float32),),
     'valid': False},
    {'return_value': (numpy.array([1], numpy.int32),), 'valid': True},
)
class TestFunctionNodeForwardDebug(unittest.TestCase):

    def setUp(self):
        self.original_debug = chainer.is_debug()
        chainer.set_debug(True)
        self.one = numpy.array([1], numpy.float32)
        self.f = chainer.FunctionNode()

    def tearDown(self):
        chainer.set_debug(self.original_debug)

    def check_debug_forward(self, x_data):
        x = chainer.Variable(x_data)
        if self.valid:
            # check if forward throws nothing
            self.f.apply((x,))
        else:
            with self.assertRaises(RuntimeError):
                self.f.apply((x,))

    def test_debug_forward_cpu(self):
        self.f.forward_cpu = mock.MagicMock(return_value=self.return_value)
        self.check_debug_forward(self.one)

    @attr.gpu
    def test_debug_forward_gpu(self):
        return_value = tuple(None if x is None else cuda.to_gpu(x)
                             for x in self.return_value)
        self.f.forward_gpu = mock.MagicMock(return_value=return_value)
        self.check_debug_forward(cuda.to_gpu(self.one))


@backend.inject_backend_tests(
    None,
    testing.product({'use_cuda': [True, False]}))
class TestFunctionNodeInvalidBackwardChecks(unittest.TestCase):
    """Tests FunctionNode.backward correctness checks"""

    def setUp(self):
        self.f = chainer.FunctionNode()

    def _dummy_func(self, bwd_return_data):
        # Create a dummy func that returns `bwd_return_data` in the
        # `backward` method.

        def one(xp):
            return xp.array(1, numpy.float32)

        class DummyFunc(chainer.FunctionNode):
            def forward_cpu(self, inputs):
                return one(numpy),

            def forward_gpu(self, inputs):
                return one(cuda.cupy),

            def backward(self, indexes, grad_outputs):
                return bwd_return_data

        return DummyFunc()

    def check_debug_backward_accumulate(
            self, backend_config, f, xs_data, errors, initial_gxs=None):
        # `errors` is a dict, where keys are True or False indicating the
        # debug mode to run the test, and values are tuple of expected
        # exception type and error message pattern.

        for debug_mode, error in errors.items():

            def to_xp(arrs):
                if backend_config.use_cuda:
                    return cuda.to_gpu(arrs)
                else:
                    return arrs

            # Convert arrays to GPU
            xs_data = to_xp(xs_data)
            if initial_gxs is not None:
                initial_gxs = to_xp(initial_gxs)

            # Call forward
            xs = [chainer.Variable(x) for x in xs_data]
            y, = f.apply(xs)

            # Set initial input grads, if given
            if initial_gxs is not None:
                assert len(xs) == len(initial_gxs)
                for x, gx in zip(xs, initial_gxs):
                    x.grad = gx

            # Call backward & check error
            with chainer.using_config('debug', debug_mode):
                if error is None:
                    y.backward()  # no error should be raised
                else:
                    error_type, error_regex = error
                    with pytest.raises(error_type, match=error_regex):
                        y.backward()

    def test_ok(self, backend_config):
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func((
                chainer.Variable(numpy.array([2.0], numpy.float32)),)),
            xs_data=(numpy.array([1], numpy.float32),),
            errors={False: None, True: None})

    def test_gradients_has_nan(self, backend_config):
        # Returns a gradient that has NaN value
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func((chainer.Variable(numpy.array(
                [float('nan')], numpy.float32)),)),
            xs_data=(numpy.array([1], numpy.float32),),
            errors={True: (RuntimeError,
                           'NaN is detected on backward computation')})

    def test_invalid_number_of_gradients(self, backend_config):
        # Returns more gradients than expected
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func((
                chainer.Variable(numpy.array([2.0], numpy.float32)),
                chainer.Variable(numpy.array([1.0], numpy.float32)))),
            xs_data=(numpy.array([1], numpy.float32),),
            errors={True: (ValueError,
                           'number of gradients returned from backward is '
                           'incorrect')})

    def test_invalid_zero_gradients(self, backend_config):
        # Returns 0 gradients while 1 expected
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func(()),
            xs_data=(numpy.array([1], numpy.float32),),
            errors={True: (ValueError,
                           'number of gradients returned from backward is '
                           'incorrect')})

    def test_invalid_gradient_shape(self, backend_config):
        # Returns gradient of incorrect shape
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func((
                chainer.Variable(
                    backend_config.xp.array([2, 3], numpy.float32)),)),
            xs_data=(numpy.array([1], numpy.float32),),
            errors={True: (ValueError,
                           'shape of gradients returned from backward is '
                           'incorrect')})

    def test_invalid_gradient_type(self, backend_config):
        # Incorrectly returns a gradient as ndarray instead of variable
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func((
                backend_config.xp.array([2.0], numpy.float32))),
            xs_data=(numpy.array([1], numpy.float32),),
            errors={True: (ValueError,
                           'type of gradients returned from backward is '
                           'incorrect')})

    def test_invalid_gradient_dtype(self, backend_config):
        # Incorrectly returns a gradient with incorrect dtype, compared to
        # initially set gradients.
        self.check_debug_backward_accumulate(
            backend_config,
            f=self._dummy_func((
                chainer.Variable(
                    backend_config.xp.array([2.0], numpy.int64)),)),
            xs_data=(numpy.array([1], numpy.float32),),
            initial_gxs=(numpy.array([1], numpy.float32),),
            errors={True: (ValueError,
                           'dtype of gradients returned from backward is '
                           'incorrect')})


class TestNoBackpropMode(unittest.TestCase):

    def setUp(self):
        self.x = chainer.Variable(numpy.array([1.], 'f'))

    def test_no_backprop_mode(self):
        y = self.x + 1
        self.assertTrue(y.creator_node is not None)

        with chainer.no_backprop_mode():
            y = self.x + 1
        self.assertTrue(y.creator_node is None)

        y = self.x + 1
        self.assertTrue(y.creator_node is not None)

    def test_force_backprop_mode(self):
        with chainer.no_backprop_mode():
            with chainer.force_backprop_mode():
                y = self.x + 1
        self.assertTrue(y.creator_node is not None)

        y = self.x + 1
        self.assertTrue(y.creator_node is not None)

        with chainer.force_backprop_mode():
            y = self.x + 1
        self.assertTrue(y.creator_node is not None)


class MyThread(threading.Thread):

    def run(self):
        x = chainer.Variable(numpy.array([1], dtype='f'))
        with chainer.no_backprop_mode():
            y = x + 1
        self.creator_is_none = y.creator_node is None


class TestBackpropModeMultiThread(unittest.TestCase):

    def test_multi_thread(self):
        t = MyThread()
        t.start()
        t.join()
        self.assertTrue(t.creator_is_none)


class FunctionNodeWithRetaining(chainer.FunctionNode):

    def forward(self, inputs):
        self.retain_inputs([1])
        self.retain_outputs([1])
        return inputs

    def backward(self, _, grad_outputs):
        self.backward_inputs = self.get_retained_inputs()
        self.backward_outputs = self.get_retained_outputs()
        return grad_outputs


class TestFunctionNodeRetaining(unittest.TestCase):

    def setUp(self):
        inputs = [chainer.Variable(numpy.array([1], dtype=numpy.float32)),
                  chainer.Variable(numpy.array([1], dtype=numpy.float32))]
        self.input_data = [x.data for x in inputs]
        self.input_nodes = [x.node for x in inputs]

        self.f1 = FunctionNodeWithRetaining()
        outputs = self.f1.apply(inputs)
        outputs[0].grad = numpy.array([1], dtype=numpy.float32)
        outputs[0].backward()
        self.f1_output_data = [y.data for y in outputs]
        self.f1_output_nodes = [y.node for y in outputs]

        inputs = None  # release non-retained inputs

    def test_retain_inputs(self):
        self.assertEqual(len(self.f1.backward_inputs), 1)
        self.assertIs(self.f1.backward_inputs[0].node, self.input_nodes[1])
        numpy.testing.assert_array_equal(self.f1.backward_inputs[0].data,
                                         self.input_data[1])

    def test_retain_outputs_f1(self):
        self.assertEqual(len(self.f1.backward_outputs), 1)
        numpy.testing.assert_array_equal(self.f1.backward_outputs[0].data,
                                         self.f1_output_data[1])


def _get_value(x):
    if isinstance(x, chainer.Variable):
        return x.data
    return x


class TestGradTypeCheck(unittest.TestCase):

    def test_type_check(self):
        x = chainer.Variable(numpy.random.uniform(-1, 1, (2, 3)).astype('f'))
        y = x * x
        gx = chainer.Variable(numpy.random.uniform(-1, 1, (2, 3)).astype('f'))
        gy = chainer.Variable(numpy.random.uniform(-1, 1, (2, 3)).astype('f'))

        chainer.grad([y], [x], [gx], [gy])
        chainer.grad((y,), (x,), (gx,), (gy,))

        with self.assertRaises(TypeError):
            chainer.grad(y, [x], [gx], [gy])
        with self.assertRaises(TypeError):
            chainer.grad([y], x, [gx], [gy])
        with self.assertRaises(TypeError):
            chainer.grad([y], [x], gx, [gy])
        with self.assertRaises(TypeError):
            chainer.grad([y], [x], [gx], gy)


class GradTestBase(object):

    shape = 3,
    x_names = ()
    y_names = ()
    loss_scale = None
    extend_graph_x = False
    extend_graph_y = False

    def _init_attrs(self, names):
        ret = []
        for name in names:
            v = chainer.Variable(
                numpy.random.randint(-4, 6, self.shape).astype('f'), name=name)
            if self.extend_graph_x:
                v *= 1.
            ret.append(v)
            setattr(self, name, v)
        return ret

    def _init_ones(self, names):
        ret = []
        for name in names:
            v = chainer.Variable(numpy.ones(self.shape, dtype='f'))
            ret.append(v)
            setattr(self, name, v)
        return ret

    @staticmethod
    def _get_value(x):
        if isinstance(x, chainer.Variable):
            return x.data
        return x

    @staticmethod
    def _to_grad_names(names):
        return ['g%s' % name for name in names]

    def setUp(self):
        self.xs = self._init_attrs(self.x_names)
        self.gxs = self._init_attrs(self._to_grad_names(self.x_names))
        self.gys = self._init_attrs(self._to_grad_names(self.y_names))
        if self.loss_scale is not None:
            self._init_ones(self._to_grad_names(self.y_names))
            self.gys = None

    def use_gpu(self):
        for value in six.itervalues(self.__dict__):
            if isinstance(value, chainer.Variable):
                value.to_gpu()

    def forward(self):
        raise NotImplementedError

    def expected_grad(self):
        raise NotImplementedError

    def expected_double_grad(self):
        raise NotImplementedError

    def _print_variables(self, name, vs):
        print('{}: '.format(name), end='')
        print(*(self._get_value(v) for v in vs), sep=', ')

    def _print_inputs(self):
        self._print_variables('xs  ', self.xs)
        self._print_variables('gxs ', self.gxs)
        self._print_variables('gys ', self.gys)

    def check_grad(self):
        self.forward()
        ys = [getattr(self, name) for name in self.y_names]
        if self.extend_graph_y:
            self._ys = [v * 1. for v in ys]
        gxs = chainer.grad(ys, self.xs, self.gys, self.gxs,
                           loss_scale=self.loss_scale)

        expected = self.expected_grad()
        for i, gx in enumerate(self.gxs):
            expected[i] += gx

        self.assertEqual(len(gxs), len(expected))
        try:
            for a, e in zip(gxs, expected):
                testing.assert_allclose(self._get_value(a), self._get_value(e))
        except Exception:
            self._print_inputs()
            self._print_variables('gxs (actual)  ', gxs)
            self._print_variables('gxs (expected)', expected)
            raise

    def test_grad_cpu(self):
        self.check_grad()

    @attr.gpu
    def test_grad_gpu(self):
        self.use_gpu()
        self.check_grad()

    def check_double_grad(self):
        self.forward()
        ys = [getattr(self, name) for name in self.y_names]
        gxs = chainer.grad(ys, self.xs, self.gys, self.gxs,
                           enable_double_backprop=True,
                           loss_scale=self.loss_scale)
        y = sum(gxs)
        ggxs = chainer.grad([y], self.xs)

        expected = self.expected_double_grad()
        self.assertEqual(len(ggxs), len(expected))
        try:
            for a, e in zip(ggxs, expected):
                testing.assert_allclose(self._get_value(a), self._get_value(e))
        except Exception:
            self._print_inputs()
            self._print_variables('gxs            ', gxs)
            self._print_variables('ggxs (actual)  ', ggxs)
            self._print_variables('ggxs (expected)', expected)
            raise

    def test_double_grad_cpu(self):
        self.check_double_grad()

    @attr.gpu
    def test_double_grad_gpu(self):
        self.use_gpu()
        self.check_double_grad()


@testing.parameterize(*testing.product({
    'loss_scale': [None, 1, 10],
}))
class TestGradSimple(GradTestBase, unittest.TestCase):

    x_names = 'x',
    y_names = 'y',

    def forward(self):
        self.y = self.x * self.x

    def expected_grad(self):
        grad = 2 * self.x * self.gy
        if self.loss_scale is not None:
            grad *= self.loss_scale
        return [grad]

    def expected_double_grad(self):
        ggrad = 2 * self.gy
        if self.loss_scale is not None:
            ggrad *= self.loss_scale
        return [ggrad]


@testing.parameterize(*testing.product({
    'extend_graph_x': [False, True],
    'extend_graph_y': [False, True],
}))
class TestGradComplex(GradTestBase, unittest.TestCase):

    x_names = 'x1', 'x2'
    y_names = 'y1', 'y2'

    def forward(self):
        self.z = self.x1 * self.x1
        self.y1 = self.z + self.x1 * self.x2 + self.x2
        self.y2 = self.z + self.y1

    def expected_grad(self):
        dz_dx = 2 * self.x1
        dy1_dx = self.gy1 + self.gy2
        return [dy1_dx * (dz_dx + self.x2) + self.gy2 * dz_dx,
                dy1_dx * (self.x1 + 1)]

    def expected_double_grad(self):
        dy1_dx = self.gy1 + self.gy2
        return [3 * dy1_dx + 2 * self.gy2, dy1_dx]


class ExpPair(chainer.FunctionNode):

    def forward(self, inputs):
        x, = inputs
        xp = chainer.backend.get_array_module(x)
        self.retain_outputs((0, 1))
        return xp.exp(x), xp.exp(x)

    def backward(self, target_input_indexes, grad_outputs):
        return sum([
            g * exp
            for g, exp in zip(grad_outputs, self.get_retained_outputs())
            if g is not None
        ]),


def exp_pair(x):
    return ExpPair().apply((x,))


@testing.parameterize(*testing.product({
    'keep_y2': [False, True],
}))
class TestGradDelRetainedOutput(GradTestBase, unittest.TestCase):

    x_names = 'x1',
    y_names = 'y1',

    def forward(self):
        self.y1, y2 = exp_pair(self.x1)
        if self.keep_y2:
            self.y2 = y2

    def expected_grad(self):
        return [self.gy1 * self.y1]

    def expected_double_grad(self):
        return [self.gy1 * self.y1]


class ExpAndExpm1(chainer.FunctionNode):

    def forward(self, inputs):
        x, = inputs
        xp = chainer.backend.get_array_module()
        y0 = xp.exp(x)
        y1 = xp.expm1(x)
        self.retain_outputs((0,))
        return y0, y1

    def backward(self, target_input_indexes, grad_outputs):
        g0, g1 = grad_outputs
        y0, = self.get_retained_outputs()
        gx = []
        if g0 is not None:
            gx.append(g0 * y0)
        if g1 is not None:
            gx.append(g1 * y0)
        return chainer.functions.add(*gx),


def exp_and_expm1(x):
    return ExpAndExpm1().apply((x,))


class TestGradDelRetainedOutput2(unittest.TestCase):

    def test_retain_output(self):
        xp = numpy
        x_array = xp.random.randn(3)
        y1_grad = xp.random.randn(3)
        x_grad_grad = xp.random.randn(3)

        x = chainer.Variable(x_array, name='x')
        y0, y1 = exp_and_expm1(x)
        del y0

        # (x: Variable) requires grad
        # (y1_grad: ndarray) does not require grad
        gx, = chainer.grad([y1], [x], [y1_grad], enable_double_backprop=True)

        # assert gx == exp(x) * y1_grad
        xp.testing.assert_allclose(
            gx.array,
            xp.exp(x.array) * y1_grad)

        gx_, = chainer.grad([gx], [x], [x_grad_grad])
        xp.testing.assert_allclose(
            gx_.array,
            gx.array * x_grad_grad)


class TestGradV3Compat1(unittest.TestCase):

    def _var(self, val):
        return chainer.Variable(numpy.array(val, numpy.float32))

    def check(self, option, grads_before, grads_after):
        vs = []
        v = self._var(0.5)
        for _ in range(4):
            vs.append(v)
            v += v
            vs.append(v)
            v *= 1.
        _, x1, _, x2, _, y1, _, y2 = vs
        gx1 = self._var(1000.)
        gx2 = self._var(100.)
        gy1 = self._var(10.)
        gy2 = self._var(1.)
        for v, g in zip(vs, grads_before):
            if g is not None:
                v.grad_var = self._var(g)
        grads = chainer.grad(
            [y1, y2], [x1, x2], [gy1, gy2], [gx1, gx2], **option)
        numpy.testing.assert_allclose(grads[0].array, 1248.)
        numpy.testing.assert_allclose(grads[1].array, 124.)
        for v, ans in zip(vs, grads_after):
            if ans is None:
                self.assertIsNone(v.grad)
            else:
                numpy.testing.assert_allclose(v.grad, ans)

    def test_no_option(self):
        self.check({}, [None] * 8, [None] * 8)
        self.check({}, [-1.] * 8, [-1.] * 8)

    def test_set_grad(self):
        self.check(
            {'set_grad': True},
            [None] * 8,
            [None, 1248., None, 124., None, None, None, None])
        self.check(
            {'set_grad': True},
            [-1.] * 8,
            [-1., 1248., -1., 124., -1., -1., -1., -1.])

    def test_retain_grad(self):
        self.check(
            {'retain_grad': True},
            [None] * 8,
            [None, 1248., 248., 124., 24., 12., 2., 1.]
            # Before v5, the result was
            # [None, 1248., 248., 124., 24., 12., 2., None]
        )
        self.check(
            {'retain_grad': True},
            [-1.] * 8,
            [-1., 1248., 248., 124., 24., 12., 2., 1.]
            # Before v5, the result was
            # [-1., 1248., 248., 124., 24., 12., 2., -1.]
        )


testing.run_module(__name__, __file__)
