import unittest

import numpy
import six

import chainer
from chainer import cuda
from chainer import function_hooks
from chainer import functions
from chainer.functions.connection import linear
from chainer import links
from chainer import testing
from chainer.testing import attr


def check_history(self, t, function_type, used_bytes_type,
                  acquired_bytes_type):
    self.assertIsInstance(t[0], function_type)
    self.assertIsInstance(t[1], used_bytes_type)
    self.assertIsInstance(t[2], acquired_bytes_type)


@attr.gpu
class TestCupyMemoryProfileHookToLink(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CupyMemoryProfileHook()
        self.l = links.Linear(5, 5)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def test_name(self):
        self.assertEqual(self.h.name, 'CupyMemoryProfileHook')

    def check_forward(self, x):
        with self.h:
            self.l(chainer.Variable(x))
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      linear.LinearFunction, int, int)

    def test_forward_gpu(self):
        self.l.to_gpu()
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.l(x)
        y.grad = gy
        with self.h:
            y.backward()
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      linear.LinearFunction, int, int)

    def test_backward_gpu(self):
        self.l.to_gpu()
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))


@attr.gpu
class TestCupyMemoryProfileHookToFunction(unittest.TestCase):

    def setUp(self):
        self.h = function_hooks.CupyMemoryProfileHook()
        self.f = functions.Exp()
        self.f.add_hook(self.h)
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        self.gy = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)

    def tearDown(self):
        self.f.delete_hook(self.h.name)

    def check_forward(self, x):
        self.f(chainer.Variable(x))
        self.assertEqual(1, len(self.h.call_history))
        check_history(self, self.h.call_history[0],
                      functions.Exp, int, int)

    def test_forward_gpu(self):
        self.check_forward(cuda.to_gpu(self.x))

    def check_backward(self, x, gy):
        x = chainer.Variable(x)
        y = self.f(x)
        y.grad = gy
        y.backward()
        self.assertEqual(2, len(self.h.call_history))
        check_history(self, self.h.call_history[1],
                      functions.Exp, int, int)

    def test_backward_gpu(self):
        self.check_backward(cuda.to_gpu(self.x), cuda.to_gpu(self.gy))

    def test_reentrant(self):
        # In/grad data are random; these do not simulate the actually possible
        # cases.
        f = self.f
        g = functions.Identity()  # any function other than f: Exp is ok

        self.h.backward_preprocess(f, (self.x,), (self.gy,))
        self.h.forward_preprocess(g, (self.x,))
        self.h._memory_hook.used_bytes += 512
        self.h._memory_hook.acquired_bytes += 512
        self.h.forward_postprocess(g, (self.x,))
        self.h._memory_hook.used_bytes += 512
        self.h._memory_hook.acquired_bytes += 512
        self.h.backward_postprocess(f, (self.x,), (self.gy,))

        history = {f: (u, a, d) for (f, u, a, d) in self.h.call_history}
        self.assertEqual(len(history), 2)
        self.assertIn(f, history)
        self.assertIn(g, history)
        f_used_bytes, f_acquired_bytes, f_depth = history[f]
        g_used_bytes, g_acquired_bytes, g_depth = history[g]
        self.assertEqual(f_depth, 0)
        self.assertEqual(g_depth, 1)
        self.assertGreater(f_used_bytes, g_used_bytes)
        self.assertGreater(f_acquired_bytes, g_acquired_bytes)

    def test_reentrant_total_bytes(self):
        f = self.f
        g = functions.Identity()

        self.h.backward_preprocess(f, (self.x,), (self.gy,))
        self.h.forward_preprocess(g, (self.x,))
        self.h._memory_hook.used_bytes += 512
        self.h._memory_hook.acquired_bytes += 512
        self.h.forward_postprocess(g, (self.x,))
        self.h._memory_hook.used_bytes += 512
        self.h._memory_hook.acquired_bytes += 512
        self.h.backward_postprocess(f, (self.x,), (self.gy,))

        self.assertEqual(self.h.total_used_bytes(), 1024)
        self.assertEqual(self.h.total_acquired_bytes(), 1024)


@attr.gpu
class TestCupyMemoryProfileReport(unittest.TestCase):

    def setUp(self):
        cuda.memory_pool.free_all_blocks()
        self.h = function_hooks.CupyMemoryProfileHook()
        self.f1 = functions.Exp()
        self.f2 = functions.Log()
        self.x = numpy.random.uniform(-0.1, 0.1, (3, 5)).astype(numpy.float32)
        x = cuda.to_gpu(self.x)
        with self.h:
            self.f1(chainer.Variable(x))
            self.f1(chainer.Variable(x))
            self.f2(chainer.Variable(x))
            self.f2(chainer.Variable(x))

    def test_call_history(self):
        self.assertEqual(4, len(self.h.call_history))

    def test_total_used_bytes(self):
        self.assertNotEqual(0, self.h.total_used_bytes())

    def test_total_acquired_bytes(self):
        self.assertNotEqual(0, self.h.total_acquired_bytes())

    def test_summary(self):
        self.assertEqual(2, len(self.h.summary()))

    def test_print_report(self):
        io = six.StringIO()
        self.h.print_report(file=io)
        expect = r'''\AFunctionName  UsedBytes  AcquiredBytes  Occurrence
 +Exp +[0-9.\-e]+.?B +[0-9.\-e]+.?B +[0-9]+
 +Log +[0-9.\-e]+.?B +[0-9.\-e]+.?B +[0-9]+$
'''
        actual = io.getvalue()
        six.assertRegex(self, actual, expect)


testing.run_module(__name__, __file__)
