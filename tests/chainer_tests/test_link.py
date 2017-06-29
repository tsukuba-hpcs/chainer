import unittest

import mock
import numpy

import chainer
from chainer import cuda
from chainer import initializers
from chainer import testing
from chainer.testing import attr


class TestLink(unittest.TestCase):

    def setUp(self):
        x_shape_0 = 2
        x_shape_1 = numpy.int64(3)
        self.link = chainer.Link(x=((x_shape_0, x_shape_1), 'd'),
                                 u=(None, 'd'))
        with self.link.init_scope():
            self.link.y = chainer.Parameter(shape=(2,))
            self.link.v = chainer.Parameter()
        self.p = numpy.array([1, 2, 3], dtype='f')
        self.link.add_persistent('p', self.p)
        self.link.name = 'a'
        self.link.x.update_rule = chainer.UpdateRule()
        self.link.x.update_rule.enabled = False
        self.link.u.update_rule = chainer.UpdateRule()
        if cuda.available:
            self.current_device_id = cuda.cupy.cuda.get_device_id()

    def tearDown(self):
        if cuda.available \
                and cuda.cupy.cuda.get_device_id() != self.current_device_id:
            cuda.Device(self.current_device_id).use()

    def check_param_init(self, name, shape, dtype, data_value=numpy.nan):
        self.assertTrue(hasattr(self.link, name))
        var = getattr(self.link, name)
        self.assertEqual(var.name, name)
        self.assertIsInstance(var, chainer.Parameter)
        self.assertEqual(var.data.shape, shape)
        self.assertEqual(var.data.dtype, dtype)
        numpy.testing.assert_array_equal(var.data, data_value)
        self.assertEqual(var.grad.shape, shape)
        self.assertEqual(var.grad.dtype, dtype)
        numpy.testing.assert_array_equal(var.grad, numpy.nan)

    def check_param_uninit(self, name, initializer=None):
        self.assertTrue(hasattr(self.link, name))
        var = getattr(self.link, name)
        self.assertIsInstance(var, chainer.Parameter)
        self.assertEqual(var.name, name)
        self.assertIsNone(var.data)
        if initializer is not None:
            self.assertIs(var.initializer, initializer)

    def test_init(self):
        self.check_param_init('x', (2, 3), 'd')
        self.check_param_init('y', (2,), 'f')
        self.check_param_uninit('u')
        self.link.u.initialize((2, 3))
        self.check_param_init('u', (2, 3), 'd')
        self.check_param_uninit('v')
        self.link.v.initialize((2, 3))
        self.check_param_init('v', (2, 3), 'f')

    def test_assign_param_outside_of_init_scope(self):
        p = chainer.Parameter()
        self.link.p = p
        self.assertTrue(all(p is not param for param in self.link.params()))

    def test_assign_var_in_init_scope(self):
        p = chainer.Variable()
        with self.link.init_scope():
            self.link.p = p
        self.assertTrue(all(p is not param for param in self.link.params()))

    def test_add_param(self):
        self.link.add_param('z', (2, 3))
        self.check_param_init('z', (2, 3), 'f')

        self.link.add_param('w', (2, 3), dtype='d')
        self.check_param_init('w', (2, 3), 'd')

        self.link.add_param('r')
        self.check_param_uninit('r')
        self.link.r.initialize((2, 3))
        self.check_param_init('r', (2, 3), 'f')

        self.link.add_param('s', dtype='d')
        self.check_param_uninit('s')
        self.link.s.initialize((2, 3))
        self.check_param_init('s', (2, 3), 'd')

        initializer = initializers.Zero('d')
        self.link.add_param('t', initializer=initializer)
        self.check_param_uninit('t', initializer)
        self.link.t.initialize((2, 3))
        self.check_param_init('t', (2, 3), 'd', 0)

    def test_add_param_direct_initialization(self):
        z = numpy.random.rand(2, 3).astype('f')
        self.link.add_param('z', initializer=z)
        self.assertIsInstance(self.link.z.data, numpy.ndarray)
        numpy.testing.assert_array_equal(self.link.z.data, z)

    def test_add_param_duplicated_with_persistent(self):
        self.link.add_persistent('z', 'abc')
        with self.assertRaises(AttributeError):
            self.link.add_param('z', (2, 3))

    def test_add_persistent(self):
        self.assertTrue(hasattr(self.link, 'p'))
        self.assertIs(self.link.p, self.p)

        self.link.add_persistent('q', 'abc')
        self.assertTrue(hasattr(self.link, 'q'))
        self.assertEqual(self.link.q, 'abc')

    def test_copy(self):
        link = self.link.copy()
        self.assertIsInstance(link._params, set)
        self.assertIsInstance(link._persistent, set)
        self.assertTrue(hasattr(link, 'x'))
        self.assertTrue(hasattr(link, 'y'))
        self.assertTrue(hasattr(link, 'u'))
        self.assertTrue(hasattr(link, 'p'))
        self.assertIsNot(link.x, self.link.x)
        self.assertIs(link.x.data, self.link.x.data)
        self.assertIsNot(link.y, self.link.y)
        self.assertIs(link.y.data, self.link.y.data)
        self.assertIsNone(link.u.data)
        self.assertIs(link.p, self.link.p)
        self.assertIs(link.name, None)

    def test_to_cpu_on_cpu(self):
        x = self.link.x.data
        gx = self.link.x.grad
        y = self.link.y.data
        gy = self.link.y.grad
        p = self.link.p
        self.link.to_cpu()
        self.assertIs(self.link.x.data, x)
        self.assertIs(self.link.x.grad, gx)
        self.assertIs(self.link.y.data, y)
        self.assertIs(self.link.y.grad, gy)
        self.assertIsNone(self.link.u.data)
        self.assertIsNone(self.link.u.grad)
        self.assertIs(self.link.p, p)

    @attr.gpu
    def test_to_cpu(self):
        self.link.to_gpu()
        self.link.to_cpu()
        self.link.v.initialize((2, 3))
        self.assertIs(self.link.xp, numpy)
        self.assertIsInstance(self.link.x.data, numpy.ndarray)
        self.assertIsInstance(self.link.x.grad, numpy.ndarray)
        self.assertIsInstance(self.link.y.data, numpy.ndarray)
        self.assertIsInstance(self.link.y.grad, numpy.ndarray)
        self.assertIsNone(self.link.u.data)
        self.assertIsNone(self.link.u.grad)
        self.assertIsInstance(self.link.v.data, numpy.ndarray)
        self.assertIsInstance(self.link.v.grad, numpy.ndarray)
        self.assertIsInstance(self.link.p, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.link.to_gpu()
        self.link.v.initialize((2, 3))
        self.assertIs(self.link.xp, cupy)
        self.assertIsInstance(self.link.x.data, cupy.ndarray)
        self.assertIsInstance(self.link.x.grad, cupy.ndarray)
        self.assertIsInstance(self.link.y.data, cupy.ndarray)
        self.assertIsInstance(self.link.y.grad, cupy.ndarray)
        self.assertIsNone(self.link.u.data)
        self.assertIsNone(self.link.u.grad)
        self.assertIsInstance(self.link.v.data, cupy.ndarray)
        self.assertIsInstance(self.link.v.grad, cupy.ndarray)
        self.assertIsInstance(self.link.p, cupy.ndarray)

    @attr.multi_gpu(2)
    def test_to_gpu_different_device(self):
        cuda.Device(1).use()
        self.link.to_gpu(0)
        self.assertEqual(self.link._device_id, 0)

    @attr.multi_gpu(2)
    def test_to_gpu_current_device(self):
        cuda.Device(1).use()
        self.link.to_gpu()
        self.assertEqual(self.link._device_id, 1)

    def test_params(self):
        params = list(self.link.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.link.x), id(self.link.y),
                          id(self.link.u), id(self.link.v)})

    def test_params_skip_uninit(self):
        params = list(self.link.params(include_uninit=False))
        self.assertEqual({id(p) for p in params},
                         {id(self.link.x), id(self.link.y)})

    def test_namedparams(self):
        namedparams = list(self.link.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/x', id(self.link.x)), ('/y', id(self.link.y)),
                          ('/u', id(self.link.u)), ('/v', id(self.link.v))})

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.link.namedparams(include_uninit=False))
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/x', id(self.link.x)), ('/y', id(self.link.y))})

    def test_links(self):
        links = list(self.link.links())
        self.assertIs(links[0], self.link)

    def test_links_skipself(self):
        links = list(self.link.links(skipself=True))
        self.assertFalse(links)  # empty

    def test_namedlinks(self):
        pl = list(self.link.namedlinks())
        self.assertEqual(len(pl), 1)
        self.assertEqual(pl[0][0], '/')
        self.assertIs(pl[0][1], self.link)

    def test_copyparams(self):
        self.link.x.grad.fill(0)
        self.link.y.grad.fill(1)
        self.link.u.initialize((2, 3))
        self.link.u.data.fill(0)
        self.link.u.grad.fill(1)
        self.link.v.zerograd()
        gx = self.link.x.grad.copy()
        gy = self.link.y.grad.copy()
        gu = self.link.u.grad.copy()

        l = chainer.Link(x=(2, 3), y=2, u=(2, 3), v=(3, 2))
        l.x.data.fill(2)
        l.x.grad.fill(3)
        l.y.data.fill(4)
        l.y.grad.fill(5)
        l.u.data.fill(6)
        l.u.grad.fill(7)
        l.v.data.fill(8)
        l.v.grad.fill(9)

        self.link.copyparams(l)
        numpy.testing.assert_array_equal(self.link.x.data, l.x.data)
        numpy.testing.assert_array_equal(self.link.x.grad, gx)
        numpy.testing.assert_array_equal(self.link.y.data, l.y.data)
        numpy.testing.assert_array_equal(self.link.y.grad, gy)
        numpy.testing.assert_array_equal(self.link.u.data, l.u.data)
        numpy.testing.assert_array_equal(self.link.u.grad, gu)
        numpy.testing.assert_array_equal(self.link.v.data, l.v.data)
        numpy.testing.assert_array_equal(self.link.v.grad, 0)

    def test_cleargrads(self):
        self.link.cleargrads()
        self.assertIsNone(self.link.x.grad)
        self.assertIsNone(self.link.y.grad)
        self.link.u.initialize((2, 3))
        self.link.v.initialize((2, 3))
        self.assertIsNone(self.link.u.grad)
        self.assertIsNone(self.link.v.grad)

    def test_zerograds(self):
        gx_expect = numpy.zeros_like(self.link.x.data)
        gy_expect = numpy.zeros_like(self.link.y.data)
        self.link.zerograds()
        numpy.testing.assert_array_equal(self.link.x.grad, gx_expect)
        numpy.testing.assert_array_equal(self.link.y.grad, gy_expect)
        self.link.u.initialize((2, 3))
        self.link.v.initialize((2, 3))
        gu_expect = numpy.zeros_like(self.link.u.data)
        gv_expect = numpy.zeros_like(self.link.v.data)
        numpy.testing.assert_array_equal(self.link.u.grad, gu_expect)
        numpy.testing.assert_array_equal(self.link.v.grad, gv_expect)

    def test_addgrads(self):
        l = chainer.Link(x=(2, 3), y=2, u=(2, 3), v=None)
        l.x.grad.fill(1)
        l.y.grad.fill(2)
        l.u.grad.fill(3)

        self.link.x.grad.fill(-1)
        self.link.y.grad.fill(-2)
        self.link.u.cleargrad()

        self.link.addgrads(l)

        gx_expect = numpy.zeros_like(l.x.grad)
        gy_expect = numpy.zeros_like(l.y.grad)
        gu_expect = l.u.grad
        numpy.testing.assert_array_equal(self.link.x.grad, gx_expect)
        numpy.testing.assert_array_equal(self.link.y.grad, gy_expect)
        numpy.testing.assert_array_equal(self.link.u.grad, gu_expect)
        self.assertIsNone(self.link.v.grad, None)

    def test_serialize(self):
        serializer = mock.MagicMock(return_value=3)
        l = chainer.Link(x=(2, 3), y=2)
        l.add_persistent('z', 1)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 3)
        serializer.assert_any_call('x', l.x.data)
        serializer.assert_any_call('y', l.y.data)
        serializer.assert_any_call('z', 1)
        self.assertEqual(l.z, 3)

    def test_serialize_param_shape_placeholder(self):
        serializer = mock.MagicMock(return_value=3)
        l = chainer.Link(y=2, x=None)
        l.x.initialize((2, 3))
        l.add_persistent('z', 1)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 3)
        serializer.assert_any_call('x', l.x.data)
        serializer.assert_any_call('y', l.y.data)
        serializer.assert_any_call('z', 1)
        self.assertEqual(l.z, 3)

    def test_serialize_deserialize_to_uninitialized_param(self):
        ret = numpy.random.rand(2, 3).astype('f')
        serializer = mock.MagicMock(return_value=ret)
        l = chainer.Link(x=None)
        l.serialize(serializer)
        self.assertEqual(serializer.call_count, 1)
        serializer.assert_any_call('x', None)
        self.assertIsInstance(l.x.data, numpy.ndarray)
        numpy.testing.assert_array_equal(l.x.data, ret)

    def test_enable_update(self):
        self.link.enable_update()
        self.assertTrue(self.link.x.update_rule.enabled)
        self.assertTrue(self.link.u.update_rule.enabled)

    def test_disable_update(self):
        self.link.disable_update()
        self.assertFalse(self.link.x.update_rule.enabled)
        self.assertFalse(self.link.u.update_rule.enabled)

    def test_update_enabled(self):
        self.assertTrue(self.link.update_enabled)
        self.link.disable_update()
        self.assertFalse(self.link.update_enabled)
        self.link.enable_update()
        self.assertTrue(self.link.update_enabled)


class CountParameter(chainer.Parameter):

    def __init__(self, v):
        super(CountParameter, self).__init__(v.data, name=v.name)
        self.data = v.data
        self.grad = v.grad
        self.count_to_cpu = 0
        self.count_to_gpu = 0
        self.count_zerograd = 0

    def to_cpu(self):
        self.count_to_cpu += 1
        super(CountParameter, self).to_cpu()

    def to_gpu(self, device=None):
        self.count_to_gpu += 1
        super(CountParameter, self).to_gpu(device)

    def zerograd(self):
        self.count_zerograd += 1
        super(CountParameter, self).zerograd()


class TestChain(unittest.TestCase):

    def setUp(self):
        self.l1 = chainer.Link(x=(2, 3))
        self.l2 = chainer.Link(x=2)
        self.l3 = chainer.Link(x=None)

        self.c1 = chainer.Chain(l1=self.l1)
        self.c1.add_link('l2', self.l2)
        self.c2 = chainer.Chain(c1=self.c1)
        with self.c2.init_scope():
            self.c2.l3 = self.l3

    def test_init(self):
        self.assertIs(self.c1.l1, self.l1)
        self.assertIs(self.c1['l1'], self.l1)
        self.assertEqual(self.l1.name, 'l1')

        self.assertIs(self.c2.c1, self.c1)
        self.assertIs(self.c2['c1'], self.c1)
        self.assertEqual(self.c1.name, 'c1')

        self.assertIs(self.c2.l3, self.l3)
        self.assertIs(self.c2['l3'], self.l3)
        self.assertEqual(self.l3.name, 'l3')

    def test_add_link(self):
        self.assertIs(self.c1.l2, self.l2)
        self.assertEqual(self.l2.name, 'l2')

    def test_add_link_to_existing_attribute(self):
        self.l1.z = 0
        with self.assertRaises(AttributeError):
            self.l1.add_link('z', chainer.Link())

    def test_assign_link_outside_of_init_scope(self):
        l = chainer.Link()
        self.l1.l = l
        self.assertTrue(all(l is not link for link in self.l1.links()))

    def test_copy(self):
        c2 = self.c2.copy()
        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, set)
        self.assertTrue(hasattr(c2, 'c1'))
        self.assertEqual(c2.c1.name, 'c1')
        self.assertIsInstance(c2.c1._children, set)
        self.assertIsNot(c2.c1, self.c1)
        self.assertEqual(c2.c1.l1.name, 'l1')
        self.assertIsNot(c2.c1.l1, self.l1)
        self.assertIsNot(c2.c1.l1.x, self.l1.x)
        self.assertIs(c2.c1.l1.x.data, self.l1.x.data)
        self.assertIs(c2.c1.l1.x.grad, None)
        self.assertIs(c2.name, None)

        self.assertTrue(hasattr(c2.c1, 'l2'))
        self.assertEqual(c2.c1.l2.name, 'l2')
        self.assertIsNot(c2.c1.l2, self.l2)
        self.assertIsNot(c2.c1.l2.x, self.l2.x)
        self.assertIs(c2.c1.l2.x.data, self.l2.x.data)
        self.assertIs(c2.c1.l2.x.grad, None)

        self.assertTrue(hasattr(c2, 'l3'))
        self.assertEqual(c2.l3.name, 'l3')
        self.assertIsNot(c2.l3, self.l3)
        self.assertIsNot(c2.l3.x, self.l3.x)
        self.assertIs(c2.l3.x.data, self.l3.x.data)
        self.assertIs(c2.l3.x.grad, None)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.x.data
        gx1 = self.l1.x.grad
        x2 = self.l2.x.data
        gx2 = self.l2.x.grad
        x3 = self.l3.x.data
        gx3 = self.l3.x.grad

        self.c2.to_cpu()
        self.assertIs(self.l1.x.data, x1)
        self.assertIs(self.l1.x.grad, gx1)
        self.assertIs(self.l2.x.data, x2)
        self.assertIs(self.l2.x.grad, gx2)
        self.assertIs(self.l3.x.data, x3)
        self.assertIs(self.l3.x.grad, gx3)

    def set_count_parameters(self):
        self.l1.x = CountParameter(self.l1.x)
        self.l2.x = CountParameter(self.l2.x)
        self.l3.x = CountParameter(self.l3.x)

    @attr.gpu
    def test_to_cpu(self):
        self.set_count_parameters()
        self.c2.to_gpu()
        self.c2.to_cpu()
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsNone(self.l3.x.data)
        self.assertIsNone(self.l3.x.grad)
        self.assertEqual(self.l1.x.count_to_cpu, 1)
        self.assertEqual(self.l1.x.count_to_gpu, 1)
        self.assertEqual(self.l2.x.count_to_cpu, 1)
        self.assertEqual(self.l2.x.count_to_gpu, 1)
        self.assertEqual(self.l3.x.count_to_cpu, 1)
        self.assertEqual(self.l3.x.count_to_gpu, 1)

        self.l3.x.initialize(3)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        self.set_count_parameters()
        cupy = cuda.cupy
        self.c2.to_gpu()
        self.assertIs(self.c2.xp, cupy)
        self.assertIs(self.c1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.x.data, cupy.ndarray)
        self.assertIsInstance(self.l1.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.x.data, cupy.ndarray)
        self.assertIsInstance(self.l2.x.grad, cupy.ndarray)
        self.assertIsNone(self.l3.x.data)
        self.assertIsNone(self.l3.x.grad)
        self.assertEqual(self.l1.x.count_to_gpu, 1)
        self.assertEqual(self.l2.x.count_to_gpu, 1)
        self.assertEqual(self.l3.x.count_to_gpu, 1)

        self.l3.x.initialize(3)
        self.assertIsInstance(self.l3.x.data, cupy.ndarray)
        self.assertIsInstance(self.l3.x.grad, cupy.ndarray)

    def test_params(self):
        params = list(self.c2.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.x), id(self.l2.x), id(self.l3.x)})

    def test_params_skip_uninit(self):
        params = list(self.c2.params(include_uninit=False))
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.x), id(self.l2.x)})

    def test_namedparams(self):
        namedparams = list(self.c2.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/c1/l1/x', id(self.l1.x)),
                          ('/c1/l2/x', id(self.l2.x)),
                          ('/l3/x', id(self.l3.x))})

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.c2.namedparams(include_uninit=False))
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/c1/l1/x', id(self.l1.x)),
                          ('/c1/l2/x', id(self.l2.x))})

    def test_links(self):
        links = list(self.c2.links())
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3,
                                          self.c1, self.c2]})

    def test_links_skipself(self):
        links = list(self.c2.links(skipself=True))
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3, self.c1]})

    def test_namedlinks(self):
        namedlinks = list(self.c2.namedlinks())
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/', id(self.c2)),
                          ('/c1', id(self.c1)),
                          ('/c1/l1', id(self.l1)),
                          ('/c1/l2', id(self.l2)),
                          ('/l3', id(self.l3))})

    def test_namedlinks_skipself(self):
        namedlinks = list(self.c2.namedlinks(skipself=True))
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/c1', id(self.c1)),
                          ('/c1/l1', id(self.l1)),
                          ('/c1/l2', id(self.l2)),
                          ('/l3', id(self.l3))})

    def test_children(self):
        children = list(self.c2.children())
        self.assertEqual({id(c) for c in children}, {id(self.c1), id(self.l3)})

    def test_copyparams(self):
        l1 = chainer.Link(x=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.Chain(l1=l1, l2=l2)
        c2 = chainer.Chain(c1=c1, l3=l3)
        l1.x.data.fill(0)
        l2.x.data.fill(1)
        l3.x.data.fill(2)

        self.c2.copyparams(c2)

        numpy.testing.assert_array_equal(self.l1.x.data, l1.x.data)
        numpy.testing.assert_array_equal(self.l2.x.data, l2.x.data)
        numpy.testing.assert_array_equal(self.l3.x.data, l3.x.data)

    def test_zerograds(self):
        self.set_count_parameters()
        self.c2.zerograds()
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        self.assertEqual(self.l1.x.count_zerograd, 1)
        self.assertEqual(self.l2.x.count_zerograd, 1)
        self.assertEqual(self.l3.x.count_zerograd, 1)

        self.l3.x.initialize(3)
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

    def test_addgrads(self):
        l1 = chainer.Link(x=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.Chain(l1=l1, l2=l2)
        c2 = chainer.Chain(c1=c1, l3=l3)
        l1.x.grad.fill(1)
        l2.x.grad.fill(2)
        l3.x.grad.fill(3)

        self.l1.x.grad.fill(-1)
        self.l2.x.grad.fill(-2)
        self.l3.zerograds()

        self.c2.addgrads(c2)
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.full(3, 3.))

    def test_serialize(self):
        mocks = {'l1': mock.MagicMock(), 'l2': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        self.c1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('l1')
        serializer.__getitem__.assert_any_call('l2')

        mocks['l1'].assert_called_with('x', self.l1.x.data)
        mocks['l2'].assert_called_with('x', self.l2.x.data)


class TestChainList(unittest.TestCase):

    def setUp(self):
        self.l1 = chainer.Link(x=(2, 3), y=None)
        self.l2 = chainer.Link(x=2)
        self.l3 = chainer.Link(x=3)
        self.c1 = chainer.ChainList(self.l1)
        self.c1.add_link(self.l2)
        self.c2 = chainer.ChainList(self.c1)
        self.c2.append(self.l3)

    def test_init(self):
        self.assertIs(self.c1[0], self.l1)
        self.assertEqual(self.l1.name, '0')
        self.assertIs(self.c2[0], self.c1)
        self.assertEqual(self.c1.name, '0')

    def test_add_link(self):
        self.assertIs(self.c1[1], self.l2)
        self.assertEqual(self.l2.name, '1')

    def test_append(self):
        self.assertIs(self.c2[1], self.l3)
        self.assertEqual(self.l3.name, '1')

    def test_iter(self):
        links = list(self.c2)
        self.assertEqual(2, len(links))
        self.assertIs(links[0], self.c1)
        self.assertIs(links[1], self.l3)

    def test_len(self):
        self.assertEqual(len(self.c1), 2)
        self.assertEqual(len(self.c2), 2)

    def test_copy(self):
        c2 = self.c2.copy()

        self.assertIs(c2.name, None)
        self.assertIsInstance(c2._children, list)
        self.assertIsNot(c2[0], self.c1)
        self.assertEqual(c2[0].name, '0')
        self.assertIsInstance(c2[0]._children, list)
        self.assertIsNot(c2[0][0], self.l1)
        self.assertEqual(c2[0][0].name, '0')
        self.assertIsNot(c2[0][0].x, self.l1.x)
        self.assertIs(c2[0][0].x.data, self.l1.x.data)
        self.assertIs(c2[0][0].x.grad, None)

        self.assertIsNot(c2[0][1], self.l2)
        self.assertEqual(c2[0][1].name, '1')
        self.assertIsNot(c2[0][1].x, self.l2.x)
        self.assertIs(c2[0][1].x.data, self.l2.x.data)
        self.assertIs(c2[0][1].x.grad, None)

        self.assertIsNot(c2[1], self.l3)
        self.assertEqual(c2[1].name, '1')
        self.assertIsNot(c2[1].x, self.l3.x)
        self.assertIs(c2[1].x.data, self.l3.x.data)
        self.assertIs(c2[1].x.grad, None)

    @attr.gpu
    def test_copy_and_send_to_gpu(self):
        c2 = self.c2.copy()
        self.c2.to_gpu()
        self.assertIsInstance(self.c2[0][0].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(self.c2[0][1].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(c2[0][0].x.data, numpy.ndarray)
        self.assertIsInstance(c2[0][1].x.data, numpy.ndarray)

    @attr.gpu
    def test_copy_and_send_to_gpu_2(self):
        c2 = self.c2.copy()
        c2.to_gpu()
        self.assertIsInstance(self.c2[0][0].x.data, numpy.ndarray)
        self.assertIsInstance(self.c2[0][1].x.data, numpy.ndarray)
        self.assertIsInstance(c2[0][0].x.data, cuda.cupy.ndarray)
        self.assertIsInstance(c2[0][1].x.data, cuda.cupy.ndarray)

    @attr.multi_gpu(2)
    def test_copy_and_send_to_gpu_multi(self):
        c2 = self.c2.copy()
        self.c2.to_gpu(0)
        c2.to_gpu(1)
        self.assertEqual(self.c2[0][0].x.data.device.id, 0)
        self.assertEqual(self.c2[0][1].x.data.device.id, 0)
        self.assertEqual(c2[0][0].x.data.device.id, 1)
        self.assertEqual(c2[0][1].x.data.device.id, 1)

    def test_to_cpu_on_cpu(self):
        x1 = self.l1.x.data
        gx1 = self.l1.x.grad
        x2 = self.l2.x.data
        gx2 = self.l2.x.grad
        x3 = self.l3.x.data
        gx3 = self.l3.x.grad

        self.c2.to_cpu()

        self.assertIs(self.l1.x.data, x1)
        self.assertIs(self.l1.x.grad, gx1)
        self.assertIs(self.l2.x.data, x2)
        self.assertIs(self.l2.x.grad, gx2)
        self.assertIs(self.l3.x.data, x3)
        self.assertIs(self.l3.x.grad, gx3)

    @attr.gpu
    def test_to_cpu(self):
        self.c2.to_gpu()
        self.c2.to_cpu()
        self.assertIs(self.c2.xp, numpy)
        self.assertIs(self.c1.xp, numpy)
        self.assertIs(self.l1.xp, numpy)
        self.assertIs(self.l2.xp, numpy)
        self.assertIs(self.l3.xp, numpy)
        self.assertIsInstance(self.l1.x.data, numpy.ndarray)
        self.assertIsInstance(self.l1.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l2.x.data, numpy.ndarray)
        self.assertIsInstance(self.l2.x.grad, numpy.ndarray)
        self.assertIsInstance(self.l3.x.data, numpy.ndarray)
        self.assertIsInstance(self.l3.x.grad, numpy.ndarray)

    @attr.gpu
    def test_to_gpu(self):
        cupy = cuda.cupy
        self.c2.to_gpu()
        self.assertIs(self.c2.xp, cupy)
        self.assertIs(self.c1.xp, cupy)
        self.assertIs(self.l1.xp, cupy)
        self.assertIs(self.l2.xp, cupy)
        self.assertIs(self.l3.xp, cupy)
        self.assertIsInstance(self.l1.x.data, cupy.ndarray)
        self.assertIsInstance(self.l1.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l2.x.data, cupy.ndarray)
        self.assertIsInstance(self.l2.x.grad, cupy.ndarray)
        self.assertIsInstance(self.l3.x.data, cupy.ndarray)
        self.assertIsInstance(self.l3.x.grad, cupy.ndarray)

    def test_params(self):
        params = list(self.c2.params())
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.x), id(self.l1.y),
                          id(self.l2.x), id(self.l3.x)})

    def test_params_skip_uninit(self):
        params = list(self.c2.params(include_uninit=False))
        self.assertEqual({id(p) for p in params},
                         {id(self.l1.x), id(self.l2.x), id(self.l3.x)})

    def test_namedparams(self):
        namedparams = list(self.c2.namedparams())
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/0/0/x', id(self.l1.x)),
                          ('/0/0/y', id(self.l1.y)),
                          ('/0/1/x', id(self.l2.x)),
                          ('/1/x', id(self.l3.x))})

    def test_namedparams_skip_uninit(self):
        namedparams = list(self.c2.namedparams(include_uninit=False))
        self.assertEqual({(name, id(p)) for name, p in namedparams},
                         {('/0/0/x', id(self.l1.x)),
                          ('/0/1/x', id(self.l2.x)),
                          ('/1/x', id(self.l3.x))})

    def test_links(self):
        links = list(self.c2.links())
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3,
                                          self.c1, self.c2]})

    def test_links_skipself(self):
        links = list(self.c2.links(skipself=True))
        self.assertEqual({id(l) for l in links},
                         {id(l) for l in [self.l1, self.l2, self.l3, self.c1]})

    def test_namedlinks(self):
        namedlinks = list(self.c2.namedlinks())
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/', id(self.c2)),
                          ('/0', id(self.c1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))})

    def test_namedlinks_skipself(self):
        namedlinks = list(self.c2.namedlinks(skipself=True))
        self.assertEqual({(name, id(l)) for name, l in namedlinks},
                         {('/0', id(self.c1)),
                          ('/0/0', id(self.l1)),
                          ('/0/1', id(self.l2)),
                          ('/1', id(self.l3))})

    def test_children(self):
        self.assertEqual(tuple(id(c) for c in self.c2.children()),
                         (id(self.c1), id(self.l3)))

        self.assertEqual(tuple(id(c) for c in self.c1.children()),
                         (id(self.l1), id(self.l2)))

    def test_copyparams(self):
        l1 = chainer.Link(x=(2, 3), y=None)
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.ChainList(l1, l2)
        c2 = chainer.ChainList(c1, l3)
        l1.x.data.fill(0)
        l2.x.data.fill(1)
        l3.x.data.fill(2)

        self.c2.copyparams(c2)

        numpy.testing.assert_array_equal(self.l1.x.data, l1.x.data)
        numpy.testing.assert_array_equal(self.l2.x.data, l2.x.data)
        numpy.testing.assert_array_equal(self.l3.x.data, l3.x.data)

    def test_zerograds(self):
        self.c2.zerograds()
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))
        self.l1.y.initialize((2, 3))
        numpy.testing.assert_array_equal(self.l1.y.grad, numpy.zeros((2, 3)))

    def test_cleargrads(self):
        self.c2.cleargrads()
        self.assertIsNone(self.l1.x.grad)
        self.assertIsNone(self.l2.x.grad)
        self.assertIsNone(self.l3.x.grad)
        self.l1.y.initialize((2, 3))
        self.assertIsNone(self.l1.y.grad)

    def test_addgrads(self):
        l1 = chainer.Link(x=(2, 3), y=(2, 3))
        l2 = chainer.Link(x=2)
        l3 = chainer.Link(x=3)
        c1 = chainer.ChainList(l1, l2)
        c2 = chainer.ChainList(c1, l3)
        l1.x.grad.fill(1)
        l2.x.grad.fill(2)
        l3.x.grad.fill(3)
        l1.y.grad.fill(4)

        self.l1.x.grad.fill(-1)
        self.l1.y.cleargrad()
        self.l2.x.grad.fill(-2)
        self.l3.x.grad.fill(-3)

        self.c2.addgrads(c2)
        numpy.testing.assert_array_equal(self.l1.x.grad, numpy.zeros((2, 3)))
        numpy.testing.assert_array_equal(self.l1.y.grad, l1.y.grad)
        numpy.testing.assert_array_equal(self.l2.x.grad, numpy.zeros(2))
        numpy.testing.assert_array_equal(self.l3.x.grad, numpy.zeros(3))

    def test_serialize(self):
        l1 = chainer.Link(y=(1, 1))

        l2 = chainer.Link()
        with l2.init_scope():
            l2.x = chainer.Parameter(0, 2)
        c1 = chainer.ChainList(l1, l2)
        mocks = {'0': mock.MagicMock(), '1': mock.MagicMock()}
        serializer = mock.MagicMock()
        serializer.__getitem__.side_effect = lambda k: mocks[k]
        serializer.return_value = None
        mocks['0'].return_value = None
        mocks['1'].return_value = None
        c1.serialize(serializer)

        self.assertEqual(serializer.call_count, 0)
        self.assertEqual(serializer.__getitem__.call_count, 2)
        serializer.__getitem__.assert_any_call('0')
        serializer.__getitem__.assert_any_call('1')

        mocks['0'].assert_called_with('y', l1.y.data)
        mocks['1'].assert_called_with('x', l2.x.data)


class TestSequential(unittest.TestCase):

    def setUp(self):
        self.l1 = chainer.links.Linear(3, 3)
        self.l2 = chainer.links.Linear(3, 3)
        self.l3 = chainer.links.Linear(3, 4)
        self.x = chainer.Variable(numpy.zeros((1, 3), dtype=numpy.float32))
        self.model = chainer.Sequential(self.l1, self.l2, self.l3)
        self.layers = [self.l1, self.l2, self.l3]

    def test_init(self):
        with self.assertRaises(ValueError):
            chainer.Sequential(5)

    def test_len(self):
        self.assertIs(len(self.model), 3)

    def test_getitem(self):
        self.assertIs(self.model[0], self.l1)

    def test_delitem(self):
        del self.model[0]
        self.assertIsNot(self.model[0], self.l1)

    def test_iter(self):
        i = 0
        for layer in self.model:
            self.assertIs(layer, self.layers[i])
            i += 1

    def test_reversed(self):
        i = 0
        for layer in reversed(self.model):
            self.assertIs(layer, self.layers[len(self.layers) - 1 - i])
            i += 1

    def test_contains(self):
        for layer in self.layers:
            self.assertTrue(layer in self.model)

    def test_add(self):
        l1 = chainer.links.Linear(3, 3)
        l2 = chainer.links.Linear(3, 3)
        l3 = chainer.links.Linear(3, 4)
        other = chainer.Sequential(l1, l2, l3)
        add = self.model + other
        self.assertEqual(len(add), len(self.model) + len(other))
        for i, layer in enumerate(add):
            if i < len(self.model):
                self.assertIs(layer, self.model[i])
            else:
                i -= len(self.model)
                self.assertIs(layer, other[i])

        with self.assertRaises(ValueError):
            self.model + 5

    def test_radd(self):
        l0 = chainer.links.Linear(3, 3)
        add = l0 + self.model
        for i, layer in enumerate(add):
            if i == 0:
                self.assertIs(layer, l0)
            else:
                i -= 1
                self.assertIs(layer, self.model[i])

        with self.assertRaises(ValueError):
            5 + self.model

    def test_iadd(self):
        l4 = chainer.links.Linear(3, 3)
        self.model += l4
        for i, layer in enumerate(self.model):
            if i < len(self.layers):
                self.assertIs(layer, self.layers[i])
            else:
                self.assertIs(layer, l4)

        with self.assertRaises(ValueError):
            self.model += 5

    def test_mul(self):
        mul = self.model * 3
        for i, layer in enumerate(mul):
            j = i % 3
            self.assertEqual(layer.__class__, self.model[j].__class__)
        self.assertEqual(len(mul), 3 * len(self.model))

        mul = self.model * 0
        self.assertEqual(len(mul), 0)

    def test_rmul(self):
        mul = 3 * self.model
        for i, layer in enumerate(mul):
            j = i % 3
            self.assertEqual(layer.__class__, self.model[j].__class__)
        self.assertEqual(len(mul), 3 * len(self.model))

    def test_imul(self):
        mul = self.model.copy()
        mul *= 3
        for i, layer in enumerate(mul):
            j = i % 3
            self.assertEqual(layer.__class__, self.model[j].__class__)
        self.assertEqual(len(mul), 3 * len(self.model))

    def test_call(self):
        l1 = mock.MagicMock()
        l2 = mock.MagicMock()
        l3 = mock.MagicMock()
        model = chainer.Sequential(l1, l2, l3)
        y = model(self.x)
        model[0].assert_called_once()
        model[1].assert_called_once()
        model[2].assert_called_once()
        self.assertEqual(model[0], l1)
        self.assertEqual(model[1], l2)
        self.assertEqual(model[2], l3)

        y = self.model(self.x)
        self.assertIs(y.creator.inputs[1].data, self.l3.W.data)
        self.assertIs(y.creator.inputs[2].data, self.l3.b.data)

    def test_call_with_multiple_inputs(self):
        model = chainer.Sequential(
            lambda x: (x * 2, x * 3),
            lambda x, y: x + y
        )
        y = model(3)  # 6 + 9 = 15
        self.assertEqual(y, 15)

    def test_append(self):
        l4 = chainer.links.Linear(3, 3)
        self.model.append(l4)
        self.assertIs(self.model[-1], l4)

    def test_extend(self):
        l1 = chainer.links.Linear(3, 3)
        l2 = chainer.links.Linear(3, 3)
        l3 = chainer.links.Linear(3, 3)
        model_b = chainer.Sequential(l1, l2, l3)
        orig_n = len(self.model)
        self.model.extend(model_b)
        for i, layer in enumerate(self.model):
            if i < orig_n:
                self.assertIs(layer, self.model[i])
            else:
                self.assertIs(layer, model_b[i - orig_n])
        self.assertEqual(len(self.model), orig_n + len(model_b))

    def test_insert(self):
        l = chainer.links.Linear(3, 3)
        self.model.insert(2, l)
        self.assertEqual(len(self.model), 4)
        self.assertIs(self.model[2], l)

    def test_remove(self):
        l2 = self.layers[1]
        self.model.remove(l2)
        self.assertEqual(len(self.model), 2)
        self.assertIs(self.layers[0], self.model[0])
        self.assertIs(self.layers[2], self.model[1])

    def test_remove_by_layer_type(self):
        self.model.insert(2, chainer.functions.relu)
        self.model.remove_by_layer_type('Linear')
        self.assertEqual(len(self.model), 1)
        self.assertIs(self.model[0], chainer.functions.relu)

    def test_pop(self):
        l2 = self.model.pop(1)
        self.assertIs(l2, self.layers[1])
        self.assertEqual(len(self.model), len(self.layers) - 1)

    def test_clear(self):
        self.model.clear()
        self.assertEqual(len(self.model), 0)

    def test_index(self):
        l3 = self.layers[2]
        self.assertEqual(self.model.index(l3), 2)

    def test_count(self):
        self.model.insert(1, chainer.functions.relu)
        self.model.insert(3, chainer.functions.relu)
        self.assertEqual(self.model.count(chainer.functions.relu), 2)
        self.assertEqual(self.model.count(self.layers[0]), 1)
        self.model.append(self.layers[1])
        self.assertEqual(self.model.count(self.layers[1]), 2)

    def test_count_by_layer_type(self):
        self.assertEqual(self.model.count_by_layer_type('Linear'), 3)
        self.model.insert(1, chainer.functions.relu)
        self.model.insert(3, chainer.functions.relu)
        self.assertEqual(self.model.count_by_layer_type('relu'), 2)

    def test_sort(self):
        with self.assertRaises(NotImplementedError):
            self.model.sort()

    def test_reverse(self):
        with self.assertRaises(NotImplementedError):
            self.model.reverse()

    def test_copy(self):
        model = self.model.copy()
        for l1, l2 in zip(model, self.model):
            self.assertIsNot(l1, l2)
            if isinstance(l1, chainer.Link):
                self.assertEqual(l1.__class__.__name__, l2.__class__.__name__)
            else:
                self.assertEqual(l1.__name__, l2.__name__)


testing.run_module(__name__, __file__)
