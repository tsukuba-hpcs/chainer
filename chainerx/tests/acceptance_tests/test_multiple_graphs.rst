Acceptance tests for multiple graphs
====================================

Double backprop with different graphs
-------------------------------------

>>> import chainerx as xc

>>> with xc.backprop_scope('weight') as weight_backprop:
...     with xc.backprop_scope('input') as input_backprop:
...         x = xc.ndarray((3,), xc.float32, [1, 2, 3]).require_grad(input_backprop)
...         w = xc.ndarray((3,), xc.float32, [4, 5, 6]).require_grad(weight_backprop)
...         y = x * w
...         y.is_backprop_required(input_backprop)
True

...         y.is_backprop_required(weight_backprop)
True

...         y.is_backprop_required()  # 'default'
False

...         xc.backward(y, backprop_id=input_backprop)
...         gx = x.get_grad(input_backprop)
...         gx  # == w
array([4., 5., 6.], shape=(3,), dtype=float32, device='native:0', backprop_ids=['weight'])

...         w.get_grad(input_backprop)
Traceback (most recent call last):
  ...
chainerx.XchainerError: Array does not belong to the graph: 'input'.

...     z = gx * w  # == w * w
...     xc.backward(z, backprop_id=weight_backprop)
...     w.get_grad(weight_backprop)  # == 2 * w
array([ 8., 10., 12.], shape=(3,), dtype=float32, device='native:0')

...     x.get_grad(weight_backprop)
Traceback (most recent call last):
  ...
chainerx.XchainerError: Array does not belong to the graph: 'weight'.

Double backprop with single graph
---------------------------------

>>> x = xc.ndarray((3,), xc.float32, [1, 2, 3]).require_grad()
>>> w = xc.ndarray((3,), xc.float32, [4, 5, 6]).require_grad()
>>> y = x * w
>>> y.is_backprop_required()
True
>>> with xc.backprop_scope('foo') as foo:
...     y.is_backprop_required(foo)  # unknown backprop name
False

>>> xc.backward(y, enable_double_backprop=True)
>>> gx = x.get_grad()
>>> gx  # == w
array([4., 5., 6.], shape=(3,), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> w.get_grad()  # == x
array([1., 2., 3.], shape=(3,), dtype=float32, device='native:0', backprop_ids=['<default>'])

>>> w.cleargrad()
>>> z = gx * w  # == w * w
>>> xc.backward(z)
>>> w.get_grad()  # == 2 * w
array([ 8., 10., 12.], shape=(3,), dtype=float32, device='native:0')
>>> x.get_grad()  # the second backprop does not reach here
array([4., 5., 6.], shape=(3,), dtype=float32, device='native:0', backprop_ids=['<default>'])
>>> x.get_grad() is gx
True
