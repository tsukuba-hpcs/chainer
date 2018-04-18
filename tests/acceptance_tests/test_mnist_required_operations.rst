Acceptance tests for operations required in MNIST
=================================================

>>> import xchainer as xc
>>> import numpy as np

Dot
---

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()
>>> b = xc.Array((3, 2), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()

>>> y = a.dot(b)
>>> y
array([[22., 28.],
       [49., 64.]], shape=(2, 2), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=True)
>>> a.grad
array([[ 3.,  7., 11.],
       [ 3.,  7., 11.]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])
>>> b.grad
array([[5., 5.],
       [7., 7.],
       [9., 9.]], shape=(3, 2), dtype=float32, device='native:0', graph_ids=['default'])

Subtract
--------

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()
>>> b = xc.Array((1, 3), xc.float32, [2, 2, 2]).require_grad()

>>> y = a - b
>>> y
array([[-1.,  0.,  1.],
       [ 2.,  3.,  4.]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=True)
>>> a.grad
array([[1., 1., 1.],
       [1., 1., 1.]], shape=(2, 3), dtype=float32, device='native:0')
>>> b.grad
array([[-2., -2., -2.]], shape=(1, 3), dtype=float32, device='native:0')


Divide
------

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()
>>> b = xc.Array((1, 3), xc.float32, [4, 5, 6]).require_grad()

>>> y = a / b
>>> y
array([[0.25, 0.4 , 0.5 ],
       [1.  , 1.  , 1.  ]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=False)
>>> a.grad
array([[0.25      , 0.2       , 0.16666667],
       [0.25      , 0.2       , 0.16666667]], shape=(2, 3), dtype=float32, device='native:0')
>>> b.grad
array([[-0.3125, -0.28  , -0.25  ]], shape=(1, 3), dtype=float32, device='native:0')

Max
---

>>> a = xc.Array((2, 3), xc.float32, [3, 2, 1, 4, 5, 6]).require_grad()

>>> y = xc.amax(a, axis=(1,), keepdims=True)
>>> y
array([[3.],
       [6.]], shape=(2, 1), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=True)
>>> a.grad
array([[1., 0., 0.],
       [0., 0., 1.]], shape=(2, 3), dtype=float32, device='native:0')

Argmax
------

>>> a = xc.Array((2, 3), xc.float32, [3, 2, 1, 4, 5, 6])

>>> y = xc.argmax(a, axis=1)
>>> y
array([0, 2], shape=(2,), dtype=int64, device='native:0')

Log
---

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()

>>> y = xc.log(a)
>>> y
array([[0.        , 0.69314718, 1.0986123 ],
       [1.38629436, 1.60943794, 1.79175949]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=False)
>>> a.grad
array([[1.        , 0.5       , 0.33333334],
       [0.25      , 0.2       , 0.16666667]], shape=(2, 3), dtype=float32, device='native:0')

Exp
---

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()

>>> y = xc.exp(a)
>>> y
array([[  2.71828175,   7.3890562 ,  20.08553696],
       [ 54.59814835, 148.41316223, 403.42880249]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=False)
>>> a.grad
array([[  2.71828175,   7.3890562 ,  20.08553696],
       [ 54.59814835, 148.41316223, 403.42880249]], shape=(2, 3), dtype=float32, device='native:0')


Negative
--------

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()

>>> y = -a
>>> y
array([[-1., -2., -3.],
       [-4., -5., -6.]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=True)
>>> a.grad
array([[-1., -1., -1.],
       [-1., -1., -1.]], shape=(2, 3), dtype=float32, device='native:0')

Log of Softmax
--------------

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()

>>> y = xc.log_softmax(a)
>>> y
array([[-2.40760589, -1.40760589, -0.40760589],
       [-2.40760612, -1.40760612, -0.40760612]], shape=(2, 3), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=False)
>>> a.grad
array([[0.72990829, 0.26581454, -0.99572289],
       [0.72990829, 0.26581454, -0.99572289]], shape=(2, 3), dtype=float32, device='native:0')


AsType
------

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()

>>> y = a.astype(xc.float32)
>>> y is a
False
>>> y = a.astype(xc.float32, copy=False)
>>> y is a
True
>>> y = a.astype(xc.float64, copy=False)
>>> y is a
False

>>> y = a.astype(xc.float64)
>>> y
array([[1., 2., 3.],
       [4., 5., 6.]], shape=(2, 3), dtype=float64, device='native:0', graph_ids=['default'])
>>> y.backward(enable_double_backprop=True)
>>> a.grad
array([[1., 1., 1.],
       [1., 1., 1.]], shape=(2, 3), dtype=float32, device='native:0')

>>> y = a.astype(xc.int32)
>>> y  # not backpropagatable
array([[1, 2, 3],
       [4, 5, 6]], shape=(2, 3), dtype=int32, device='native:0')

Take
----

>>> a = xc.Array((2, 3), xc.float32, [1, 2, 3, 4, 5, 6]).require_grad()
>>> indicies = xc.Array((2,), xc.int64, [1, 2])
>>> y = a.take(indicies, axis=1)
>>> y
array([[2., 3.],
       [5., 6.]], shape=(2, 2), dtype=float32, device='native:0', graph_ids=['default'])

>>> y.backward(enable_double_backprop=True)
>>> a.grad
array([[0., 1., 1.],
       [0., 1., 1.]], shape=(2, 3), dtype=float32, device='native:0')
