import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check


def _extract_gates(x):
    # reshape:change shape, //:divide(integer)
    r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
    return (r[:, :, i] for i in six.moves.range(4))


def _sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) { return 1 / (1 + exp(-x)); }
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_ + c_prev); \
    T af = sigmoid(f + c_prev); \
'''


class Peephole(function.Function):

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 5)
        #c_type, x_type = in_types

        #type_check.expect(
        #    c_type.dtype == numpy.float32,
        #    x_type.dtype == numpy.float32,

        #    c_type.ndim >= 2,
        #    x_type.ndim >= 2,
        #    c_type.ndim == x_type.ndim,

        #    x_type.shape[0] == c_type.shape[0],
        #    x_type.shape[1] == 4 * c_type.shape[1],
        #)
        #for i in range(2, c_type.ndim.eval()):
        #    type_check.expect(x_type.shape[i] == c_type.shape[i])

    def forward(self, inputs):
        c_prev, x, peep_in_i, peep_in_f, peep_in_o = inputs
        a, i, f, o = _extract_gates(x)

        if isinstance(x, numpy.ndarray):
            self.a = numpy.tanh(a)
            self.i = _sigmoid(i + peep_in_i)
            self.f = _sigmoid(f + peep_in_f)
            self.o = _sigmoid(o + peep_in_o)
            
            self.c = self.a * self.i + self.f * c_prev
            h = self.o * numpy.tanh(self.c)
        #else:
        #    self.c, h = cuda.elementwise(
        #        'T c_prev, T a, T i_, T f, T o', 'T c, T h',
        #        '''
        #            COMMON_ROUTINE;
        #            c = aa * ai + af * c_prev;
        #            T ao = sigmoid(o + c); 
        #            h = ao * tanh(c);
        #        ''',
        #        'peep_fwd', preamble=_preamble)(c_prev, a, i, f, o)

        return self.c, h
            
    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        c_prev, x, peep_in_i, peep_in_f, peep_in_o = inputs
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        ga, gi, gf, go = _extract_gates(gx)
        gpeep_in_i = xp.empty([c_prev.size, c_prev.size]) 
        gpeep_in_f = xp.empty([c_prev.size, c_prev.size]) 
        gpeep_in_o = xp.empty([c_prev.size, c_prev.size])   

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        if xp is numpy:
            co = numpy.tanh(self.c)
            gc_prev = gh * self.o * _grad_tanh(co) + gc  # multiply f later 
            ga[:] = gc_prev * self.i * _grad_tanh(self.a)
            gi[:] = gc_prev * self.a * _grad_sigmoid(self.i)
            gpeep_in_i = c_prev  * self.a * _grad_sigmoid(self.i) 
            gf[:] = gc_prev * c_prev * _grad_sigmoid(self.f)
            gpeep_in_f = c_prev * c_prev * _grad_sigmoid(self.f) 
            go[:] = gh * self.c * _grad_sigmoid(self.o)         
            gpeep_in_o = gh * self.c * _grad_sigmoid(self.o) 
            gc_prev *= self.f  # multiply f here
        #else:
        return gc_prev, gx, gpeep_in_i, gpeep_in_f, gpeep_in_o    


def peephole(c_prev, x, peep_in_i, peep_in_f, peep_in_o):
    return Peephole()(c_prev, x, peep_in_i, peep_in_f, peep_in_o)
