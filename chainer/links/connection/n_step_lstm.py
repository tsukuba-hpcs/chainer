import numpy
import six

from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_lstm as rnn
from chainer import link


def argsort_list_descent(lst):
    return numpy.argsort([-len(x.data) for x in lst]).astype('i')


def permutate_list(lst, indices, inv):
    ret = [None] * len(lst)
    if inv:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


class NStepLSTM(link.ChainList):

    def __init__(
            self, n_layers, in_size, out_size, dropout, seed=1337,
            use_cudnn=True):
        weights = []
        for i in six.moves.range(n_layers):
            weight = link.Link()
            for j in six.moves.range(8):
                if i == 0 and j < 4:
                    w_in = in_size
                else:
                    w_in = out_size
                weight.add_param('w%d' % j, (out_size, w_in))
                weight.add_param('b%d' % j, (out_size,))
                getattr(weight, 'w%d' % j).data[...] = numpy.random.normal(
                    0, numpy.sqrt(1. / w_in), (out_size, w_in))
                getattr(weight, 'b%d' % j).data[...] = 0
            weights.append(weight)

        super(NStepLSTM, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout = dropout
        self.seed = seed
        self.use_cudnn = use_cudnn

    def __call__(self, hx, cx, xs, train=True):
        assert isinstance(xs, (list, tuple))
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, inv=False)
        hx = permutate.permutate(hx, indices, axis=1, inv=False)
        cx = permutate.permutate(cx, indices, axis=1, inv=False)
        trans_x = transpose_sequence.transpose_sequence(xs)

        ws = [[w.w0, w.w1, w.w2, w.w3, w.w4, w.w5, w.w6, w.w7] for w in self]
        bs = [[w.b0, w.b1, w.b2, w.b3, w.b4, w.b5, w.b6, w.b7] for w in self]

        hy, cy, trans_y = rnn.n_step_lstm(
            self.n_layers, self.dropout, hx, cx, ws, bs, trans_x,
            seed=self.seed, train=train, use_cudnn=self.use_cudnn)

        hy = permutate.permutate(hy, indices, axis=1, inv=True)
        cy = permutate.permutate(cy, indices, axis=1, inv=True)
        ys = transpose_sequence.transpose_sequence(trans_y)
        ys = permutate_list(ys, indices, inv=True)

        return hy, cy, ys
