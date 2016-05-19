import numpy

from chainer import cuda
from chainer.functions.array import permutate
from chainer.functions.array import transpose_sequence
from chainer.functions.connection import n_step_lstm as rnn
from chainer import link


def argsort_list_descent(lst):
    return numpy.argsort([-len(x.data) for x in lst])


def permutate_list(lst, indices, rev):
    ret = [None] * len(lst)
    if rev:
        for i, ind in enumerate(indices):
            ret[ind] = lst[i]
    else:
        for i, ind in enumerate(indices):
            ret[i] = lst[ind]
    return ret


class NStepLSTM(link.ChainList):

    def __init__(self, n_layers, in_size, out_size, dropout, seed):
        weights = []
        for i in range(0, n_layers):
            weight = link.Link()
            for j in range(0, 8):
                if i == 0 and j < 4:
                    w_in = in_size
                else:
                    w_in = out_size
                weight.add_param('w%d' % j, (out_size, w_in))
                weight.add_param('b%d' % j, (out_size,))
            weights.append(weight)

        super(NStepLSTM, self).__init__(*weights)

        handle = cuda.cupy.cudnn.get_handle()
        states = rnn.DropoutStates.create(handle, dropout, seed)
        self.states = states
        self.n_layers = n_layers

    def __call__(self, h, c, xs):
        assert isinstance(xs, list)
        indices = argsort_list_descent(xs)

        xs = permutate_list(xs, indices, rev=False)
        hx = permutate.permutate(h, indices, axis=1, rev=False)
        cx = permutate.permutate(c, indices, axis=1, rev=False)
        trans_x = transpose_sequence.transpose_sequence(*xs)

        args = [hx, cx]
        for w in self:
            for i in range(0, 8):
                args.append(getattr(w, 'w%d' % i))
        for w in self:
            for i in range(0, 8):
                args.append(getattr(w, 'b%d' % i))
        args.extend(trans_x)
        ret = rnn.NStepLSTM(self.n_layers, self.states)(*args)
        hy, cy = ret[:2]
        trans_y = ret[2:]

        hy = permutate.permutate(hy, indices, axis=1, rev=True)
        cy = permutate.permutate(cy, indices, axis=1, rev=True)
        ys = transpose_sequence.transpose_sequence(*trans_y)
        ys = permutate_list(ys, indices, True)

        return hy, cy, ys
