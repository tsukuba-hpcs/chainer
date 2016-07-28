from __future__ import print_function
import functools
import six


def mulexp(xs, init=None):
    if init:
        return functools.reduce('{} * {}'.format, xs, init)
    else:
        return functools.reduce('{} * {}'.format, xs)


def andexp(xs, init=None):
    if init:
        return functools.reduce('{} && {}'.format, xs, init)
    else:
        return functools.reduce('{} && {}'.format, xs)


def muladdexp(xs, ys, init=None):
    def aux(exp, arg):
        x, y = arg
        return '({} + {} * {})'.format(y, x, exp)
    if init:
        return functools.reduce(aux, zip(xs, ys), init)
    else:
        return functools.reduce(aux, zip(xs, ys))


def _succ_sublists(xs):
    # Returns successive sublists of xs.
    return [xs[i:] for i in six.moves.range(len(xs))]


def _vars(prefix, n):
    return ['{}_{}'.format(prefix, i) for i in six.moves.range(n)]


def _writer():
    _indent = [0]
    _lines = []

    def _aux(line=None, indent=None):
        if line is None:
            return '\n'.join(_lines)
        else:
            if indent == 'dec' or indent == 'decinc':
                _indent[0] -= 1
            _lines.append('  ' * _indent[0] + line)
            if indent == 'inc' or indent == 'decinc':
                _indent[0] += 1
    return _aux


#
# im2col

class Im2colNDKernel(object):

    def __init__(self, ndim):
        self.ndim = ndim
        self.ds = _vars('d', ndim)
        self.outs = _vars('out', ndim)
        self.ks = _vars('k', ndim)
        self.ss = _vars('s', ndim)
        self.ps = _vars('p', ndim)

    def _in_params(self):
        # 2D: raw T img, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(['raw T img'] + map(
            aux, self.ds + self.outs + self.ks + self.ss + self.ps))

    def _out_params(self):
        return 'T col'

    def _compile_c0(self):
        # 2D: int c0 = i / (k_0 * k_1 * out_0 * out_1)
        return ['int c0 = i / ({});'.format(mulexp(self.ks + self.outs))]

    def _compile_kx(self):
        # 2D: int kx_0 = i / (k_1 * out_0 * out_1) % k_0;
        #     int kx_1 = i / (out_0 * out_1) % k_1;
        def aux(kx, xs):
            head = xs[0]
            tail = xs[1:] + self.outs
            if tail:
                return 'int {} = i / ({}) % {};'.format(kx, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(kx, head)
        kxs = _vars('kx', self.ndim)
        kx_decls = map(aux, kxs, _succ_sublists(self.ks))
        return kx_decls, kxs

    def _compile_out_x(self):
        # 2D: int out_x0 = i / (out_1) % out_0;
        #     int out_x1 = i % out_1;
        def aux(out_x, xs):
            head = xs[0]
            tail = xs[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(
                    out_x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(out_x, head)
        out_xs = _vars('out_x', self.ndim)
        out_x_decls = map(aux, out_xs, _succ_sublists(self.outs))
        return out_x_decls, out_xs

    def _compile_main(self, kxs, out_xs):
        # 2D: int in_0 = kx_0 + out_x_0 * s_0 - p_0;
        #     int in_1 = kx_1 + out_x_1 * s_1 - p_1;
        #     if (0 <= in_0 && in_0 < d_0 && 0 <= in_1 && in_1 < d_1) {
        #       int idx_0 = in_0 * d_0 + c0;
        #       int idx_1 = in_1 * d_1 + idx_0;
        #       col = img[idx_1];
        #     } else {
        #       col = (T)0;
        #     }
        w = _writer()

        ins = _vars('in', self.ndim)
        for (_in, kx, out_x, s, p) in zip(ins, kxs, out_xs, self.ss, self.ps):
            w('int {} = {} + {} * {} - {};'.format(_in, kx, out_x, s, p))

        def rel_aux(_in, d):
            return '0 <= {} && {} < {}'.format(_in, _in, d)
        w('if ({}) {{'.format(andexp(map(rel_aux, ins, self.ds))), 'inc')

        idxs = _vars('idx', self.ndim)
        idx0s = ['c0'] + idxs[:-1]
        for (idx, _in, d, idx0) in zip(idxs, ins, self.ds, idx0s):
            w('int {} = {} * {} + {};'.format(idx, _in, d, idx0))

        w('col = img[{}];'.format(idxs[-1]))
        w('} else {', 'decinc')
        w('col = (T)0;')
        w('}', 'dec')

        return [w()]

    def _operation(self):
        c0 = self._compile_c0()
        kx, kxs = self._compile_kx()
        out_x, out_xs = self._compile_out_x()
        main = self._compile_main(kxs, out_xs)
        return '\n'.join(c0 + kx + out_x + main)

    def generate(self):
        in_params = self._in_params()
        out_params = self._out_params()
        operation = self._operation()
        name = name = 'im2col_{}d'.format(self.ndim)
        return in_params, out_params, operation, name


#
# col2im

class Col2imNDKernel(object):

    def __init__(self, ndim):
        self.ndim = ndim
        self.ds = _vars('d', ndim)
        self.outs = _vars('out', ndim)
        self.ks = _vars('k', ndim)
        self.ss = _vars('s', ndim)
        self.ps = _vars('p', ndim)

    def _in_params(self):
        # 2D: raw T col, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(['raw T col'] + map(
            aux, self.ds + self.outs + self.ks + self.ss + self.ps))

    def _out_params(self):
        return 'T img'

    def _compile_c0(self):
        # 2D: int c0 = i / (d_0 * d_1);
        return ['int c0 = i / ({});'.format(mulexp(self.ds))]

    def _compile_x(self):
        # 2D: int x_0 = i / (d_1) % d_0 + p_0;
        #     int x_1 = i % d_1 + p_1;
        def aux(x, ds, p):
            head = ds[0]
            tail = ds[1:]
            if tail:
                return 'int {} = i / ({}) % {} + {};'.format(
                    x, mulexp(tail), head, p)
            else:
                return 'int {} = i % {} + {};'.format(x, head, p)
        xs = _vars('x', self.ndim)
        x_decls = map(aux, xs, _succ_sublists(self.ds), self.ps)
        return x_decls, xs

    def _compile_loop(self, xs):
        # 2D: int out_x0_0 = max(0,     (x_0 - k_0 + s_0) / s_0);
        #     int out_x1_0 = min(out_0, (x_0       + s_0) / s_0);
        #     int out_x0_1 = max(0,     (x_1 - k_1 + s_1) / s_1);
        #     int out_x1_1 = min(out_1, (x_1       + s_1) / s_1);
        #     ... Before-part here ...
        #     for (int out_x_0 = out_x0_0; out_x_0 < out_x1_0; ++out_x_0) {
        #       int kx_0 = x_0 - out_x_0 * s_0 + k_0 * c0;
        #       for (int out_x_1 = out_x0_1; out_x_1 < out_x1_1; ++out_x_1) {
        #         int kx_1 = x_1 - out_x_1 * s_1 + k_1 * kx_0;
        #         ... Main-part here ...
        #       }
        #     }
        #     ... After-part here ...
        def aux(out_x0, out_x1, out, x, k, s):
            return [
                'int {} = max(0, ({} - {} + {}) / {});'.format(
                    out_x0, x, k, s, s),
                'int {} = min({}, ({} + {}) / {});'.format(
                    out_x1, out, x, s, s)]
        out_x0s = _vars('out_x0', self.ndim)
        out_x1s = _vars('out_x1', self.ndim)
        bounds = sum(map(
            aux, out_x0s, out_x1s, self.outs, xs, self.ks, self.ss), [])

        def _loop_main(main):
            w = _writer()

            # Loop openings.
            out_xs = _vars('out_x', self.ndim)
            kxs = _vars('kx', self.ndim)
            kxs1 = ['c0'] + kxs[:-1]
            for (out_x, out_x0, out_x1, kx, s, x, k, kx1) in zip(
                    out_xs, out_x0s, out_x1s, kxs, self.ss, xs, self.ks, kxs1):
                w('for (int {} = {}; {} < {}; ++{}) {{'.format(
                    out_x, out_x0, out_x, out_x1, out_x), 'inc')
                w('int {} = {} - {} * {} + {} * {};'.format(
                    kx, x, out_x, s, k, kx1))

            # Main-part.
            kx = kxs[-1]
            for l in main(kx, out_xs).split('\n'):
                w(l)

            # Loop closings.
            for _ in out_xs:
                w('}', 'dec')

            return [w()]

        return bounds, _loop_main

    def _compile_procedure(self, xs):
        # 2D: val = val + col[(out_x_1 + out_1 * (out_x_0 + out_0 * kx_1))];
        def _main(kx, out_xs):
            index = muladdexp(self.outs, out_xs, kx)
            return 'val = val + col[{}];'.format(index)
        before = ['T val = 0;']
        after = ['img = val;']
        return before, _main, after

    def _operation(self):
        c0 = self._compile_c0()
        x, xs = self._compile_x()
        loop_bounds, loop_main = self._compile_loop(xs)
        before, main, after = self._compile_procedure(xs)
        return '\n'.join(
            c0 + x + loop_bounds + before + loop_main(main) + after)

    def generate(self):
        in_params = self._in_params()
        out_params = self._out_params()
        operation = self._operation()
        name = 'col2im_{}d'.format(self.ndim)
        return in_params, out_params, operation, name
