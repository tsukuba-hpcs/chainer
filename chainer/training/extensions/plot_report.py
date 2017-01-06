import json
from os import path

import numpy
import six

from chainer import reporter
from chainer.training import extension
import chainer.training.trigger as trigger_module

try:
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plot

    _available = True

except ImportError:
    _available = False


def _check_available():
    if not _available:
        msg = '''matplotlib is not installed on your environment.
Please install matplotlib to plot figure.

  $ pip install matplotlib'''
        raise RuntimeError(msg)


class PlotReport(extension.Extension):

    """Trainer extension to output plots.

    This extension accumulates the observations of the trainer to
    :class:`~chainer.DictSummary` at a regular interval specified by a supplied
    trigger, and plot a graph with using them.

    There are two triggers to handle this extension. One is the trigger to
    invoke this extension, which is used to handle the timing of accumulating
    the results. It is set to ``1, 'iteration'`` by default. The other is the
    trigger to determine when to emit the result. When this trigger returns
    True, this extension appends the summary of accumulated values to the list
    of past summaries, and writes the list to the log file. Then, this
    extension makes a new fresh summary object which is used until the next
    time that the trigger fires.

    It also adds ``'epoch'`` and ``'iteration'`` entries to each result
    dictionary, which are the epoch and iteration counts at the output.

    Args:
        y_keys (iterable of strs): Keys of values regarded as y. If this is
            None, nothing is output to the graph.
        x_key (str): Keys of values regarded as x. The default value is
            'iteration'.
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or ``<int>,
            'iteration'``, it is passed to :class:`IntervalTrigger`.
        postprocess: Callback to postprocess the result dictionaries. Figure
            object, Axes object, and all plot data are passed to this callback
            in this order. This callback can modify the figure.
        file_name (str): Name of the figure file under the output directory.
            It can be a format string.

    """

    def __init__(self, y_keys, x_key='iteration', trigger=(1, 'epoch'),
                 postprocess=None, file_name='plot.png'):

        _check_available()

        self._x_key = x_key
        if isinstance(y_keys, str):
            y_keys = (y_keys,)

        self._y_keys = y_keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._file_name = file_name
        self._postprocess = postprocess
        self._init_summary()
        self._data = {k: [] for k in y_keys}

    def __call__(self, trainer):
        keys = self._y_keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._trigger(trainer):
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            updater = trainer.updater
            stats_cpu['epoch'] = updater.epoch
            stats_cpu['iteration'] = updater.iteration
            x = stats_cpu[self._x_key]
            data = self._data

            for k in keys:
                if k in stats_cpu:
                    data[k].append((x, stats_cpu[k]))

            f = plot.figure()
            a = f.add_subplot(111)
            a.set_xlabel(self._x_key)
            a.grid()

            for k in keys:
                xy = data[k]
                if len(xy) == 0:
                    continue

                xy = numpy.array(xy)
                a.plot(xy[:, 0], xy[:, 1], marker='x', label=k)

            if a.has_data():
                if self._postprocess is not None:
                    self._postprocess(f, a, summary)
                l = a.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
                f.savefig(path.join(trainer.out, self._file_name),
                          bbox_extra_artists=(l,), bbox_inches='tight')

            plot.close()
            self._init_summary()

    def serialize(self, serializer):
        if isinstance(serializer, serializer_module.Serializer):
            serializer('_plot_{}'.format(self._file_name),
                       json.dumps(self._data))

        else:
            self._data = json.loads(
                serializer('_plot_{}'.format(self._file_name), ''))

    def _init_summary(self):
        self._summary = reporter.DictSummary()
