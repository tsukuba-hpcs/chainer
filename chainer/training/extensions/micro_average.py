import collections

from chainer import reporter
from chainer.training import extension
from chainer.training import util


class MicroAverage(extension.Extension):

    """Calculates micro-average accuracy.

    Give :math:`N` batches and values :math:`\\{n_1, \dots, n_N\\}` and
    :math:`\\{d_1, \dots, d_N\\}`, this extension calculates macro-average of
    these ratio defined as:

    .. math::

       \\frac{\\sum_i^N n_i}{\\sum_i^N d_i}.

    A user usually uses the number of examples which a system correctly
    predict as :math:`n_i` and the number of total examples in :math:`i`-th
    batch as :math:`d_i`. This value is called macro-average of precision.

    Note that macro-average is defined as:

    .. math::

       \\frac{1}{N}\\sum_i^N (n_i / d_i),

    It is same to the macro-average when each mini-batch has the same
    :math:`d_i`.

    Args:
        numerator_key (str): Key string of obserbation storing a numerator
            value.
        denominator_key (str): Key string of obserbation storing a denominator
            value.
        result_key (str): Key string of obserbation to store a result.
        trigger: Trigger that decides when to calcurate average.
            This is distinct from the trigger of this extension itself.
            If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.

    """

    priority = extension.PRIORITY_EDITOR

    def __init__(
            self, numerator_key, denominator_key, result_key,
            trigger=(1, 'epoch')):
        self._trigger = util.get_trigger(trigger)

        self._numerator_key = numerator_key
        self._denominator_key = denominator_key
        self._result_key = result_key
        self._numerator = 0
        self._denominator = 0

    def __call__(self, trainer):
        observation = trainer.observation
        if not (self._numerator_key in observation and
                self._denominator_key in observation):
            return

        self._numerator += observation[self._numerator_key]
        self._denominator += observation[self._denominator_key]

        if self._trigger(trainer):
            result = float(self._numerator) / self._denominator
            self._numerator = 0
            self._denominator = 0
            reporter.report({self._result_key: result})
