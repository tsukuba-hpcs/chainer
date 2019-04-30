from chainer.iterators._statemachine import IteratorState
from chainer.iterators._statemachine import iterator_statemachine
from chainer.iterators.order_samplers import ShuffleOrderSampler
import numpy

dataset_length = 100
batch_size = 32
repeat = True
order_sampler = ShuffleOrderSampler()

initial_order = order_sampler(numpy.arange(dataset_length), 0)
current_state = IteratorState(0, 0, False, initial_order)

for _ in range(10):
    current_state, indices = iterator_statemachine(
        current_state,
        batch_size,
        repeat,
        order_sampler,
        dataset_length
    )

    print(current_state)
    print(indices)
