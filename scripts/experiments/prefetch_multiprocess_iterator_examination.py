import numpy
import multiprocessing
import logging

from chainer.datasets.image_dataset import ExtendedLabeledImageDataset
from chainer.iterators._statemachine import IteratorState
from chainer.iterators.order_samplers import ShuffleOrderSampler
from chainer.iterators.prefetch_multiprocess_iterator import _Communicator
from chainer.iterators.prefetch_multiprocess_iterator import _PrefetchPipeline
from chainer.iterators.prefetch_multiprocess_iterator import PrefetchMultiprocessIterator

order_sampler = ShuffleOrderSampler()
dataset = ExtendedLabeledImageDataset(
    pairs='/Users/kazuhiroserizawa/Documents/python-work/prefetch_pipeline_poc/img_test.ssv',
    root='/Users/kazuhiroserizawa/Documents/python-work/prefetch_pipeline_poc'
)
communicator = _Communicator(n_prefetch=10, dataset_timeout=30.0)
initial_order = order_sampler(numpy.arange(len(dataset)), 0)
prefetch_state = IteratorState(0, 0, False, initial_order)
communicator.reset(prefetch_state)

# prefetch_pipeline = _PrefetchPipeline(
#     dataset=dataset,
#     batch_size=8,
#     local_storage_base='/Users/kazuhiroserizawa/Documents/python-work/prefetch_pipeline_poc/local_storage',
#     n_prefetch_from_backend=2,
#     n_generate_batch=2,
#     n_remove_example=2,
#     comm=communicator,
#     order_sampler=order_sampler
# )
# prefetch_pipeline.launch_thread()
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# print(prefetch_pipeline._comm.get()[1])
# prefetch_pipeline.terminate()
#

multiprocessing.log_to_stderr(logging.INFO)

iterator = PrefetchMultiprocessIterator(
    dataset=dataset,
    batch_size=8,
    local_storage_base='/Users/kazuhiroserizawa/Documents/python-work/prefetch_pipeline_poc/local_storage',
    n_prefetch=64,
    n_prefetch_from_backend=2,
    n_generate_batch=2,
    n_remove_example=2
)
iterator.__next__()
iterator.__next__()
iterator.__next__()
iterator.__next__()
iterator.__next__()
iterator.finalize()
