import argparse
import time
import sys
import chainer

from chainer.datasets.image_dataset import LabeledImageDataset
from chainer.datasets.image_dataset import ExtendedLabeledImageDataset
from chainer.iterators.prefetch_multiprocess_iterator import PrefetchMultiprocessIterator

parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=100)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--root', type=str, required=True)
parser.add_argument('--local_storage_base', type=str, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--n_prefetch', type=int, required=True)
parser.add_argument('--n_prefetch_from_backend', type=int, required=True)
parser.add_argument('--n_generate_batch', type=int, required=True)
parser.add_argument('--n_remove_example', type=int, required=True)

args = parser.parse_args()
print(vars(args))

dataset = ExtendedLabeledImageDataset(
    pairs=args.train,
    root=args.root
)

iterator = PrefetchMultiprocessIterator(
    dataset=dataset,
    batch_size=args.batch_size,
    local_storage_base=args.local_storage_base,
    n_prefetch=args.n_prefetch,
    n_prefetch_from_backend=args.n_prefetch_from_backend,
    n_generate_batch=args.n_generate_batch,
    n_remove_example=args.n_remove_example
)

s = time.time()
elapsed_times = []
for i in range(args.count):
    s_iteration = time.time()
    sys.stderr.write(f'{i}/{args.count}\r')
    sys.stderr.flush()
    data = iterator.__next__()
    iteration_elapsed_time = time.time() - s_iteration
    elapsed_times.append(iteration_elapsed_time)
    '''
    print(
        type(data), len(data),
        type(data[0]), len(data[0]),
        type(data[0][0]), len(data[0][0]), data[0][0].shape,
        type(data[0][1]), data[0][1], data[0][1].shape,
        file=sys.stderr
    )
    '''
elapsed_time = time.time() - s
sys.stderr.write('\n')
print(elapsed_time)
print(f'min_elapsed_time: {min(elapsed_times)}')
print(f'max_elapsed_time: {max(elapsed_times)}')
iterator.finalize()

