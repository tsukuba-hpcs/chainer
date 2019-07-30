import argparse
import multiprocessing
import sys
import time

import numpy as np

from chainer.datasets.image_dataset import ExtendedLabeledImageDataset
from chainer.iterators.prefetch_multiprocess_iterator import PrefetchMultiprocessIterator


def main():
    multiprocessing.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--local_storage_base', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_prefetch', type=int, required=True)
    parser.add_argument('--n_prefetch_from_backend', type=int, required=True)
    parser.add_argument('--n_generate_batch', type=int, required=True)

    args = parser.parse_args()
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
        n_remove_example=1
    )

    # warming up
    _data = iterator.__next__()
    iterator.dataset.reset_all_timers()

    s = time.time()
    elapsed_times = []
    for i in range(args.count):
        s_iteration = time.time()
        # sys.stderr.write(f'{i}/{args.count}\r')
        # sys.stderr.flush()
        _data = iterator.__next__()
        iteration_elapsed_time = time.time() - s_iteration
        elapsed_times.append(iteration_elapsed_time)

    elapsed_time = time.time() - s
    # sys.stderr.write('\n')
    elapsed_times = np.array(elapsed_times)
    '''
    print(
        f'min: {elapsed_times.min()},',
        f'max: {elapsed_times.max()},',
        f'mean: {elapsed_times.mean()},',
        f'median: {np.median(elapsed_times)},',
        f'var: {elapsed_times.var()}'
    )
    '''
    print('total', elapsed_time, file=sys.stdout, flush=True)
    print('get_example_time', dataset.get_example_time, file=sys.stdout, flush=True)
    print('file_open_and_read_time', dataset.file_open_and_read_time, file=sys.stdout, flush=True)
    print('task_time', iterator.task_time, file=sys.stdout, flush=True)
    print('task_count', iterator.task_count, file=sys.stdout, flush=True)
    print('cached_index_get_time', iterator.cached_index_get_time, file=sys.stdout, flush=True)
    print('fetch_data_time', iterator.fetch_data_time, file=sys.stdout, flush=True)
    print('unpack_and_organize_batch_time', iterator.unpack_and_organize_batch_time, file=sys.stdout, flush=True)
    print('prefetch_time', file=sys.stdout, flush=True)
    prefetch_time = iterator.prefetch_time

    for key in prefetch_time.keys():
        times = np.array(prefetch_time[key])
        print(f'[{key}] n: {times.size} sum: {times.sum()}, min: {times.min()}, '
              f'max: {times.max()}, mean: {times.mean()}, '
              f'median: {np.median(times)}, var: {times.var()}', flush=True)
    iterator.finalize()


if __name__ == '__main__':
    main()
