import argparse
import multiprocessing
import sys
import time

import numpy as np

from chainer.datasets.image_dataset import LabeledImageDataset
from chainer.iterators.multiprocess_iterator import MultiprocessIterator


def desc(data: np.ndarray):
    print(f'n: {data.size}, sum: {data.sum()},'
          f'min: {data.min()}, max: {data.max()},'
          f'mean: {data.mean()}, median: {np.median(data)},'
          f'var: {data.var()}')


def main():
    multiprocessing.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_processes', type=int, required=True)
    parser.add_argument('--n_prefetch', type=int, required=True)
    parser.add_argument('--n_remove_example', type=int)

    args = parser.parse_args()

    dataset = LabeledImageDataset(
        pairs=args.train,
        root=args.root,
        measure=True
    )

    iterator = MultiprocessIterator(
        dataset=dataset,
        batch_size=args.batch_size,
        n_processes=args.n_processes,
        n_prefetch=args.n_prefetch,
        measure=True
    )

    # warming up
    _data = iterator.__next__()

    s = time.time()
    elapsed_times = []
    for i in range(args.count):
        s_iteration = time.time()
        # sys.stderr.write(f'{i}/{args.count}\r')
        # sys.stderr.flush()
        _data = iterator.__next__()
        iteration_elapsed_time = time.time() - s_iteration
        elapsed_times.append(iteration_elapsed_time)
        '''
        print(
            f'label: {data[0][1]}',
            file=sys.stderr
        )
        '''
    elapsed_time = time.time() - s
    # sys.stderr.write('\n')
    # print(elapsed_times)
    print('total', elapsed_time)
    print('task_time', iterator.task_time)
    print('task_count', iterator.task_count)
    print('fetch_data_time', iterator.fetch_data_time)
    print('unpack_and_organize_batch_time', iterator.unpack_and_organize_batch_time)
    print('launch_thread_time', iterator.launch_thread_time)
    print('generate_batch_pool_time', iterator.generate_batch_pool_time)
    print('generate_batch_thread_and_start_time', iterator.generate_batch_thread_and_start_time)

    fetch_run_times = np.array(iterator.fetch_run_times)
    get_example_times = np.array(iterator.get_example_times)
    read_data_times = np.array(iterator.read_data_times)

    print('fetch_run_times')
    desc(fetch_run_times)
    print('get_example_times')
    desc(get_example_times)
    print('read_data_times')
    desc(read_data_times)

    sys.stdout.flush()
    iterator.finalize()


if __name__ == '__main__':
    main()
