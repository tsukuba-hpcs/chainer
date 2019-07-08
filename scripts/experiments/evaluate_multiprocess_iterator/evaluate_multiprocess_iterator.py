import argparse
import time
import sys
import chainer
import numpy as np
import multiprocessing

from chainer.iterators.multiprocess_iterator import MultiprocessIterator
from chainer.datasets.image_dataset import LabeledImageDataset


def main():
    multiprocessing.set_start_method('forkserver')

    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=100)
    parser.add_argument('--train', type=str, required=True)
    parser.add_argument('--root', type=str, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--n_processes', type=int, required=True)
    parser.add_argument('--n_prefetch', type=int, required=True)

    args = parser.parse_args()

    dataset = LabeledImageDataset(
        pairs=args.train,
        root=args.root
    )

    iterator = MultiprocessIterator(
        dataset=dataset,
        batch_size=args.batch_size,
        n_processes=args.n_processes,
        n_prefetch=args.n_prefetch
    )

    s = time.time()
    elapsed_times = []
    for i in range(args.count):
        s_iteration = time.time()
        # sys.stderr.write(f'{i}/{args.count}\r')
        # sys.stderr.flush()
        data = iterator.__next__()
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
    print(elapsed_time)

    '''
    elapsed_times = np.array(elapsed_times)
    print(
        f'min: {elapsed_times.min()},',
        f'max: {elapsed_times.max()},',
        f'mean: {elapsed_times.mean()},',
        f'median: {np.median(elapsed_times)},',
        f'var: {elapsed_times.var()}'
    )
    '''
    iterator.finalize()

if __name__ == '__main__':
    main()

