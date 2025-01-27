import argparse
import time
import sys
import chainer
import numpy as np
import multiprocessing

from chainer.datasets.image_dataset import LabeledImageDataset
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
    parser.add_argument('--n_remove_example', type=int, required=True)

    args = parser.parse_args()
    # print(vars(args))

    dataset = ExtendedLabeledImageDataset(
        pairs=args.train,
        root=args.root
    )
    
    print(len(dataset))
    
    return 0

    iterator = PrefetchMultiprocessIterator(
        dataset=dataset,
        batch_size=args.batch_size,
        local_storage_base=args.local_storage_base,
        n_prefetch=args.n_prefetch,
        n_prefetch_from_backend=args.n_prefetch_from_backend,
        n_generate_batch=args.n_generate_batch,
        n_remove_example=args.n_remove_example
    )

    return 0

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
	    f'current: {i}, first img label in batch: {data[0][1]}',
            file=sys.stderr
        )
        '''
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
    print(elapsed_time, file=sys.stdout)
    sys.stdout.flush()
    iterator.finalize()

if __name__ == '__main__':
    main()

