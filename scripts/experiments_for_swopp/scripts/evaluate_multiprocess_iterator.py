import argparse
import multiprocessing
import sys
import time

from chainer.datasets.image_dataset import LabeledImageDataset
from chainer.iterators.multiprocess_iterator import MultiprocessIterator


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
    print('total', elapsed_time, file=sys.stdout)
    print('get_example_time', dataset.get_example_time, file=sys.stdout)
    print('file_open_and_read_time', dataset.file_open_and_read_time, file=sys.stdout)
    sys.stdout.flush()
    iterator.finalize()


if __name__ == '__main__':
    main()
