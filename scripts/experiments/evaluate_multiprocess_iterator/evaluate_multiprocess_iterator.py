import argparse
import time
import sys
import chainer

from chainer.iterators.multiprocess_iterator import MultiprocessIterator
from chainer.datasets.image_dataset import LabeledImageDataset

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
for i in range(args.count):
    # sys.stderr.write(f'{i}/{args.count}\r')
    # sys.stderr.flush()
    iterator.__next__()
elapsed_time = time.time() - s
sys.stderr.write('\n')
print(elapsed_time)
iterator.finalize()

