#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import random
import sys

import numpy as np

import chainer
import chainer.cuda
from chainer import training
from chainer.training import extensions

import chainermn


import models.alex as alex
import models.googlenet as googlenet
import models.googlenetbn as googlenetbn
import models.nin as nin
import models.resnet50 as resnet50

# Check Python version if it supports multiprocessing.set_start_method,
# which was introduced in Python 3.4
major, minor, _, _, _ = sys.version_info
if major <= 2 or (major == 3 and minor < 4):
    sys.stderr.write('Error: ImageNet example uses '
                     'chainer.iterators.MultiprocessIterator, '
                     'which works only with Python >= 3.4. \n'
                     'For more details, see '
                     'http://chainermn.readthedocs.io/en/master/'
                     'tutorial/tips_faqs.html#using-multiprocessiterator\n')
    exit(-1)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, dataset, mean, crop_size, random=True):
        self.base = dataset
        self.mean = mean.astype(chainer.get_dtype())
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    # for ExtendedLabeledImageDataset
    @property
    def pairs(self):
        return self.base._pairs

    @property
    def root(self):
        return self.base._root

    def get_example_by_path(self, full_path, int_label):
        crop_size = self.crop_size
        image, label = self.base.get_example_by_path(full_path, int_label)
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image.astype(np.float16), label


    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image.astype(np.float16), label


# chainermn.create_multi_node_evaluator can be also used with user customized
# evaluator classes that inherit chainer.training.extensions.Evaluator.
class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    # Check if GPU is available
    # (ImageNet example does not support CPU execution)
    if not chainer.cuda.available:
        raise RuntimeError('ImageNet requires GPU support.')

    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--communicator', default='hierarchical')
    parser.set_defaults(test=False)
    parser.add_argument('--n_prefetch', type=int, required=True)
    parser.add_argument('--iterator', type=str, required=True)
    parser.add_argument('--local_storage_base', type=str)
    parser.add_argument('--prefetchjob', type=int)
    args = parser.parse_args()

    # Start method of multiprocessing module need to be changed if we
    # are using InfiniBand and MultiprocessIterator. This is because
    # processes often crash when calling fork if they are using
    # Infiniband.  (c.f.,
    # https://www.open-mpi.org/faq/?category=tuning#fork-warning )
    # Also, just setting the start method does not seem to be
    # sufficient to actually launch the forkserver processes, so also
    # start a dummy process.
    # See also our document:
    # https://chainermn.readthedocs.io/en/stable/tutorial/tips_faqs.html#using-multiprocessiterator
    # This must be done *before* ``chainermn.create_communicator``!!!
    multiprocessing.set_start_method('forkserver')
    p = multiprocessing.Process()
    p.start()
    p.join()

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print(f'This script is {__file__}')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Using {} iterator'.format(args.iterator))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('Num n_prefetch: {}'.format(args.n_prefetch))
        print('Num prefetchjob: {}'.format(args.prefetchjob))
        print('Num loaderjob: {}'.format(args.loaderjob))
        print('train: {}'.format(args.train))
        print('val: {}'.format(args.val))
        print('root: {}'.format(args.root))
        print('mean: {}'.format(args.mean))
        print('out: {}'.format(args.out))
        print('==========================================')

    # Optimization
    # make cuDNN workspace bigger
    chainer.cuda.set_max_workspace_size(512*1024*1024)
    # use autotune
    chainer.config.autotune = True
    # use cudnn fast BN
    chainer.config.cudnn_fast_batch_normalization = True
    # use always float16
    chainer.config.dtype = np.float16

    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
    model.to_gpu()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(args.mean)
    if comm.rank == 0:
        base_dataset = chainer.datasets.ExtendedLabeledImageDataset(args.train, args.root)
        train = PreprocessedDataset(base_dataset, mean, model.insize)
        val = PreprocessedDataset(base_dataset, mean, model.insize, False)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset_extended(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm)


    if args.iterator == 'prefetch_multiprocess':
        train_iter = chainer.iterators.PrefetchMultiprocessIterator(
            dataset=train, 
            batch_size=args.batchsize, 
            local_storage_base=args.local_storage_base,
            n_prefetch=args.n_prefetch,
            n_prefetch_from_backend=args.prefetchjob,
            n_generate_batch=args.loaderjob,
            dataset_start = train.start,
            dataset_finish = train.finish
        )
    else:
        # A workaround for processes crash should be done before making
        # communicator above, when using fork (e.g. MultiProcessIterator)
        # along with Infiniband.
        train_iter = chainer.iterators.MultiprocessIterator(
                train, args.batchsize, n_processes=args.loaderjob)
        
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9), 
        comm, 
        measure=True,
        double_buffering=True)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=device, use_chainermn=True)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out, use_chainermn=True)

    checkpoint_interval = (10, 'iteration') if args.test else (1, 'epoch')
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')

    # checkpointer = chainermn.create_multi_node_checkpointer(
    #    name='imagenet-example', comm=comm)
    # checkpointer.maybe_load(trainer, optimizer)
    # trainer.extend(checkpointer, trigger=checkpoint_interval)

    # Create a multi node evaluator from an evaluator.
    # evaluator = TestModeEvaluator(val_iter, model, device=device)
    # evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    # trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        # trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        # trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]), trigger=(1000, 'iteration'))
        trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run(show_loop_exception_msg=True)


if __name__ == '__main__':
    main()
