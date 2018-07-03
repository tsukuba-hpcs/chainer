#!/usr/bin/env python
"""Example code of learning a large scale convnet from ILSVRC2012 dataset.

Prerequisite: To run this example, crop the center of ILSVRC2012 training and
validation images, scale them to 256x256 and convert them to RGB, and make
two lists of space-separated CSV whose first column is full path to image and
second column is zero-origin label (this format is same as that used by Caffe's
ImageDataLayer).

"""

import argparse
import six
import time

import numpy as np

import xchainer as xc

from image_dataset import get_dataset
import resnet50


def compute_loss(y, t):
    # softmax cross entropy
    score = xc.log_softmax(y, axis=1)
    mask = (t[:, xc.newaxis] == xc.arange(1000, dtype=t.dtype)).astype(score.dtype)
    # TODO(beam2d): implement mean
    return -(score * mask).sum() * (1 / y.shape[0])


def evaluate(model, X_test, Y_test, eval_size, batch_size):
    N_test = X_test.shape[0] if eval_size is None else eval_size

    if N_test > X_test.shape[0]:
        raise ValueError(f'Test size can be no larger than {X_test.shape[0]}')

    model.no_grad()

    # TODO(beam2d): make xc.array(0, dtype=...) work
    total_loss = xc.zeros((), dtype=xc.float32)
    num_correct = xc.zeros((), dtype=xc.int64)
    for i in range(0, N_test, batch_size):
        x = X_test[i:min(i + batch_size, N_test)]
        t = Y_test[i:min(i + batch_size, N_test)]

        y = model(x)
        total_loss += compute_loss(y, t) * batch_size
        num_correct += (y.argmax(axis=1).astype(t.dtype) == t).astype(xc.int32).sum()

    model.require_grad()

    mean_loss = float(total_loss) / N_test
    accuracy = int(num_correct) / N_test
    return mean_loss, accuracy


def main():
    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    # parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
    #                     help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--iteration', '-I', type=int, default=None,
                        help='Number of iterations to train. Epoch is ignored if specified.')
    # parser.add_argument('--gpu', '-g', type=int, default=-1,
    #                     help='GPU ID (negative value indicates CPU')
    # parser.add_argument('--initmodel',
    #                     help='Initialize the model from given file')
    # parser.add_argument('--loaderjob', '-j', type=int,
    #                     help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    # parser.add_argument('--resume', '-r', default='',
    #                     help='Initialize the trainer from given file')
    # parser.add_argument('--out', '-o', default='result',
    #                     help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    # parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--device', '-d', default='native',
                        help='Device to use')

    args = parser.parse_args()

    xc.set_default_device(args.device)

    # Prepare model
    model = resnet50.ResNet50()

    # Prepare datasets and mean file
    mean = np.load(args.mean)
    X, Y = get_dataset(args.train, args.root, mean, model.insize)
    X, Y = xc.array(X), xc.array(Y)
    X_test, Y_test = get_dataset(args.val, args.root, mean, model.insize, False)
    X_test, Y_test = xc.array(X_test), xc.array(Y_test)

    N = X.shape[0]
    all_indices_np = np.arange(N, dtype=np.int64)
    batch_size = args.batchsize
    eval_size = args.val_batchsize

    # Train
    model.require_grad()

    it = 0
    epoch = 0
    is_finished = False
    start = time.time()

    while not is_finished:
        np.random.shuffle(all_indices_np)
        all_indices = xc.array(all_indices_np)

        for i in range(0, N // batch_size):
            indices = all_indices[i * batch_size: (i + 1) * batch_size]
            x = X.take(indices, axis=0)
            t = Y.take(indices, axis=0)
            y = model(x)
            loss = compute_loss(y, t)

            loss.backward()
            model.update(lr=0.01)

            it += 1
            if args.iteration is not None:
                elapsed_time = time.time() - start
                mean_loss, accuracy = evaluate(model, X_test, Y_test, eval_size, batch_size)
                print(f'iteration {it}... loss={mean_loss},\taccuracy={accuracy},\telapsed_time={elapsed_time}')
                if it >= args.iteration:
                    is_finished = True

        epoch += 1
        if args.iteration is None:
            elapsed_time = time.time() - start
            mean_loss, accuracy = evaluate(model, X_test, Y_test, eval_size, batch_size)
            print(f'epoch {epoch}... loss={mean_loss},\taccuracy={accuracy},\telapsed_time={elapsed_time}')
            if epoch >= args.epoch:
                is_finished = True


if __name__ == '__main__':
    main()
