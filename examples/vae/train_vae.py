#!/usr/bin/env python
"""Chainer example: train a VAE on Binarized MNIST
"""
import argparse
import os

import numpy as np

import chainer
from chainer import training
from chainer.training import extensions

import net


def main():
    parser = argparse.ArgumentParser(description='Chainer example: VAE')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='results',
                        help='Directory to output the result')
    parser.add_argument('--epoch', '-e', default=100, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--dim-z', '-z', default=20, type=int,
                        help='dimention of encoded vector')
    parser.add_argument('--dim-h', default=500, type=int,
                        help='dimention of hidden layer')
    parser.add_argument('--beta', default=1.0, type=float,
                        help='Regularization coefficient for '
                             'the second term of ELBO bound')
    parser.add_argument('--batch-size', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--test', action='store_true',
                        help='Use tiny datasets for quick tests')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# dim z: {}'.format(args.dim_z))
    print('# Minibatch-size: {}'.format(args.batch_size))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Prepare VAE model, defined in net.py
    encoder = net.make_encoder(784, args.dim_z, args.dim_h)
    decoder = net.make_decoder(784, args.dim_z, args.dim_h)
    prior = net.make_prior(args.dim_z, device=args.gpu)
    avg_elbo_loss = net.AvgELBOLoss(encoder, decoder, prior,
                                    beta=args.beta)

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(avg_elbo_loss)

    # Initialize
    if args.initmodel:
        chainer.serializers.load_npz(args.initmodel, avg_elbo_loss)

    # Load the MNIST dataset
    train, test = chainer.datasets.get_mnist(withlabel=False)

    # Binarize dataset
    train[train >= 0.5] = 1.0
    train[train < 0.5] = 0.0
    test[test >= 0.5] = 1.0
    test[test < 0.5] = 0.0

    if args.test:
        train, _ = chainer.datasets.split_dataset(train, 100)
        test, _ = chainer.datasets.split_dataset(test, 100)

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)

    # Set up an updater. StandardUpdater can explicitly specify a loss function
    # used in the training with 'loss_func' option
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer,
        device=args.gpu, loss_func=avg_elbo_loss)

    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    trainer.extend(extensions.Evaluator(
        test_iter, avg_elbo_loss, device=args.gpu))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/rec', 'main/penalty', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()

    # Visualize the results
    def save_images(x, filename):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 3, figsize=(9, 9), dpi=100)
        for ai, xi in zip(ax.flatten(), x):
            ai.imshow(xi.reshape(28, 28))
        fig.savefig(filename)

    encoder.to_cpu()
    decoder.to_cpu()
    prior.to_cpu()
    train_ind = [1, 3, 5, 10, 2, 0, 13, 15, 17]
    x = chainer.Variable(np.asarray(train[train_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = decoder(encoder(x).mean).mean
    save_images(x.data, os.path.join(args.out, 'train'))
    save_images(x1.data, os.path.join(args.out, 'train_reconstructed'))

    test_ind = [3, 2, 1, 18, 4, 8, 11, 17, 61]
    x = chainer.Variable(np.asarray(test[test_ind]))
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        x1 = decoder(encoder(x).mean).mean
    save_images(x.data, os.path.join(args.out, 'test'))
    save_images(x1.data, os.path.join(args.out, 'test_reconstructed'))

    # draw images from randomly sampled z
    z = prior().sample(9)
    x = decoder(z).mean
    save_images(x.data, os.path.join(args.out, 'sampled'))


if __name__ == '__main__':
    main()
