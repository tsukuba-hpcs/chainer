#!/usr/bin/env python3

import argparse
import gzip
import pathlib
import time

import numpy as np

import xchainer as xc


class MLP:

    def __init__(self):
        self.W1, self.b1 = new_linear_params(784, 1000)
        self.W2, self.b2 = new_linear_params(1000, 1000)
        self.W3, self.b3 = new_linear_params(1000, 10)

    @property
    def params(self):
        return self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

    def forward(self, x):
        h = xc.maximum(0, x.dot(self.W1) + self.b1)
        h = xc.maximum(0, h.dot(self.W2) + self.b2)
        return h.dot(self.W3) + self.b3

    def update(self, lr):
        for param in self.params:
            # TODO(beam2d): make it easier
            p = param.as_grad_stopped()
            p -= lr * param.grad.as_grad_stopped()  # TODO(beam2d): make grad not have graph by default
            param.cleargrad()

    def no_grad(self):
        # TODO(beam2d): implement a mode to not create a graph
        self.W1 = self.W1.as_grad_stopped()
        self.b1 = self.b1.as_grad_stopped()
        self.W2 = self.W2.as_grad_stopped()
        self.b2 = self.b2.as_grad_stopped()
        self.W3 = self.W3.as_grad_stopped()
        self.b3 = self.b3.as_grad_stopped()

    def require_grad(self):
        for param in self.params:
            param.require_grad()


def new_linear_params(n_in, n_out):
    W = np.random.randn(n_in, n_out).astype(np.float32)  # TODO(beam2d): not supported in xc
    W /= np.sqrt(n_in)  # TODO(beam2d): not supported in xc
    W = xc.array(W)
    # TODO(beam2d): make zeros accept int as shape
    b = xc.zeros((n_out,), dtype=xc.float32)
    return W, b


def compute_loss(y, t):
    # softmax cross entropy
    score = xc.log_softmax(y, axis=1)
    mask = (t[:, xc.newaxis] == xc.arange(10, dtype=t.dtype)).astype(score.dtype)
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

        y = model.forward(x)
        total_loss += compute_loss(y, t) * batch_size
        num_correct += (y.argmax(axis=1).astype(t.dtype) == t).astype(xc.int32).sum()

    model.require_grad()

    mean_loss = float(total_loss) / N_test
    accuracy = int(num_correct) / N_test
    return mean_loss, accuracy


def main():
    parser = argparse.ArgumentParser('Train a neural network on MNIST dataset')
    parser.add_argument('--batchsize', '-B', type=int, default=100, help='Batch size')
    parser.add_argument('--epoch', '-E', type=int, default=20, help='Number of epochs to train')
    parser.add_argument('--iteration', '-I', type=int, default=None, help='Number of iterations to train. Epoch is ignored if specified.')
    parser.add_argument('--data', '-p', default='mnist',
                        help='Path to the directory that contains MNIST dataset')
    parser.add_argument('--device', '-d', default='native', help='Device to use')
    parser.add_argument('--eval-size', default=None, type=int,
                        help='Number of samples to use from the test set for evaluation. None to use all.')
    args = parser.parse_args()

    xc.set_default_device(args.device)

    # Prepare dataset
    X, Y = get_mnist(args.data, 'train')
    X_test, Y_test = get_mnist(args.data, 't10k')

    # Prepare model
    model = MLP()

    # Training
    N = X.shape[0]   # TODO(beam2d): implement len
    all_indices_np = np.arange(N, dtype=np.int64)  # TODO(beam2d): support int32 indexing
    batch_size = args.batchsize
    eval_size = args.eval_size

    # Train
    model.require_grad()

    it = 0
    epoch = 0
    is_finished = False
    start = time.time()

    while not is_finished:
        np.random.shuffle(all_indices_np)  # TODO(beam2d): not suupported in xc
        all_indices = xc.array(all_indices_np)

        for i in range(0, N, batch_size):
            indices = all_indices[i:i + batch_size]
            x = X.take(indices, axis=0)
            t = Y.take(indices, axis=0)

            y = model.forward(x)
            loss = compute_loss(y, t)

            loss.backward()
            model.update(lr=0.01)

            it += 1
            if args.iteration is not None:
                mean_loss, accuracy = evaluate(model, X_test, Y_test, eval_size, batch_size)
                elapsed_time = time.time() - start
                print(f'iteration {it}... loss={mean_loss},\taccuracy={accuracy},\telapsed_time={elapsed_time}')
                if it >= args.iteration:
                    is_finished = True
                    break

        epoch += 1
        if args.iteration is None:  # stop based on epoch, instead of iteration
            mean_loss, accuracy = evaluate(model, X_test, Y_test, eval_size, batch_size)
            elapsed_time = time.time() - start
            print(f'epoch {epoch}... loss={mean_loss},\taccuracy={accuracy},\telapsed_time={elapsed_time}')
            if epoch >= args.epoch:
                is_finished = True


def get_mnist(path, name):
    path = pathlib.Path(path)
    x_path = path / f'{name}-images-idx3-ubyte.gz'
    y_path = path / f'{name}-labels-idx1-ubyte.gz'

    with gzip.open(x_path, 'rb') as fx:
        fx.read(16)  # skip header
        # read/frombuffer is used instead of fromfile because fromfile does not handle gzip file correctly
        x = np.frombuffer(fx.read(), dtype=np.uint8).reshape(-1, 784)
        x.flags.writeable = True  # TODO(beam2d): remove this workaround

    with gzip.open(y_path, 'rb') as fy:
        fy.read(8)  # skip header
        y = np.frombuffer(fy.read(), dtype=np.uint8)
        y.flags.writeable = True  # TODO(beam2d): remove this workaround

    assert x.shape[0] == y.shape[0]

    x = x.astype(np.float32)
    x /= 255
    y = y.astype(np.int32)
    return xc.array(x), xc.array(y)


if __name__ == '__main__':
    main()
