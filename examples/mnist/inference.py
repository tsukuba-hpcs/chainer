import argparse

import chainer
from train_mnist import MLP


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--snapshot', '-s', default='result/',
                        help='The path to a saved snapshot (NPZ)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# unit: {}'.format(args.unit))
    print('')

    # Create a same model object as what you used for training
    model = MLP(args.unit, 10)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)

    # Load saved parameters from a NPZ file of the Trainer object
    chainer.serializers.load_npz(
        args.snapshot, model, path='updater/model:main/predictor/')

    # Prepare data
    train, test = chainer.datasets.get_mnist()
    x, answer = test[0]
    if args.gpu >= 0:
        x = chainer.cuda.cupy.asarray(x)
    prediction = model(x[None, ...])[0].array.argmax()

    print('Prediction:', prediction)
    print('Answer:', answer)


if __name__ == '__main__':
    main()
