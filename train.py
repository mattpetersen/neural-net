from typing import List

import click
# pip install mnist-py
from mnist import MNIST

from models.layers import *
from models.network import Network


@click.command()
@click.option(
    '-e', '--epochs',
    help='Number of epochs to train for',
    type=click.IntRange(1, None),
    default=7,
)
@click.option(
    '-b', '--batch-size',
    help='Number of training examples each step',
    type=click.IntRange(1, None),
    default=100,
)
@click.option(
    '-r', '--learning-rate',
    help='Vanilla gradient descent learning rate',
    type=click.FloatRange(0, None),
    default=1e-1,
)
@click.option(
    '-l1', '--l1-coef',
    help='Linear penalty on size of each weight matrix',
    type=click.FloatRange(0, None),
    default=1e-2,
)
@click.option(
    '-l2', '--l2-coef',
    help='Squared penalty on size of each weight matrix',
    type=click.FloatRange(0, None),
    default=1e-2,
)
def main(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    l1_coef: float,
    l2_coef: float,
) -> None:

    net = Network(
        Sigmoid(784, 800),
        SoftmaxCrossEntropy(800, 10),
    )

    mnist = MNIST()
    for n in range(epochs):

        for images, labels in mnist.train_set.minibatches(batch_size):
            net.train_step(
                data=images,
                labels=labels,
                learning_rate=learning_rate,
                l1_coef=l1_coef,
                l2_coef=l2_coef,
            )

        preds = net.forward(mnist.train_set.images)
        epoch_acc = np.mean(preds.argmax(1) == mnist.train_set.labels.argmax(1))
        print(f'Epoch {n + 1} / {epochs} train accuracy: {epoch_acc.round(2)}')

    preds = net.forward(mnist.test_set.images)
    test_acc = np.mean(preds.argmax(1) == mnist.test_set.labels.argmax(1))
    print(f'Test accuracy: {test_acc.round(2)}')


if __name__ == '__main__':
    main()
