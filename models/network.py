from functools import reduce

import numpy as np

from models.layers import Layer


class Network:

    def __init__(self, *layers: Layer) -> None:
        self.layers = layers

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def forward(self, data: np.array) -> np.array:
        return reduce(lambda x, lay: lay.forward(x), self, data)

    def backward(self, labels: np.array) -> None:
        reduce(lambda x, lay: lay.backward(x), reversed(self), labels)

    def update(self, *args, **kwargs) -> None:
        reduce(lambda _, lay: lay.update(*args, **kwargs), self)

    def train_step(self,
        data: np.array,
        labels: np.array,
        *args, **kwargs,
    ) -> None:

        self.forward(data)
        self.backward(labels)
        self.update(*args, **kwargs)
