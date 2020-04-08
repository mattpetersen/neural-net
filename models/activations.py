from typing import Callable

import numpy as np


# Boolean determines whether to calculate derivate instead
Activation = Callable[[np.array, bool], np.array]


def linear(x: np.array, d: bool = False) -> np.array:
    if d: return np.ones(x.shape)
    else: return x


def relu(x: np.array, d: bool = False) -> np.array:
    z = x > 0
    if d: return z
    else: return z * x


def sigmoid(x: np.array, d: bool = False) -> np.array:
    z = 1 / (1 + np.exp(-x))
    if d: return z * (1 - z)
    else: return z


def softmax(x: np.array) -> np.array:
    """Apply softmax independently to each row."""
    z = np.exp(x - x.max(1)[:, None])
    return z / z.sum(1)[:, None]


def tanh(x: np.array, d: bool = False) -> np.array:
    z = np.tanh(x)
    if d: return 1 - (z ** 2)
    else: return z
