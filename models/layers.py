from dataclasses import dataclass

import numpy as np

from models.activations import Activation, linear, relu, sigmoid, softmax, tanh


@dataclass
class Layer:
    n_rows: int
    n_cols: int
    activation: Activation

    def __post_init__(self) -> None:
        self.shape = self.n_rows, self.n_cols
        self.w = np.random.randn(*self.shape) * np.sqrt(2 / self.shape[0])

    def forward(self, x: np.array) -> np.array:
        # Store input to this layer for updating weights later
        self.x = x
        # Store input to activation for backward pass later
        self.z = x.dot(self.w)
        return self.activation(self.z)

    def backward(self, da: np.array) -> np.array:
        # Bring error backward through activation function
        self.dz = da * self.activation(self.z, d=True)
        # Bring error backward through matrix multiplication
        return self.dz.dot(self.w.T)

    def update(self,
        learning_rate: float,
        l1_coef: float,
        l2_coef: float,
    ) -> None:

        self.w -= learning_rate * self.x.T.dot(self.dz)
        self.w -= l1_coef * np.sign(self.w)
        self.w -= l2_coef * np.abs(self.w)


@dataclass
class Linear(Layer):
    activation: Activation = linear


@dataclass
class Relu(Layer):
    activation: Activation = relu


@dataclass
class Sigmoid(Layer):
    activation: Activation = sigmoid


@dataclass
class Tanh(Layer):
    activation: Activation = tanh


@dataclass
class SoftmaxCrossEntropy(Layer):
    activation: Activation = softmax

    def forward(self, x: np.array) -> np.array:
        self.predicted_labels = super().forward(x)
        return self.predicted_labels

    def backward(self, actual_labels: np.array) -> np.array:
        self.dz = self.predicted_labels - actual_labels
        return self.dz.dot(self.w.T)
