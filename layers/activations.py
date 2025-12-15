import numpy as np
from .base import Layer


class Sigmoid(Layer):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.output * (1 - self.output))