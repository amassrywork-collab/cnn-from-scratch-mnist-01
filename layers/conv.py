import numpy as np
from scipy.signal import correlate, convolve
from .base import Layer


class ConvolutionalLayer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
        input_shape: (input_depth, input_height, input_width)
        kernel_size: int
        depth: number of kernels (output depth)
        """
        self.input_depth, self.input_height, self.input_width = input_shape
        self.kernel_size = kernel_size
        self.depth = depth

        self.output_height = self.input_height - kernel_size + 1
        self.output_width = self.input_width - kernel_size + 1

        self.kernels = np.random.randn(
            depth, self.input_depth, kernel_size, kernel_size
        )
        self.biases = np.random.randn(
            depth, self.output_height, self.output_width
        )

    def forward(self, input):
        self.input = input
        self.output = np.zeros((self.depth, self.output_height, self.output_width))

        for d in range(self.depth):
            for i in range(self.input_depth):
                self.output[d] += correlate(
                    input[i],
                    self.kernels[d, i],
                    mode="valid"
                )
            self.output[d] += self.biases[d]

        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels.shape)
        input_gradient = np.zeros(self.input.shape)

        for d in range(self.depth):
            for i in range(self.input_depth):
                kernels_gradient[d, i] = correlate(
                    self.input[i],
                    output_gradient[d],
                    mode="valid"
                )
                input_gradient[i] += convolve(
                    output_gradient[d],
                    self.kernels[d, i],
                    mode="full"
                )

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient