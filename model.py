from layers.conv import ConvolutionalLayer
from layers.dense import Dense
from layers.reshape import Reshape
from layers.activations import Sigmoid


def build_model():
    return [
        ConvolutionalLayer(input_shape=(1, 28, 28), kernel_size=3, depth=5),
        Sigmoid(),
        Reshape((5, 26, 26), (5 * 26 * 26, 1)),
        Dense(5 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 2),
        Sigmoid()
    ]