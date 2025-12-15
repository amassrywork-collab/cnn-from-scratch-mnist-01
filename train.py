from model import build_model
from losses.binary_cross_entropy import BinaryCrossEntropy
from utils import normalize, one_hot
from data.mnist_binary import load_mnist_binary


def train():
    model = build_model()
    loss_fn = BinaryCrossEntropy()
    learning_rate = 0.01
    epochs = 5

    X_train, y_train = load_mnist_binary()

    for epoch in range(epochs):
        error = 0

        for x, y in zip(X_train, y_train):
            x = normalize(x).reshape(1, 28, 28)
            y = one_hot(y)

            output = x
            for layer in model:
                output = layer.forward(output)

            error += loss_fn.loss(y, output)

            grad = loss_fn.gradient(y, output)
            for layer in reversed(model):
                grad = layer.backward(grad, learning_rate)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {error / len(X_train)}")


if __name__ == "__main__":
    train()