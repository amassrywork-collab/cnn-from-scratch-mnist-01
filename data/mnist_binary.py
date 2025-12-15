import numpy as np
from sklearn.datasets import fetch_openml


def load_mnist_binary(limit_per_class=100):
    """
    Loads MNIST dataset and keeps only digits 0 and 1.
    A small subset is used to reduce training time.
    """

    # Load MNIST from OpenML
    X, y = fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True,
        as_frame=False
    )

    y = y.astype(int)

    # Select only digits 0 and 1
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    # Limit number of samples per class
    zero_idx = np.where(y == 0)[0][:limit_per_class]
    one_idx = np.where(y == 1)[0][:limit_per_class]
    indices = np.concatenate([zero_idx, one_idx])

    X = X[indices]
    y = y[indices]

    # Shuffle data
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Reshape images to (28, 28)
    X = X.reshape(-1, 28, 28)

    return X, y
