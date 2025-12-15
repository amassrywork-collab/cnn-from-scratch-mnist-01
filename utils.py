import numpy as np


def normalize(images):
    return images / 255.0


def one_hot(label, num_classes=2):
    vec = np.zeros((num_classes, 1))
    vec[label] = 1
    return vec