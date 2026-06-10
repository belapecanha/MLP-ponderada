import numpy as np


def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)
