import numpy as np


def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)  # porta binária: passa o gradiente onde o neurônio estava ativo, bloqueia onde não

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)  

def softmax(z):
    # subtrai o max antes do exp para evitar overflow
    e = np.exp(z - np.max(z, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)  # normaliza por linha: cada exemplo vira distribuição de probabilidade
