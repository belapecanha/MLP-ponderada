import numpy as np
from .activations import relu, relu_deriv, sigmoid, sigmoid_deriv, softmax
from .losses import cross_entropy


class MLP:
    def __init__(self, layer_sizes, activation='relu'):
        self.activation = activation
        self.params = {}
        self.num_layers = len(layer_sizes) - 1
        for i in range(self.num_layers):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            self.params[f'W{i}'] = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            self.params[f'b{i}'] = np.zeros((1, n_out))

    def forward(self, X):
        self.cache = {'A0': X}  
        for i in range(self.num_layers):
            A_prev = self.cache[f'A{i}']
            Z = A_prev @ self.params[f'W{i}'] + self.params[f'b{i}']
            if i == self.num_layers - 1:
                A = softmax(Z)
            elif self.activation == 'relu':
                A = relu(Z)
            else:
                A = sigmoid(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i + 1}'] = A
        return self.cache[f'A{self.num_layers}']

    def loss(self, Y_pred, Y_true):
        return cross_entropy(Y_pred, Y_true)

    def accuracy(self, Y_pred, Y_true):
        pred = np.argmax(Y_pred, axis=1)
        true = np.argmax(Y_true, axis=1)
        return np.mean(pred == true)

    def backward(self, Y_true):
        N = Y_true.shape[0]
        self.grads = {}
        dA = (self.cache[f'A{self.num_layers}'] - Y_true) / N
        for i in reversed(range(self.num_layers)):
            A_prev = self.cache[f'A{i}']
            Z      = self.cache[f'Z{i}']
            if i == self.num_layers - 1:
                dZ = dA
            elif self.activation == 'relu':
                dZ = dA * relu_deriv(Z)
            else:
                dZ = dA * sigmoid_deriv(Z)
            self.grads[f'dW{i}'] = A_prev.T @ dZ
            self.grads[f'db{i}'] = np.sum(dZ, axis=0, keepdims=True)
            dA = dZ @ self.params[f'W{i}'].T

    def update(self, lr):
        for i in range(self.num_layers):
            self.params[f'W{i}'] -= lr * self.grads[f'dW{i}']
            self.params[f'b{i}'] -= lr * self.grads[f'db{i}']
