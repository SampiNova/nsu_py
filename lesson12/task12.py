import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, input_weights, bias):
        self.weights = np.asarray(input_weights)
        self.bias = bias

    def forward(self, X):
        return np.asarray(X) @ self.weights
