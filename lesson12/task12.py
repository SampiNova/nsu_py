import numpy as np
import matplotlib.pyplot as plt
from torch.distributed import new_subgroups


class Neuron:
    def __init__(self, weights, bias, function=(lambda x: x), dfunction=(lambda x: 1)):
        self._func = function
        self._dfunc = dfunction
        self._weights = weights
        self._bias = bias
        self._in = None
        self._out = None
        self._delta = None
        self.dw = None
        self.db = None

    def forward(self, X):
        self._in = X
        self._out = self._func(np.dot(self._weights, self._in) + self._bias)
        return self._out

    def backward(self, loss):
        self.db = loss * self._dfunc(np.dot(self._weights, self._in) + self._bias)
        self.dw = self.db @ self._in
        return self.db


    def update_weights(self, lr):
        self._weights -= lr * self.dw
        self._bias -= lr * self.db


class Model:
    def __init__(self, layers, neurons_on_layer, function=(lambda x: x), dfunction=(lambda x: 1)):
        self._layers = []
        for i in range(1, layers):
            layer = []
            for j in range(neurons_on_layer[i]):
                w = np.random.uniform(0.0, 1.0, size=neurons_on_layer[i - 1])
                layer.append(Neuron(w, np.random.uniform(0.0, 1.0), function, dfunction))
            self._layers.append(layer)

    def forward(self, X):
        last = X
        for layer in self._layers:
            new_last = []
            for neuron in layer:
                new_last.append(neuron.forward(last))
            last = np.array(new_last)
        return last

    def backward(self, loss, lr):
        last_loss = loss
        for layer in self._layers[::-1]:
            new_loss = []
            for neuron in layer:
                delt = neuron.backward(last_loss)
                new_loss.append(delt)
            last_loss = np.array(new_loss)
        for layer in self._layers:
            for neuron in layer:
                neuron.update_weights(lr)


sigmoid = (lambda x: 1 / (1 + np.exp(-x)))
dsigmoid = (lambda x: sigmoid(x) / (1 - sigmoid(x)))

error = (lambda x, y: np.sum((x - y) ** 2))

model = Model(3, [2, 2, 1], sigmoid, dsigmoid)

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
label = np.array([0, 1, 1, 0], dtype=np.float64)

lam = 0.001

res = []
for _ in range(10000):
    accur = 0
    for i in range(len(x)):
        y = model.forward(x[i])
        err = error(y, label[i])
        model.backward(err, lam)
        accur += label[i] - y
    res.append(accur / len(x))

for xi in x:
    y = model.forward(xi)
    print(y)

plt.plot(list(range(len(res))), res)
plt.show()
