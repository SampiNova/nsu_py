import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, input_weights, bias):
        self.weights = np.asarray(input_weights)
        self.bias =np.asarray(bias)
        self.f = (lambda x: 1 / (1 + np.exp(-x)))
        self.df = (lambda x: self.f(x) * (1 - self.f(x)))

        self.db = None
        self.dw = None
        self.t = None
        self.h = None

    def forward(self, X):
        self.t = np.asarray(X) @ self.weights + self.bias
        self.h = self.f(self.t)
        return self.h

    def backprop(self, X, dEdh):
        self.db = np.asarray(dEdh) * self.df(self.t)  # sEdb
        self.dw = self.db * np.asarray(X)
        return self.db

    def update(self, mu):
        self.bias -= mu * self.db
        self.weights -= mu * self.dw

first_neuron = Neuron(np.random.uniform(0.0, 1.0, 2), np.random.uniform(0.0, 1.0))

dataX = [[0, 0], [0, 1], [1, 0], [1, 1]]
dataY = [0, 1, 1, 0]
lam = 0.01

error = (lambda x, y: np.sum((np.asarray(x) - np.asarray(y)) ** 2))

res = list()
for i in range(100):
    for xj, yj in zip(dataX, dataY):
        pred = first_neuron.forward(xj)
        dEdt = first_neuron.backprop(xj, error(pred, yj))
        dEdx = np.asarray(dEdt) @ first_neuron.weights
        first_neuron.update(lam)
    pred = sum([error(xi, yi) for xi, yi in zip(dataX, dataY)]) / 4
    res.append(pred)

plt.plot(list(range(len(res))), res)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, input_weights, bias):
        self.weights = np.asarray(input_weights)
        self.bias =np.asarray(bias)
        self.f = (lambda x: x)
        self.df = (lambda x: x ** 0)

        self.db = None
        self.dw = None
        self.t = None
        self.h = None

    def forward(self, X):
        self.t = np.asarray(X) @ self.weights + self.bias
        self.h = self.f(self.t)
        return self.h

    def backprop(self, X, dEdh):
        self.db = np.asarray(dEdh) * self.df(self.t)  # sEdb
        self.dw = self.db * np.asarray(X)
        return self.db

    def update(self, mu):
        self.bias -= mu * self.db
        self.weights -= mu * self.dw

first_neuron = Neuron(np.random.uniform(0.0, 1.0, 2), np.random.uniform(0.0, 1.0))

dataX = [[0, 0], [0, 1], [1, 0], [1, 1]]
dataY = [0, 1, 1, 0]
lam = 0.01

error = (lambda x, y: (np.asarray(x) - np.asarray(y)) ** 2)
derror = (lambda x, y: 2 * (np.asarray(x) - np.asarray(y)))

res = list()
for i in range(100):
    for xj, yj in zip(dataX, dataY):
        pred = first_neuron.forward(xj)
        dEdt = first_neuron.backprop(xj, derror(pred, yj))
        # dEdx = np.asarray(dEdt) @ first_neuron.weights
        first_neuron.update(lam)
    pred = np.array([first_neuron.forward(xj) for xj in dataX])
    mse = np.mean(error(pred, dataY))
    res.append(mse)

plt.plot(list(range(len(res))), res)
plt.show()
