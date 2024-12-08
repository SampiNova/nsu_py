import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

def reshape(array):
    array = np.asarray(array)
    if len(array.shape) < 2:
        return np.expand_dims(array, 0)
    return array

def xavier_normal(shape, gain=1.0):
  fan_in, fan_out = shape[0], shape[1]
  stddev = gain * np.sqrt(2.0 / (fan_in + fan_out))
  return np.random.normal(0, stddev, size=shape)


error = (lambda x, y: (np.asarray(x) - np.asarray(y)) ** 2)
derror = (lambda x, y: (np.asarray(x) - np.asarray(y)) * 2)

class Neuron:
    def __init__(self, input_weights, bias):
        self.weights = np.asarray(input_weights)
        self.bias = np.asarray(bias)
        self.f = (lambda x: 1 / (1 + np.exp(-x)))
        self.df = (lambda x: self.f(x) * (1 - self.f(x)))

        self.X = None
        self.db = None
        self.dw = None
        self.t = None
        self.h = None

    def forward(self, X):
        self.X = reshape(X)
        self.t = self.X @ self.weights + self.bias
        self.h = self.f(self.t)
        return self.h

    def backprop(self, dEdh):
        self.db = dEdh * self.df(self.t)[0]
        # print("db:", self.db)
        self.dw = self.X.T @ self.db
        self.dw = reshape(self.dw).T
        # print("dw:", self.dw)
        return self.db @ self.weights.T

    def update(self, mu):
        self.bias -= mu * self.db
        self.weights -= mu * self.dw


class Model:
    def __init__(self, layers):
        self.layers = []
        for i in range(1, len(layers)):
            neurons = []
            for j in range(layers[i]):
                neurons.append(Neuron(xavier_normal((layers[i - 1], 1)), xavier_normal((1, 1))))
            self.layers.append(neurons)

    def forward(self, X):
        last = np.asarray(X)
        for layer in self.layers:
            last = np.asarray([neuron.forward(last)[0] for neuron in layer]).T
        return last

    def backprop(self, dEdh):
        last_dx = np.asarray(dEdh)
        for layer in self.layers[::-1]:
            new_dx = np.asarray([neuron.backprop(dx) for neuron, dx in zip(layer, last_dx)])
            # print(new_dx)
            last_dx = np.sum(new_dx, 0)
            # print(last_dx)

    def update(self, mu):
        for layer in self.layers:
            for neuron in layer:
                neuron.update(mu)


myX = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
myY = np.asarray([0, 1, 1, 0], dtype=np.float64)
size = len(myY)

lam = 0.01
epoch = 100000

model = Model([2, 2, 1])

test = []
for _ in range(epoch):
    for i in range(size):
        pred = model.forward(myX[i])
        model.backprop(derror(pred, myY[i]))
        model.update(lam)
    temp = []
    for i in range(size):
        pred = model.forward(myX[i])
        temp.append(error(pred, myY[i])[0, 0])
    test.append(sum(temp) / size)

for layer in model.layers:
    for neuron in layer:


my_pred = []
for obj in myX:
    pred = model.forward(obj)[0][0]
    my_pred.append(pred)
    print(pred)
my_pred = np.round(np.asarray(my_pred))

print(accuracy_score(myY, my_pred))

X = list(range(1, len(test) + 1))
plt.plot(X, test)
plt.show()
