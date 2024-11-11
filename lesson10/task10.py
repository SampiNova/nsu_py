import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE


def encode_label(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

class Data:
    def __init__(self, _data):
        self.data = _data
        self.prep_X_by_digits = [[] for _ in range(10)]
        self.prep_X = []
        self.prep_Y = []
        self.average_digits = [[] for _ in range(10)]
        self.shape()

    def shape(self):
        for elem in self.data:
            tmp = (np.reshape(elem[0][0].numpy(), (784,)), encode_label(elem[1]))
            self.prep_X.append(tmp[0])
            self.prep_X_by_digits[elem[1]].append(tmp[0])
            self.prep_Y.append(tmp[1])
            self.average_digits[elem[1]].append(tmp[0])
        new_avgs = []
        for i in range(10):
            new_avgs.append(np.average(np.asarray(self.average_digits[i]), axis=0))
        self.average_digits = np.array(new_avgs)

    def iter(self):
        return iter(self.prep_X), iter(self.prep_Y)

    def get_by_digit(self, digit):
        return iter(self.prep_X_by_digits[digit])

    def getW(self):
        return self.average_digits

    def __getitem__(self, item):
        return self.prep_X[item], self.prep_Y[item]

    def __call__(self, digit):
        return self.average_digits[digit]


class Ensemble:
    def __init__(self, weights, biases):
        self.W = weights
        self.b = biases
        self.softmax = (lambda v: np.exp(v) / np.sum(np.exp(v)))
        self.func = (lambda x: 1 / (1 + np.exp(-x)))

    def predict_digit(self, X, digit):
        return self.func(np.transpose(X) @ self.W[digit] + self.b[digit])

    def predict_all(self, X):
        return np.transpose(X) @ np.transpose(self.W) + self.b

    def predict(self, X):
        return self.softmax(self.predict_all(X))


def test_model(_model, X, Y, unit=False):
    good = 0
    count = 0
    if unit:
        for x in X:
            if _model.predict_digit(x, Y) >= 0.65:
                good += 1
            count += 1
    else:
        for x, y in zip(X, Y):
            if np.argmax(y) == np.argmax(_model.predict(x)):
                good += 1
            count += 1
    return good / count


train_data = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False)

test_data = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)


train = Data(train_data)
test = Data(test_data)

model = Ensemble(train.average_digits, [-13, -6, -18, -12, -9, -12, -15, -10, -16, -11])

tstX, tstY = test.iter()
# tstX = list(test.get_by_digit(2))

print(test_model(model, tstX, tstY))

'''I = 9
results = []
model.b[I] = 0
results.append(test_model(model, tstX, I, True))
for i in range(1, 101):
    model.b[I] = i
    results.append(test_model(model, tstX, I, True))
    model.b[I] = -i
    results = [(test_model(model, tstX, I, True))] + results

plt.plot(list(range(-100, 101)), results)
plt.show()'''

'''for i, digit in enumerate(train.average_digits):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.reshape(digit, (28, 28)), cmap="inferno")
plt.show()'''


