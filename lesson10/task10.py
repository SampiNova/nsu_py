import numpy as np
import matplotlib.pyplot as plt
import torchvision
from sklearn.metrics import accuracy_score, recall_score, precision_score
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
    accuracy_res = []
    precision_res = []
    recall_res = []
    for x, y in zip(X, Y):
        maxi = np.argmax(_model.predict(x))
        pred = encode_label(maxi)

        accuracy_res.append(accuracy_score(y, pred))
        precision_res.append(precision_score(y, pred, zero_division=0.0))
        recall_res.append(recall_score(y, pred))
    return np.mean(accuracy_res), np.mean(precision_res), np.mean(recall_res)


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
'''for i in range(10):
    print(f"{i} - accuracy:", test_model(model, test.get_by_digit(i), i, True))'''
acc, prec, rec = test_model(model, np.array(list(tstX)), np.array(list(tstY)))
print(f"Accuracy: {acc}\nPrecision: {prec}\nRecall: {rec}")

tests = []
res = []
for i in range(10):
    tsne_model = TSNE(n_components=2, perplexity=20)
    t = np.array(list(test.get_by_digit(i)))[:30]
    squeezed = tsne_model.fit_transform(t)
    tests.append(squeezed)

    temp = []
    for e in t:
        temp.append(model.predict(e))
    tsne_model = TSNE(n_components=2, perplexity=20)
    squeezed = tsne_model.fit_transform(np.array(temp))
    res.append(squeezed)

fig, (axis1, axis2) = plt.subplots(ncols=2, nrows=1)

axis1.set_aspect(1)
for i in range(10):
    axis1.scatter(tests[i][:, 0], tests[i][:, 1], label=str(i))
axis1.legend()

axis2.set_aspect(1)
for i in range(10):
    axis2.scatter(res[i][:, 0], res[i][:, 1], label=str(i))
axis2.legend()

plt.show()
