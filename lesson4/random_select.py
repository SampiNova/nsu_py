import numpy as np
from random import choices
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("txt1", type=str)
parser.add_argument("txt2", type=str)
parser.add_argument("P", type=float)
arg = parser.parse_args()

name1, name2, P = vars(arg)["txt1"], vars(arg)["txt2"], vars(arg)["P"]

with open(name1, "r") as file:
    x = list(map(int, file.readline().split(' ')))
with open(name2, "r") as file:
    y = list(map(int, file.readline().split(' ')))
n = len(x)

# 1
t1 = choices(x, k=n - round(n * P))
t2 = choices(y, k=round(n * P))
temp1 = np.array(t1 + t2)
np.random.shuffle(temp1)
print(temp1)

# 2
temp2 = np.array(x + y)
print(np.array(choices(temp2, weights=([(1 - P) / n] * n + [P / n] * n), k=n)))

# 3
temp3 = np.stack((x, y), axis=-1)
lam = (lambda t: np.random.choice(t, p=[1 - P, P]))
print(np.apply_along_axis(lam, 1, temp3))
