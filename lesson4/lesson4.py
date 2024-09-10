import numpy as np

# 1
np.random.seed(18182)
a = np.random.randint(5, size=20)
nums, freq = np.unique(a, return_counts=True)
nums = list(nums)
print(a)
print(np.array(sorted(a, key=lambda x: int(freq[nums.index(x)]))))

# 2
h, w = 10, 20
image = np.array(np.random.randint(255, size=(h, w)), dtype=np.uint8)
print(len(np.unique(image)))

# 3
vec = np.random.randint(125, size=10)


def func(v, n):
    summ = np.cumsum(np.insert(v, 0, 0))
    return (summ[n:] - summ[:-n]) / n


print(vec)
print(func(vec, 3))

# 4
n = 30
inp = np.random.randint(180, size=(n, 3))
lam = (lambda y: y[0] + y[1] > y[2] and
                 y[0] + y[2] > y[1] and
                 y[1] + y[2] > y[0])
print(list(filter(lam, inp)))
