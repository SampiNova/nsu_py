import numpy as np

# 1
print("\n1:")
np.random.seed(18182)
a = np.random.randint(5, size=20)
nums, freq = np.unique(a, return_counts=True)
# ============================================
new = np.stack((nums, freq), axis=-1)
new = new[new[:, 1].argsort()]
# ============================================
lam = (lambda x: np.zeros((new[x, 1], 1)) + new[x, 0])
new = [lam(i) for i in range(new.shape[0])]
print(a)
print(np.concatenate(new).reshape(a.shape))

# 2
print("\n2:")
h, w = 10, 20
image = np.array(np.random.randint(255, size=(h, w)), dtype=np.uint8)
print(len(np.unique(image)))

# 3
print("\n3:")
vec = np.random.randint(125, size=10)


def func(v, n):
    summ = np.cumsum(np.insert(v, 0, 0))
    print(summ[n:], summ[:-n])
    return (summ[n:] - summ[:-n]) / n


print(vec)
print(func(vec, 3))

# 4
print("\n4:")
n = 30
inp = np.random.randint(180, size=(n, 3))
lam = (lambda y: y[0] + y[1] > y[2] and
                 y[0] + y[2] > y[1] and
                 y[1] + y[2] > y[0])
print(list(filter(lam, inp)))

# 5
print("\n5:")

mA = np.array([[3, 4, 2], [5, 2, 3], [4, 3, 2]], dtype=np.float32)
mB = np.array([17, 23, 19], dtype=np.float32)

print(np.linalg.inv(mA) @ mB)

# 6
print("\n6:")
A = np.matrix("1 0 1; 0 1 0; 1 0 1")
U, S, Vt = np.linalg.svd(A)
print(U)
print(S)
print(Vt)
