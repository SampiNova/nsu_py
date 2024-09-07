import numpy as np

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
P = 0.2

# 1
temp = np.stack((x, y), axis=-1)
print(temp)
lam = (lambda t: np.random.choice(t, p=[1 - P, P]))
print(np.apply_along_axis(lam, 1, temp))

# 2

