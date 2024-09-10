import numpy as np
from random import choices

n = 10
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
P = 0.2

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
