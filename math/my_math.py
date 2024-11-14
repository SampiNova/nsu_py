import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from PIL.ImageOps import expand
from sympy import simplify


def simp_sq(sq_a, sq_b):
    f = [0, 0, 0]
    s = 0
    for i in range(3):
        if sq_b[i] != 0:
            if sq_a[i] != 0:
                f[i] = sq_b / sq_a / 2
                s += f[i] ** 2 * sq_a
    return f, s

def simp_pr(sq_a, sq_b, sq_c):
    return -sq_b / sq_a, sq_c / sq_b


# Общее уравнение поверхности второго порядка в матричном виде
# X.T @ A @ X + 2 * b.T * X + c = 0


Id = sp.eye(3)
o = np.matrix([[0], [0], [0]], dtype=np.float64)

a11, a22, a33 = with_sqX = np.array([1, 0, 0], dtype=np.float64)  # a11 a22 a33
a12, a13, a23 = with_rotX = np.array([0, 0, 0]) / 2  # a12(xy) a13(xz) a23(yz)
b1, b2, b3 = with_linX = np.array([0, 0, -8]) / 2  # b1 b2 b3
c = 10

A = sp.Matrix([[a11, a12, a13],
               [a12, a22, a23],
               [a13, a23, a33]])
b = sp.Matrix([b1, b2, b3])
cm = sp.Matrix([c])

bigA = sp.Matrix([[a11, a12, a13, b1],
                  [a12, a22, a23, b2],
                  [a13, a23, a33, b3],
                  [b1, b2, b3, c]])
d = sp.det(A)
D = sp.det(bigA)
S = a11 + a22 + a33
T = (sp.det(sp.Matrix([[a11, a12],
                       [a12, a22]])) +
     sp.det(sp.Matrix([[a22, a23],
                       [a23, a33]])) +
     sp.det(sp.Matrix([[a11, a13],
                       [a13, a33]])))

D_ = (sp.det(sp.Matrix([[a11, b1],
                        [b1, c]])) +
     sp.det(sp.Matrix([[a22, b2],
                       [b2, c]])) +
     sp.det(sp.Matrix([[a33, b3],
                       [b3, c]])))

first_table = [["Imaginary ellipsoid", "Ellipsoid", "Imaginary cone"],
               ["One-sheet hyperboloid", "Two-sheet hyperboloid", "Real cone"],
               ["Hyperbolic paraboloid", "Elliptic paraboloid", "Cylindrical and reducible surfaces"]]
if d != 0:
    if d * S > 0:
        first_row = 0
    else:
        first_row = 1
else:
    first_row = 2

print(d, D, D_, T, S)

A = sp.Matrix([[a11, a12, a13], [a12, a22, a23], [a13, a23, a33]])

ev = A.eigenvects()
dg = []
es = []
for li in ev:
    dg.append(li[0])
    es.append((li[2][0] / li[2][0].norm()).T)

x, y, z = sp.symbols("x y z")
X = sp.Matrix([x, y, z])

A_ = sp.diag(*dg)

Q = sp.Matrix(es)
newX = Q @ X

print(f"Input: {(X.T @ A @ X + 2 * b.T @ X + cm)[0]} = 0")
newF = (newX.T @ A_ @ newX + 2 * (Q.T @ b).T @ newX + cm)[0]
print(newF)

if T == 0:
    my_pol = sp.poly(newF, X)
    print(simp_pr(*my_pol.coeffs()))
    print(my_pol.free_symbols)
else:
    pass

