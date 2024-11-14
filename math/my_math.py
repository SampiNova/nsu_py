import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from PIL.ImageOps import expand

# Общее уравнение поверхности второго порядка в матричном виде
# X.T @ A @ X + 2 * b.T * X + c = 0

'''
Id = sp.eye(3)
o = np.matrix([[0], [0], [0]], dtype=np.float64)

a11, a22, a33 = with_sqX = np.array([1, 0, 0])  # a11 a22 a33
a12, a13, a23 = with_rotX = np.array([0, 0, 0]) / 2  # a12(xy) a13(xz) a23(yz)
b1, b2, b3 = with_linX = np.array([0, 0, -8]) / 2  # b1 b2 b3
c = 10

A = sp.Matrix([[a11, a12, a13],
               [a12, a22, a23],
               [a13, a23, a33]])
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

a = sp.symbols("a")
eq = sp.Eq(a ** 3 - S * a ** 2 + T * a - d, 0)
print(sp.roots(eq))

x, y, z, w = sp.symbols("x y z w")
eq = sp.Eq(sp.Matrix([x * y * z * w,
                      x * y * z,
                      x * y + y * z + x * z,
                      x + y + z]), sp.Matrix([D, d, T, S]))
_, sol = sp.solve(eq, set=True)
sol = list(sol)
newA = sp.Matrix(np.diag(sol[0]))

X = sp.Matrix([x, y, z])

print(X.T @ newA @ X)


A = sp.Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
b = sp.Matrix([0, 3, -4])
cs = 10
c = sp.Matrix([cs])

l = sp.symbols('l')

det_expr = (A - l * Id).det()
eq = sp.Eq(det_expr, 0)
solution = sp.roots(eq)
print(solution)

match len(solution):
    case 1:
        S = Id
        lam1 = lam2 = lam3 = 1
    case 2:
        my_root = 0
        lam2 = 0
        for root in solution:
            if solution[root] == 1:
                my_root = root
            else:
                lam2 = root
        lam3 = my_root
        lam1 = lam2
        B = A - lam3 * Id
        l3 = np.linalg.lstsq(np.matrix(B, dtype=np.float64), o, None)[0]
        l3 = l3.T
        l3 = sp.Matrix(np.matrix(l3, dtype=np.int32))
        l2 = None
        for i in range(B.cols):
            if B.col(i) != 0:
                l2 = B.col(i).T
                break
        l1 = l2.cross(l3)
        l1 = l1 / l1.norm()
        l2 = l2 / l2.norm()
        l3 = l3 / l3.norm()
        S = sp.Matrix([l1, l2, l3]).T
    case 3:
        S = Id
        lam1 = lam2 = lam3 = 1
    case _:
        print("error")
        quit()


Lam = sp.Matrix([lam1, lam2, lam3])
x, y, z = sp.symbols("x y z")
X = sp.Matrix([x, y, z])
X_sq = sp.Matrix([x ** 2, y ** 2, z ** 2])
print(f"Your input function: {(X.T @ A @ X + 2 * b.T @ X + c).expand()[0]} = 0")

b_ = S.T * b
print((Lam.T @ X_sq + 2 * b_.T @ X + c)[0])

xs = [x, y, z]
lams = [lam1, lam2, lam3]
bs = list(b_)

ans = cs
for i in range(3):
    if bs[i] != 0:
        if lams[i] != 0:
            ans += lams[i] * (xs[i] + bs[i] / lams[i]) ** 2 - bs[i] ** 2 / lams[i]
        else:
            ans += 2 * bs[i] * xs[i]
    else:
        ans += lams[i] * xs[i] ** 2

print(ans)
'''
