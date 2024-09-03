from random import randint
import numpy as np
from math import *
import os
'''
# 1
num1 = randint(100, 999)
print("1:", num1, num1 // 100 + num1 % 100 // 10 + num1 % 10)

# 2
num2 = randint(0, 9223372036854775807)
print("2:", num2, sum(map(int, list(str(num2)))))

# 3
r = int(input("r = "))
print(f"3: v = {(4 / 3) * pi * r ** 3}; s = {4 * pi * r ** 2}")

# 4
year = int(input("year = "))
print("4:", end=" ")
if year % 4 == 0:
    if year % 100 == 0:
        print("YES" if year % 400 == 0 else "NO")
    else:
        print("YES")
else:
    print("NO")

# 5
n = int(input("n = "))
grid = [1 for _ in range(n)]
ans = []

i = 2
while i ** 2 <= n:
    if grid[i - 1]:
        j = i ** 2
        while j <= n:
            grid[j - 1] = 0
            j += i
    i += 1

print("5: ", end='')
for i in range(0, n):
    if grid[i]:
        print(i + 1, end=' ')
print()

# 6
x = int(input("money = "))
y = int(input("time = "))

for _ in range(y):
    x += 0.1 * x
print("6:", x)
'''
# 7
path = input("path = ")
tree_gen = os.walk(path)


def print_tree(idx, tree):
    global path
    try:
        obj = next(tree)
    except BaseException:
        return
    print("\t" * idx + obj[0])
    if obj[1]:
        for under_dir in obj[1]:
            print_tree(idx + 1, os.walk(path + f"\\{under_dir}"))
    if obj[-1]:
        for elem in obj[-1]:
            print("\t" * (idx + 1) + elem)
    else:
        print("\t" * (idx + 1) + "none")


print("7:")
print_tree(0, tree_gen)
