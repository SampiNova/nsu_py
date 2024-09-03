
triangle = [[1]]

n = 30

for i in range(2, n + 1):
    new = [1] + [0] * (i - 2) + [1]
    for j in range(i - 2):
        new[j + 1] = triangle[i - 2][j] + triangle[i - 2][j + 1]
    triangle.append(new)


