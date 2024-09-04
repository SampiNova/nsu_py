import argparse

parser = argparse.ArgumentParser()
parser.add_argument("n", type=int)
arg = parser.parse_args()

n = vars(arg)["n"]

triangle = [[1]]

for i in range(2, n + 1):
    new = [1] + [0] * (i - 2) + [1]
    for j in range(i - 2):
        new[j + 1] = triangle[i - 2][j] + triangle[i - 2][j + 1]
    triangle.append(new)

max_size = len(str(triangle[-1][n // 2])) + 2
max_line = max_size * n

for i in range(n):
    print(' ' * ((max_line - max_size * (i + 1)) // 2) + ''.join(map(lambda x: ("{:>" + str(max_size - 1) + "} ").format(x), triangle[i])))
