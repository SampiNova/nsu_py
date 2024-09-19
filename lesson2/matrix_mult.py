mat1 = list()
mat2 = list()

with open("matrix_mult.txt", "r") as file:
    flag = 1
    for line in file.readlines():
        tmp = line[:-1].split(' ')
        if len(tmp) < 2 and not bool(tmp[0]):
            flag = 0
            continue
        tmp = list(map(int, tmp))
        if flag:
            mat1.append(tmp)
        else:
            mat2.append(tmp)

m, n, k = len(mat1), len(mat1[0]), len(mat2[0])
res = [[0] * k] * m

for r in range(m):
    for i in range(k):
        for c in range(n):
            res[r][i] += mat1[r][c] * mat2[c][i]

with open("res_mult.txt", "w") as file:
    res = list(map(lambda x: ' '.join(map(str, x)) + "\n", res))
    file.writelines(res)
