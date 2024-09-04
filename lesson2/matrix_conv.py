mat1 = list()
mat2 = list()

with open("lesson2\\matrix_conv.txt", "r") as file:
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

nx, ny, mx, my = len(mat1[0]), len(mat1), len(mat2[0]), len(mat2)
res = [[0 for _ in range(nx - mx + 1)] for _ in range(ny - my + 1)]

for i in range(ny - my + 1):
    for j in range(nx - mx + 1):
        for x in range(mx):
            for y in range(my):
                res[i][j] += mat1[i + y][j + x] * mat2[y][x]

with open("lesson2\\res_conv.txt", "w") as file:
    res = list(map(lambda u: ' '.join(map(str, u)) + "\n", res))
    file.writelines(res)
