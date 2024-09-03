mat1 = list()
mat2 = list()

with open("matrix1.txt", "r") as file:
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

n, m = len(mat1), len(mat1[0])

