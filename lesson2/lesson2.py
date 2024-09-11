from random import randint

# 1
word1 = input("1:\nword = ")
idx1 = len(word1) // 2
if word1[:idx1] == word1[-1:-idx1-1:-1] or len(word1) < 2:
    print("YES")
else:
    print("NO")

# 2
text1 = input("2:\nwords = ").split(" ")
print(max(text1, key=lambda x: len(x)))

# 3
n1 = int(input("3:\nn = "))
lst1 = [randint(-9223372036854775807, 9223372036854775807) for _ in range(n1)]
print("odd:", list(filter(lambda x: x % 2 != 0, lst1)))
print("even:", list(filter(lambda x: x % 2 == 0, lst1)))

# 4
dictionary = {"add": "odd", "b": "d", "f": "g"}
text2 = input("4:\nwords = ").split(" ")
print(' '.join([dictionary[w] if w in dictionary.keys() else w for w in text2]))

# 5
n2 = int(input("5:\nn = "))


def fib(n, i=0, res=None):
    if res is None:
        res = [0, 1]
    if n < 2:
        return res[i]
    if i < n:
        res[0], res[1] = res[1], sum(res)
        return fib(n, i + 1, res)
    return res[1]


print(fib(n2))

# 6
symbols = 0
words = 0
lines = 0
with open("test6.txt", "r") as file:
    for line in file.readlines():
        tmp = line[:-1].split(' ')
        words += len(tmp)
        symbols += sum(map(lambda x: len(x), tmp))
        lines += 1
print(f"6:\nlines: {lines}\nwords: {words}\nsymbols: {symbols}")

# 7
a, b = float(input("7:\nb1 = ")), float(input("q = "))


def gen(b_one, q):
    b_last = b_one
    while True:
        b_last *= q
        yield b_last


geom = gen(a, b)
for _ in range(100):
    print(next(geom))
