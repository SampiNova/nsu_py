from random import random
import argparse


def bubble_sort(lst, size):
    i = 1
    while i < size:
        if lst[i - 1] > lst[i]:
            lst[i - 1], lst[i] = lst[i], lst[i - 1]
            i = 1
        else:
            i += 1


parser = argparse.ArgumentParser()
parser.add_argument("n", type=int)
arg = parser.parse_args()

n = vars(arg)["n"]

nums = [random() for _ in range(n)]
print(nums)
bubble_sort(nums, n)
print(nums)
