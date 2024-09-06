import numpy as np

# 1
np.random.seed(18182)
a = np.random.randint(5, size=20)
nums, freq = np.unique(a, return_counts=True)
nums = list(nums)
print(a)
print(np.array(sorted(a, key=lambda x: int(freq[nums.index(x)]))))
