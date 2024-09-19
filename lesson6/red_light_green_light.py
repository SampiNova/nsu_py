import matplotlib.pyplot as plt
import numpy as np
import cv2


def smooth(lst):
    global n

    tmp = [lst[0]]
    for i, e in zip(range(1, n - 1), lst[1:-1]):
        tmp.append((lst[i - 1] + e + lst[i + 1]) // 3)
    tmp.append(lst[-1])

    return tmp


# =======================================

size = 6
c = 1 / size ** 2
kernel = np.array([[c] * size] * size)

cap = cv2.VideoCapture(0)

n = 50
frames = []
acc = cap.read()[1] / n
for _ in range(n - 1):
    _, frame = cap.read()
    # img = cv2.filter2D(frame, -1, kernel)
    acc += frame / n
    frames.append(frame)

cap.release()

plt.imshow(acc)
'''fig, axiss = plt.subplots(ncols=10, nrows=5)
axiss = np.reshape(np.array(axiss), (1, n))[0]

for i in range(n):
    if i > 0:
        axiss[i].imshow(frames[i - 1][:, :, ::-1] - frames[i][:, :, ::-1])
    else:
        axiss[i].imshow(frames[i][:, :, ::-1])'''

plt.show()
