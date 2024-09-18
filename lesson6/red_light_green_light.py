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

cap = cv2.VideoCapture(0)

n = 100
frames = []
for _ in range(n):
    _, frame = cap.read()
    frames.append(frame)

cap.release()

delta_f = []
delta_pr = []
delta_pg = []
delta_pb = []
for i in range(n):
    if i > 0:
        delta_f.append(frames[i - 1][:, :, ::-1] - frames[i][:, :, ::-1])
    else:
        delta_f.append(frames[i][:, :, ::-1])
    delta_pr.append(delta_f[i][120, 120, 0])
    delta_pg.append(delta_f[i][120, 120, 1])
    delta_pb.append(delta_f[i][120, 120, 2])

ox = np.linspace(0.0, 1.0, n)
fig, (axis1, axis2, axis3) = plt.subplots(nrows=1, ncols=3)

axis1.plot(ox, smooth(delta_pr), "r-*")
axis2.plot(ox, smooth(delta_pg), "g-*")
axis3.plot(ox, smooth(delta_pb), "b-*")

'''fig, axiss = plt.subplots(ncols=5, nrows=2)
axiss = np.reshape(np.array(axiss), (1, 10))[0]

for i in range(10):
    if i > 0:
        axiss[i].imshow(frames[i - 1][:, :, ::-1] - frames[i][:, :, ::-1])
    else:
        axiss[i].imshow(frames[i][:, :, ::-1])'''

plt.show()
