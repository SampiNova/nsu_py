import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

frames = []
for _ in range(10):
    _, frame = cap.read()
    frames.append(frame)

cap.release()

fig, axiss = plt.subplots(ncols=5, nrows=2)
axiss = np.reshape(np.array(axiss), (1, 10))[0]

for i in range(10):
    if i > 0:
        axiss[i].imshow(frames[i - 1][:, :, ::-1] - frames[i][:, :, ::-1])
    else:
        axiss[i].imshow(frames[i][:, :, ::-1])

plt.show()
