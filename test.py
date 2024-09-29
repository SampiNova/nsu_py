import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

n = 10
height, width, _ = frame.shape

frames = [frame]

for k in range(n - 1):
    ret, frame = cap.read()
    frames.append(frame)

frames = np.array(frames)

cap.release()

fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)

temp = (np.sin(frames[6]) + 1) / 2
print(temp)
ax1.imshow(temp[:, :, 2])
ax2.imshow(temp[:, :, 1])
ax3.imshow(temp[:, :, 0])

ax4.imshow(np.int32(temp * 255))

plt.show()
