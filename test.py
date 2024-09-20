import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

frames = []

n = 50
for _ in range(n):
    ret, frame = cap.read()
    frames.append(frame)
frames = np.array(frames)


reds = frames[:, :, :, 2]
shp_r = reds.shape
reds = reds.reshape((1, shp_r[0] * shp_r[1] * shp_r[2]))
print(reds)

plt.show()
