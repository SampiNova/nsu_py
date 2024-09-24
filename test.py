import random
import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

MAX, MEAN, MIN = 44, 22.3, 15
COLORS = []
PROB = [1.95312500e-05, 4.03645833e-04, 1.61588542e-02, 1.37457682e-01,
        2.63968099e-01, 2.54651693e-01, 1.72513021e-01, 9.13085938e-02,
        4.18164063e-02, 1.64713542e-02, 3.99739583e-03, 9.50520833e-04,
        1.62760417e-04, 9.11458333e-05, 1.30208333e-05, 9.76562500e-06,
        6.51041667e-06]
n = 10
height, width, _ = frame.shape

frames = [frame]
red_maxes = []
red_means = []
red_mins = []

for k in range(n - 1):
    ret, frame = cap.read()
    red_maxes.append(frame[:, :, 2].max())
    red_means.append(frame[:, :, 2].mean())
    red_mins.append(frame[:, :, 2].min())
    frames.append(frame)

frames = np.array(frames)

cap.release()
red_maxes = np.array(red_maxes)
red_means = np.array(red_means)
red_mins = np.array(red_mins)

fig, (axis1, axis2) = plt.subplots(ncols=2)

colors, counts = np.unique(frames[1][:, :, 2], return_counts=True)
all_pixels = sum(counts)


axis1.imshow(frames[1][:, :, 2])

attract_points =
h = np.fromfunction(lam_h, (height, width))

axis2.plot(colors, counts)

print(red_maxes.max())
print(red_means.mean())
print(red_mins.min())

plt.show()
