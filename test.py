import matplotlib.pyplot as plt
import numpy as np
import cv2


def analysis(image_):
    global n
    size_ = image_.shape[0] * image_.shape[1]
    red_data = {"min": np.min(image_[:, :, 2]),
                "max": np.max(image_[:, :, 2]),
                "mean": np.mean(image_[:, :, 2]),
                "pixels": image_[120, 120, 2],
                "count": n,
                "non_zero": np.count_nonzero(image_[:, :, 2])}
    green_data = {"min": np.min(image_[:, :, 2]),
                  "max": np.max(image_[:, :, 2]),
                  "mean": np.mean(image_[:, :, 2]),
                  "pixels": image_[120, 120, 1],
                  "count": n,
                  "non_zero": np.count_nonzero(image_[:, :, 2])}
    blue_data = {"min": np.min(image_[:, :, 2]),
                 "max": np.max(image_[:, :, 2]),
                 "mean": np.mean(image_[:, :, 2]),
                 "pixels": np.reshape(image_[:, :, 2], (n, 1)),
                 "count": n,
                 "non_zero": np.count_nonzero(image_[:, :, 2])}
    return red_data, green_data, blue_data


def furry(ys, N_):
    w = np.exp(-2 * np.pi * 1j / N_)
    W = np.fromfunction(lambda x, y: w ** (x * y), (N_, N_))
    return (1 / np.sqrt(N_)) * W @ ys


# ==============================================================================
MIN = 0.06274509803921569
MAX = 0.1450980392156863
MEAN = 0.08654021675857848
# ==============================================================================
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

n = 100
width, height, _ = frame.shape

frames = [frame]

for k in range(n):
    ret, frame = cap.read()
    temp = frames[k] - frame
    frames.append(np.abs(temp))

frames = np.array(frames)

cap.release()


print(np.min(frames[:, :, :, 2]), np.max(frames[:, :, :, 2]), np.mean(frames[:, :, :, 2]))
print(np.min(frames[:, :, :, 1]), np.max(frames[:, :, :, 1]), np.mean(frames[:, :, :, 1]))
print(np.min(frames[:, :, :, 0]), np.max(frames[:, :, :, 0]), np.mean(frames[:, :, :, 0]))
# axis2.plot(ox, (reds - 2 * MEAN + MIN) / (MAX - MIN) + 1.0)

# f = furry(R["pixels"], n)
# ox = np.linspace(0.0, n - 1, n)

'''fig, axiss = plt.subplots(ncols=a, nrows=b)
print(axiss)
for i in range(b):
    for j in range(a):
        axiss[i][j].plot(ox, reds[:, 2 * i + j])
        axiss[i][j].grid()'''

plt.show()
