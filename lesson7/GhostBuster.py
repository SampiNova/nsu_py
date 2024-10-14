import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

dir_path = "ghosts"
image = cv2.imread(f"{dir_path}\\candy_ghost.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
sift = cv2.ORB_create()
kp = sift.detect(gray, None)

result = image.copy()
result = cv2.drawKeypoints(gray, kp, result)
plt.imshow(result, cmap='gray')
plt.show()
