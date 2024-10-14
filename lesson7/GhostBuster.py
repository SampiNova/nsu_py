import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

img1 = cv2.imread("candy_ghost.png")
img2 = cv2.imread("pampkin_ghost.png")
img3 = cv2.imread("scary_ghost.png")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, ds1 = sift.detectAndCompute(img1, None)
kp2, ds2 = sift.detectAndCompute(img2, None)
kp3, ds3 = sift.detectAndCompute(img3, None)

#feature matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

matches1 = bf.match(ds1, ds2)
matches1 = sorted(matches1, key = lambda x:x.distance)

res1 = cv2.drawMatches(img1, kp1, img2, kp2, matches1[:50], img2, flags=2)

plt.imshow(res1, cmap='gray')
plt.show()
