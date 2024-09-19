import matplotlib.pyplot as plt
import numpy as np
import cv2

size1 = 10
c1 = 1 / size1 ** 2
kernel1 = np.array([[c1] * size1] * size1)

size2 = 3
c2 = 1 / size2 ** 2
kernel2 = np.array([[c2] * size2] * size2)

cap = cv2.VideoCapture(0)

last_frame = cap.read()[1]

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(20) & 0xff
    # img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = cv2.filter2D(frame, -1, kernel1)
    image_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    image_gray = cv2.filter2D(image_gray, -1, np.array([[1, 0], [0, -1]]))
    _, thresh = cv2.threshold(image_gray, 5, 255, cv2.THRESH_BINARY)
    '''contours, hierarchy = cv2.findContours(image_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image_out = np.zeros_like(image_gray)
    image_out = cv2.drawContours(image_out, contours, -1, (0, 255, 0), 2)'''

    last_frame = img

    cv2.imshow('Frame', thresh)

    if key == 27:
        break

cv2.destroyWindow('Frame')
cap.release()
