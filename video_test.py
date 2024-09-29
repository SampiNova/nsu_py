import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(20) & 0xff
    # img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    cv2.imshow('Frame', frame)
    if key == 27:
        break

cv2.destroyWindow('Frame')
cap.release()
