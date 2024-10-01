import cv2
import numpy as np
cap = cv2.VideoCapture(0)
backSub = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(20) & 0xff

    frame = cv2.medianBlur(frame, 5)
    fg_mask = backSub.apply(frame)
    retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 1000
    max_area = 3000
    large_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]
    frame_ct = cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    cv2.imshow('Frame', frame_ct)
    if key == 27:
        break

cv2.destroyWindow('Frame')
cap.release()
