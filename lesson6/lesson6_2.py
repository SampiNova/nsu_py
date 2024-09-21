import cv2
import numpy as np

mnr, mxr, mmr = 10, 43, 22.23471044921875
dmnr, dmxr = round(mmr - mnr), round(mxr - mmr)
mng, mxg, mmg = 0, 29, 20.14170888671875
dmng, dmxg = round(mmg - mng), round(mxg - mmg)
mnb, mxb, mmb = 0, 158, 18.519935872395834
dmnb, dmxb = round(mmb - mnb), round(mxb - mmb)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(20) & 0xff
    # img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame[:, :, 2] = np.where(frame[:, :, 2] > mmr, frame[:, :, 2] - dm)
    cv2.imshow('Frame', img)
    if key == 27:
        break

cv2.destroyWindow('Frame')
cap.release()
