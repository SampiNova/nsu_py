import cv2

mnr, mxr, mmr = 0.06274509803921569, 0.15294117647058825, 0.08617110983455897
mng, mxg, mmg = 0.0, 0.11372549019607843, 0.08060166245404368
mnb, mxb, mmb = 0.0, 0.611764705882353, 0.06439780394709965

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    key = cv2.waitKey(20) & 0xff
    # img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.imshow('Frame', img)
    if key == 27:
        break

cv2.destroyWindow('Frame')
cap.release()
