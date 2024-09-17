import matplotlib.pyplot as plt
import numpy as np
import cv2

cap = cv2.VideoCapture(0)

frames = []
for _ in range(10):
    _, frame = cap.read()
    frames.append(frame)

cap.release()
