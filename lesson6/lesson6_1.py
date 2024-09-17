import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


images_str = os.listdir("nails_segmentation\\images")
labels_str = os.listdir("nails_segmentation\\labels")
print(images_str)
print(labels_str)
