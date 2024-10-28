import cv2
import numpy as np

from utils import *

image = cv2.imread("test_image\\cat512.jpg")
image_x = HSL2RGB(RGB2HSL(image))

for i in range(256):
    for j in range(256):
        p1 = image[i,j,:]
        p2 = image_x[i,j,:]
        if np.square(p1 - p2).sum() > 5:
            print(f"{p1}, {p2}")

cv2.imwrite("0.jpg", image_x)
    