import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import sys
# import segment
 
img = cv2.imread("aman2.jpg")
frameHeight, frameWidth, chanel = img.shape
print(img.shape) 
blurred = cv2.GaussianBlur(img, (5, 5), 0) 
b= cv2.Canny(blurred,100,200)
print(b)
width = round(frameWidth/frameHeight*480)
height = 480
dim = (width, height)
print(type(b))
b=np.array([[0 for i in range(100)] for j in range(200)] )
print(b)
print(type(b))
# resized = cv2.resize(b, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Output-Keypoints",b)
# cv2.imwrite('chun.jpg',b)
# for i in range(480):
#     print()

# print(b[146,209],type(b))
# blurred = img.copy()
# cv2.imshow('a',blurred)
cv2.waitKey(0)
for i in b[:,305]:
    print(i)
for i in range(400):
    if b[i,305] ==255:
        print(i)
print(2)