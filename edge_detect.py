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
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(img, (5, 5), 0) 
b= cv2.Canny(blurred,10,20)
print(b)
width = round(frameWidth/frameHeight*480)
height = 480
dim = (width, height)
print(type(b))
# b=np.array([[0 for i in range(100)] for j in range(200)] )
# print(b)
# print(type(b))
resized = cv2.resize(b, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Output-Keypoints",resized)
# cv2.imwrite('chun.jpg',b)
# for i in range(480):
#     print()

# print(b[146,209],type(b))
# blurred = img.copy()
# cv2.imshow('a',blurred)
cv2.waitKey(0)
# print
for j in range(300,350):
    for i in range(170,187)    :
        print(b[i,j],end = ' ')
    print()
# print(b.shape)
print(b[179,325])
# print(sum(b[298,290:310]))