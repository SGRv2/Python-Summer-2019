#Import cv

import cv2

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# reading the image with cv2
img = cv2.imread('picklejar.jpg')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(grayimg,127,255,cv2.THRESH_BINARY)
edges = cv2.Canny(thresh ,100, 200)

# locate the largest contour aka the label
contours,hierarchy = cv2.findContours(edges, 0, 1)
areas = [cv2.contourArea(c) for c in contours]
max_index = np.argmax(areas)
cnt=contours[max_index]

# Create a mask of the label
mask = np.zeros(img.shape,np.uint8)
cv2.drawContours(mask, [cnt],0,255,-1)

# Find the 4 borders using Sobel
scale = 1
delta = 0
ddepth = cv2.CV_8U
borderType=cv2.BORDER_DEFAULT
left=cv2.Sobel(mask,ddepth,1,0,ksize=1,scale=1,delta=0,borderType=borderType)
right=cv2.Sobel(mask,ddepth,1,0,ksize=1,scale=-1,delta=0, borderType=borderType)
top=cv2.Sobel(mask,ddepth,0,1,ksize=1,scale=1,delta=0,borderType=borderType)
bottom=cv2.Sobel(mask,ddepth,0,1,ksize=1,scale=-1,delta=0,borderType=borderType)

# remove noise from borders
kernel = np.ones((2,2),np.uint8)
leftb = cv2.erode(left,kernel,iterations = 1)
rightb = cv2.erode(right,kernel,iterations = 1)
topb = cv2.erode(top,kernel,iterations = 1)
bottomb = cv2.erode(bottom,kernel,iterations = 1)

'''
Find coefficients c1,c2, ... ,c7,c8 by minimizing the error function. 
Points on the:
 left border should be mapped to (0,anything).
 right border should be mapped to (108,anything)
 top border should be mapped to (anything,0)
 bottom border should be mapped to (anything,70)
 Equations 1 and 2: 
    c1 + c2*x + c3*y + c4*x*y, c5 + c6*y + c7*x + c8*x^2
'''

sumOfSquares_y = '+'.join(["(c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
    (x,y,x,y) for y,x,z in np.transpose(np.nonzero(leftb)) ])
sumOfSquares_y += " + "
sumOfSquares_y += '+'.join(["(-108+c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
    (x,y,x,y) for y,x,z in np.transpose(np.nonzero(rightb)) ])
res_y = optimize.minimize(lambda c: eval(sumOfSquares_y),(0,0,0,0),method='SLSQP')

sumOfSquares_x = '+'.join(["(-70+c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
    (y,x,x,x) for y,x,z in np.transpose(np.nonzero(bottomb))])
sumOfSquares_x += " + "
sumOfSquares_x += '+'.join( [ "(c[0]+c[1]*%s+c[2]*%s+c[3]*%s*%s)**2" % \
    (y,x,x,x) for y,x,z in np.transpose(np.nonzero(topb)) ] )
res_x = optimize.minimize(lambda c: eval(sumOfSquares_x),(0,0,0,0), method='SLSQP')


# Map the image using equations 1 and 2 
def map_x(res, coord):
    m = res[0]+res[1]*coord[1]+res[2]*coord[0]+res[3]*coord[1]*coord[0]
    return m
def map_y(res, coord):
    m = res[0]+res[1]*coord[0]+res[2]*coord[1]+res[3]*coord[1]*coord[1]
    return m

flattened = np.zeros(img.shape, img.dtype) 
for y,x,z in np.transpose(np.nonzero(mask)):
    new_y = map_y(res_x.x,[y,x]) 
    new_x = map_x(res_y.x,[y,x])
    flattened[int(new_y)][int(new_x)] = img[y][x]
# Crop the image 
flattened = flattened[0:70, 0:105]

