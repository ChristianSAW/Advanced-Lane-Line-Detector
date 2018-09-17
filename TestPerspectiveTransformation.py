# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:29:06 2018

@author: Christian Welling
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cameraCalibrate as cam
#%matplotlib qt

# Calibrate Camera Once 
mtx,dist = cam.calib_cam()
# guess source points here
src = np.float32([[195,719],[578,460],[704,460],[1115,719]])
#print(src[0])
# determine destination points
dst = np.float32([[327,719],[327,0],[971,0],[971,719]])


#src = np.float32([[195,719],[578,460],[704,460],[1115,719]])
#dst = np.float32([[327,719],[327,0],[971,0],[971,719]])

## METHODS 

#  rough pipeline to determine 

# Generate M and Minv from Source and Destination Points
def calcM(src,dst):
    # calculate transformation coefficients
    M = cv2.getPerspectiveTransform(src,dst)
    Minv = cv2.getPerspectiveTransform(dst,src)
    
    return M,Minv

def warper(img,M):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    return warped


# START CODE HERE

# Load Test Image
#image = cv2.imread('test_images/straight_lines2.jpg')
image = cv2.imread('test_images/test2.jpg')
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Unwarped Image
undist = cam.un_dis(img,mtx,dist)

img_size = undist.shape[1::-1]
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])



M,Minv = calcM(src,dst)
warped = warper(undist,M)

# Test That This Works (WORKS)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
f.tight_layout()
ax1.imshow(undist)
ax1.set_title('Undistorted Image', fontsize=20)

h = undist.shape[0]
w = undist.shape[1]

print(h-1)
print(w-1)

ax1.axhline(460)        # top y
ax1.axhline(h-1)        # bottom y
ax1.axvline(200)        # left ax
ax1.axvline(580)        # left bx
ax1.axvline(700)        # right cx
ax1.axvline(1100)       # right dx

cv2.line(undist, tuple(src[0]), tuple(src[1]), (255, 0, 0), thickness=2, lineType=8)
cv2.line(undist, tuple(src[1]), tuple(src[2]), (255, 0, 0), thickness=2, lineType=8)
cv2.line(undist, tuple(src[2]), tuple(src[3]), (255, 0, 0), thickness=2, lineType=8)
cv2.line(undist, tuple(src[3]), tuple(src[0]), (255, 0, 0), thickness=2, lineType=8)





ax1.imshow(undist)


ax2.set_title('Warped Image', fontsize=20)
ax2.imshow(warped)
ax2.axvline(327)
ax2.axvline(971)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    


# Test Pipeline on 1 image

