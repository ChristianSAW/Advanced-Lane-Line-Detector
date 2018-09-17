# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:29:06 2018

@author: Christian Welling
"""

### laneLineFinder_SingleFrame

import numpy as np
import cv2
import cameraCalibrate as cam      # CAMERA CALIBRATE SCRIPT
import getWarped as gw             # BINARY THRESHOLD SCRIPT
import warper as w                 # WARPED IMAGE SCRIPT
import generateLine as gl          # LINE GENERATING SCRIPT
from Line import Line              # Line Class


## BEGIN
def process_image(image,mtx,dist,M,Minv,LeftL,RightL):
    
    # Undistort Image
    undist = cam.un_dis(image,mtx,dist)
    
    # Create Warped Bianry Image
    binary_warped = gw.getWarpedImg(undist,M)
    
    # Process Binary Waprped Image and Produce Processed Image
    processed_img = gl.fit_polynomial(binary_warped,undist,Minv,LeftL,RightL)
    
    return processed_img


## [1] CAMERA CALIBRATION AND UNDISTORTION COEFFICIENTS
'''
*Note: src and dst work specifically well for images within test_images folder. 
Other image files might need to be calibrated with different values for src and
dst. 
**Note: undistortion coefficients are determined speicifically for this camera
which created the images in test_images folder. 
If another camera is used, edit calib_cam() appropriatly.
'''
mtx,dist = cam.calib_cam()
src = np.float32([[195,719],[578,460],[704,460],[1115,719]])
dst = np.float32([[327,719],[327,0],[971,0],[971,719]])

## [2] UNDISTORTION COEFFICIENTS
M,Minv = w.calcM(src,dst)

## [2] CREATE LINES 
LeftL = Line()   # create left line  object
RightL = Line()  # create right line object

## [3] PROCESS IMAGE AND CREATE VIDEO FILE
'''
Enter in the name of the Image to be processed.
*Note: This process is tuned to work well for images within test_images.
un_dis() and getWarped() need to be turned for this to work for all images.
**Note: Image input must be of 720x1200 format. 
'''

# EDIT HERE    ######
image = cv2.imread('test_images/test2.jpg') # input image
write_output = 'output_images/test2_output.jpg'
# STOP EDITS   ######

img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB) # convert to RGB

processed_image = process_image(img,mtx,dist,M,Minv,LeftL,RightL)
outputImage = cv2.cvtColor(processed_image,cv2.COLOR_RGB2BGR) # convert to BGR for jpg
cv2.imwrite(write_output,outputImage)




