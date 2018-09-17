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

## [1] COMPLETE CAMERA CALIBRATION
mtx,dist = cam.calib_cam()
# guess source points here
src = np.float32([[200,719],[580,460],[700,460],[1100,719]])
# determine destination points
dst = np.float32([[327,719],[327,0],[971,0],[971,719]])

## METHODS 

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

### COMBINING THRESHOLDS
    # MUST PASS A SINGLE CHANEL IMAGE TO sobelXandY
def sobelXandY(img,skernelx,skernely):
    sobel_x = cv2.Sobel(img,cv2.CV_64F,1,0,skernelx)
    sobel_y = cv2.Sobel(img,cv2.CV_64F,0,1,skernely)
    return sobel_x, sobel_y

def abs_sobel_thresh(img, sobel_kernel=3, thresh=(0, 255), orient='x'):
    # Apply the following steps to img
    # 1)
    sobelx,sobely = sobelXandY(img,sobel_kernel,sobel_kernel)
    
    # 2) Take the absolute value of the derivative in x or y given orient = 'x' or 'y' 
    if orient == 'x':
        abs_sobel = np.absolute(sobelx)
    if orient == 'y':
        abs_sobel = np.absolute(sobely)
        
    # 3) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # 4) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel <= thresh[1]) & (scaled_sobel >= thresh[0])] = 1
    
    # 5) Return this mask as your binary_output image
    return grad_binary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Apply the following steps to img
        
    # 1) Take the gradient in x and y separately
    sobel_x,sobel_y = sobelXandY(img,sobel_kernel,sobel_kernel)
    
    # 3) Calculate the magnitude 
    abs_sobel_xy = np.sqrt((sobel_x**2)+(sobel_y**2)) 
    
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelxy = np.uint8(255*abs_sobel_xy/np.max(abs_sobel_xy))
    
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(scaled_sobelxy)
    mag_binary[(scaled_sobelxy >= mag_thresh[0]) & (scaled_sobelxy <= mag_thresh[1])] = 1
    
    # 6) Return this mask as your binary_output image
    return mag_binary


# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    # Apply the following steps to img
    sobelx,sobely = sobelXandY(img,sobel_kernel,sobel_kernel)
    
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    gradTheta = np.arctan2(abs_sobely,abs_sobelx)
    
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(gradTheta)
    
    # 6) Return this mask as your binary_output image
    binary_output[(gradTheta >= thresh[0]) & (gradTheta <= thresh[1])] = 1
    return binary_output

def getBinary(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)

    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)    
    sobel_kernel = 9    
    sy_thresh = (50,100) #(20,100)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    
    # Sobel_x L channel
    sxLbinary = abs_sobel_thresh(l_channel,sobel_kernel,sx_thresh,'x')
    
    # Sobel_y L channel
    syLbinary = abs_sobel_thresh(l_channel,sobel_kernel,sy_thresh,'y')
       
    # magnitude
    mag_binary = mag_thresh(gray, sobel_kernel, mag_thresh=(30, 100))

    # directional gradient 
    dir_binary = dir_threshold(gray, sobel_kernel, thresh=(0.7, 1.3))
    
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    # Stack each channel
    #color_binary = np.dstack((sxLbinary,mag_binary,dir_binary))*255
    #color_binary = np.dstack((np.zeros_like(mag_binary),mag_binary,dir_binary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),np.zeros_like(sxLbinary),sxLbinary))*255
    color_binary = np.dstack((np.zeros_like(sxLbinary),s_binary,sxLbinary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),sxLbinary,syLbinary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),mag_binary,dir_binary))*255

    combined = np.zeros_like(dir_binary)
    combined[((sxLbinary == 1) & (syLbinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |
            (s_binary == 1)] = 1
    #combined[((sxLbinary == 1)&(sxLbinary==1)) | (s_binary == 1)] = 1 

    return color_binary, combined
    
def framePipeline(img):
    
    return


# START CODE HERE

# Load Test Image
image = cv2.imread('test_images/test2.jpg')
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

# Unwarped Image
undist = cam.un_dis(img,mtx,dist)
M,Minv = calcM(src,dst)

## TESTING PIPELINE
result,combined = getBinary(undist,(170,255),(20,100))

# Warp Image
warped = warper(result,M)
warped2 = warper(combined,M)



