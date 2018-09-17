# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:27:19 2018

@author: Christian Welling
"""
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cameraCalibrate as cam
import warper as w


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

def getBinary(img):
    
    ## Thresholds 
    s_thresh=(110,255)    #  S-Channel  
    sx_thresh=(20, 100)   # Sobel X Direction, L-Channel
    sy_thresh = (50,100)  # Sobel Y Direction, L-Channel
    
    sxs_thresh=(10, 50)   # Sobel X Direction, S-Channel
    sys_thresh = (20,100) # Sobel Y Direction, S-Channel
    
    h_thresh = (10,25)    # H-Channel 
    
    # detecting white lines 
#    r_thresh = (220,255)
#    g_thresh = r_thresh
#    b_thresh = r_thresh

    # detecting white
    l_thresh = (95,100)   # L-Channel
    
    # Kernel
    sobel_kernel = 9    

    ## Extract Channels 
    # RGB     
#    r_channel = img[:,:,0]
#    g_channel = img[:,:,1]
#    b_channel = img[:,:,2]
    
    # Generate Gray Scale
    #gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)    
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    
    # Sobel_x L channel
    sxLbinary = abs_sobel_thresh(l_channel,sobel_kernel,sx_thresh,'x')
    sxSbinary = abs_sobel_thresh(s_channel,sobel_kernel,sxs_thresh,'x')
    
    # Sobel_y L channel
    syLbinary = abs_sobel_thresh(l_channel,sobel_kernel,sy_thresh,'y')
    sySbinary = abs_sobel_thresh(s_channel,sobel_kernel,sys_thresh,'y')
    
    ## Threshold color channel
    # RGB
#    r_binary = np.zeros_like(r_channel)
#    r_binary[(r_channel >= r_thresh[0]) & (r_channel < r_thresh[1])] = 1
#    
#    g_binary = np.zeros_like(g_channel)
#    g_binary[(g_channel >= g_thresh[0]) & (g_channel < g_thresh[1])] = 1
#    
#    b_binary = np.zeros_like(b_channel)
#    b_binary[(b_channel >= b_thresh[0]) & (b_channel < b_thresh[1])] = 1
    
    # HLS
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
            
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
        
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1


    ## Combine for Binary Image
    combinedColor = np.zeros_like(s_binary)
    combinedColor[(s_binary == 1) & (h_binary == 1)] = 1
    
    combinedSobL = np.zeros_like(s_binary)
    combinedSobL[(sxLbinary == 1) & (syLbinary == 1)] = 1
    
    combinedSobS = np.zeros_like(s_binary)
    combinedSobS[(sxSbinary == 1)&(sySbinary == 1)&(h_binary==1)] = 1
    
#    combinedRGB = np.zeros_like(r_binary)
#    combinedRGB[(r_binary == 1)&(g_binary == 1) & (b_binary == 1)] = 1
    
    combined = np.zeros_like(s_binary)
    combined[(combinedColor == 1) | (combinedSobL == 1) | (combinedSobS == 1)] = 1

    return combined

def getWarpedImg(img,M):
    combined = getBinary(img)
    unwarped = w.warper(combined,M)
    return unwarped     