# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 14:25:18 2018

@author: Christian Welling
"""

### laneLineFinder_Video

import numpy as np
import cameraCalibrate as cam      # CAMERA CALIBRATE SCRIPT
import getWarped as gw             # BINARY THRESHOLD SCRIPT
import warper as w                 # WARPED IMAGE SCRIPT
import generateLine as gl          # LINE GENERATING SCRIPT
from Line import Line              # Line Class

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

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
*Note: src and dst work specifically well for project_video.mp4. Other video  
files might need to be calibrated with different values. 
**Note: undistortion coefficients are determined speicifically for this camera
which filmed project_video.mp4. If another camera is used, edit calib_cam() 
appropriatly.
'''
mtx,dist = cam.calib_cam()
src = np.float32([[195,719],[578,460],[704,460],[1115,719]])
dst = np.float32([[327,719],[327,0],[971,0],[971,719]])

## [2] UNDISTORTION COEFFICIENTS
M,Minv = w.calcM(src,dst)

## [2] CREATE LINES 
LeftL = Line()   # create left line  object
RightL = Line()  # create right line object

## [3] PROCESS VIDEO AND CREATE VIDEO FILE
'''
Enter in the name of the video to be processed for clip1. Enter in the name of 
the processed video name for white_output. 
*Note: This process is tuned to work well for project_video and videos simmilar 
to it. un_dis() and getWarped() need to be turned for this to work for all videos.
**Note: Video input must be of 720x1200 format. 
'''

# EDIT HERE    ######
white_output = 'test_vid_outputs/project_video_processed.mp4' # Output Vid Name
# STOP EDITS   ######

## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

# EDIT HERE    ######
clip1 = VideoFileClip("project_video.mp4") # Input Video Name
# STOP EDITS   ######

#NOTE: this function (process_image) expects color (RGB) images!!
white_clip = clip1.fl_image(lambda image: process_image(image,mtx,dist,M,Minv,LeftL,RightL)) 
white_clip.write_videofile(white_output, audio=False) # I think this should work

