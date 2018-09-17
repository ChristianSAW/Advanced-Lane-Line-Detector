# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 12:53:57 2018

@author: Christian Welling

Test Script for Project 2
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import cameraCalibrate as cam      # CAMERA CALIBRATE SCRIPT
import getWarped as gw             # BINARY THRESHOLD SCRIPT
import warper as w                 # WARPED IMAGE SCRIPT
import generateLine as gl          # LINE GENERATING SCRIPT
from Line import Line              # Line Class
import os                          # for testing frame by frame

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

## Create Some Frames for Video Testing Later
def createVidFrames(vidname,outfolder):
    cap = cv2.VideoCapture(vidname)
    while not cap.isOpened():
        cap = cv2.VideoCapture(vidname)
        cv2.waitKey(1000)
        print("Wait for the header")
        
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    count = 1
    while(True):
        flag, frame = cap.read()
        if flag:
            # The frame is ready and already captured
            cv2.imshow('video', frame)
            pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            print(str(pos_frame)+" frames")
            cv2.imwrite(outfolder + "/frame%d.jpg" % count, frame)
            count = count + 1
        else:
            # The next frame is not ready, so we try to read it again
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos_frame-1)
            print("frame is not ready")
            # It is better to wait for a while for the next frame to be ready
            cv2.waitKey(1000)
                    
        if cv2.waitKey(10) == 27:
            break
        if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            # If the number of captured frames is equal to the total number of frames,
            # we stop
            break
    return
    
# create frames for project video:
vidname = "project_video.mp4"
outfolder = "test_vid_frames_project_video"
#createVidFrames(vidname,outfolder)
#cap = cv2.VideoCapture(vidname)
#numFrame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#print(vidname + " has " + str(numFrame) + " total frames.")

## DEFAULT IMAGES FOR TESTING
# Load Test Image
#image = cv2.imread('test_images/straight_lines1.jpg')
image = cv2.imread('test_images/test2.jpg')
img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)


def testUndistort(img,undist):
    #Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(undist)
    ax2.set_title('Undistorted Image', fontsize=20)
    return


## CAMERA CALIBRATION [COMPLETE]
mtx,dist = cam.calib_cam()


############## PIPELINE IMAGE ##############

#### TEST FUNCTIONS ###
def testBinThresh(undist,src,dst):
    # Thresholds 
    s_thresh=(110,255)    # green (220,221)
    s_thresh2 = (110,170)   # blue
    s_thresh3 = (170,220)  # red 
    s_thresh4 = (110,255)
    
    
    sx_thresh=(20, 100)
    sy_thresh = (50,100) #(20,100)
    
    sxs_thresh=(10, 50)   # (70,100) or (10,50)
    sys_thresh = (20,100) #(50,100) or (20,100)
    
    # detecting yellow
    h_thresh = (10,25) # red (15,25)
    h_thresh2 = (20,25) # blue
    h_thresh3 = (0,25)  #(15,25) green
    
    # detecting white lines 
    r_thresh = (220,255)
    g_thresh = r_thresh
    b_thresh = r_thresh

    # detecting white
    l_thresh = (95,100)
    
    # Kernel
    sobel_kernel = 9    

    # Test combinations  of thesholding  to find right one
    img = np.copy(undist)
    r_channel = img[:,:,0]
    g_channel = img[:,:,1]
    b_channel = img[:,:,2]
    
    
    # Generate Gray Scale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)    
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    
    # Sobel_x L channel
    sxLbinary = gw.abs_sobel_thresh(l_channel,sobel_kernel,sx_thresh,'x')
    sxSbinary = gw.abs_sobel_thresh(s_channel,sobel_kernel,sxs_thresh,'x')

    
    # Sobel_y L channel
    syLbinary = gw.abs_sobel_thresh(l_channel,sobel_kernel,sy_thresh,'y')
    sySbinary = gw.abs_sobel_thresh(s_channel,sobel_kernel,sys_thresh,'y')

       
    # magnitude
    mag_binary = gw.mag_thresh(gray, sobel_kernel, mag_thresh=(30, 100))

    # directional gradient 
    dir_binary = gw.dir_threshold(gray, sobel_kernel, thresh=(.7, 1.3)) #(.7,1.3)
    
    
    # Threshold color channel
    r_binary = np.zeros_like(r_channel)
    r_binary[(r_channel >= r_thresh[0]) & (r_channel < r_thresh[1])] = 1
    
    g_binary = np.zeros_like(g_channel)
    g_binary[(g_channel >= g_thresh[0]) & (g_channel < g_thresh[1])] = 1
    
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh[0]) & (b_channel < b_thresh[1])] = 1
    
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    s_binary2 = np.zeros_like(s_channel)
    s_binary2[(s_channel >= s_thresh2[0]) & (s_channel <= s_thresh2[1])] = 1
    
    s_binary3 = np.zeros_like(s_channel)
    s_binary3[(s_channel >= s_thresh3[0]) & (s_channel <= s_thresh3[1])] = 1
    
    s_binary4 = np.zeros_like(s_channel)
    s_binary4[(s_channel >= s_thresh4[0]) & (s_channel <= s_thresh4[1])] = 1

    
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_thresh[0]) & (h_channel <= h_thresh[1])] = 1
    
    h_binary2 = np.zeros_like(h_channel)
    h_binary2[(h_channel >= h_thresh2[0]) & (h_channel <= h_thresh2[1])] = 1
    
    h_binary3 = np.zeros_like(h_channel)
    h_binary3[(h_channel >= h_thresh3[0]) & (h_channel <= h_thresh3[1])] = 1
    
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1


    combinedColor = np.zeros_like(s_binary)
    combinedColor[(s_binary == 1) & (h_binary == 1)] = 1
    
    combinedSobL = np.zeros_like(s_binary)
    combinedSobL[(sxLbinary == 1) & (syLbinary == 1)] = 1
    
    combinedGD = np.zeros_like(s_binary)
    combinedGD[(mag_binary == 1) & (dir_binary == 1)] = 1
    
    combinedSobS = np.zeros_like(s_binary)
    combinedSobS[(sxSbinary == 1)&(sySbinary == 1)&(h_binary==1)] = 1
    
    combinedRGB = np.zeros_like(r_binary)
    combinedRGB[(r_binary == 1)&(g_binary == 1) & (b_binary == 1)] = 1
    
    combinedT = np.zeros_like(s_binary)
    combinedT[(combinedColor == 1) | (combinedSobL == 1) | (combinedSobS == 1)] = 1
    
    # Stack each channel
    #color_binary = np.dstack((sxLbinary,mag_binary,dir_binary))*255
    #color_binary = np.dstack((np.zeros_like(mag_binary),mag_binary,dir_binary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),np.zeros_like(sxLbinary),sxLbinary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),s_binary,sxLbinary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),sxLbinary,syLbinary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),mag_binary,dir_binary))*255
    #color_binary = np.dstack((np.zeros_like(sxLbinary),s_binary,s_binary2))*255
    #color_binary = np.dstack((s_binary3,s_binary,s_binary2))*255
    #color_binary = np.dstack((h_binary,s_binary,h_binary2))*255
    #color_binary = np.dstack((h_binary,s_binary,np.zeros_like(h_binary)))*255
    #color_binary = np.dstack((sxLbinary,syLbinary,np.zeros_like(h_binary)))*255
    #color_binary = np.dstack((sxSbinary,sySbinary,np.zeros_like(h_binary)))*255
    #color_binary = np.dstack((mag_binary,dir_binary,np.zeros_like(h_binary)))*255
    #color_binary = np.dstack((combinedColor,combinedSobL,combinedGD))*255
    #color_binary = np.dstack((sxSbinary,sySbinary,h_binary3))*255
    #color_binary = np.dstack((combinedSobS,h_binary3,np.zeros_like(s_binary)))*255
    #color_binary = np.dstack((s_binary4,h_binary3,np.zeros_like(s_binary)))*255
    #color_binary = np.dstack((combinedColor,combinedSobS,combinedSobL))*255
    color_binary = np.dstack((combinedT,combinedRGB,np.zeros_like(s_binary)))*255
    
    combined = np.zeros_like(dir_binary)
    #combined[((sxLbinary == 1) & (syLbinary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) |
    #        (s_binary == 1)] = 1
    #combined[((sxLbinary == 1)&(sxLbinary==1)) | (s_binary == 1)] = 1 
    combined = s_binary2
    
    M,Minv = w.calcM(src,dst)
    # Warp Image
    warped = w.warper(color_binary,M)
    warped2 = w.warper(combined,M)
    warped3 = w.warper(undist,M)
    
    ##Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    
    ax1.imshow(warped3)
    ax1.set_title('Warped Original', fontsize=20)
    
    ax2.imshow(img)
    ax2.set_title('Unwarped Origina', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    f2,(ax3,ax4) = plt.subplots(1,2,figsize=(16, 6))
    ax3.imshow(warped)
    ax3.set_title('Color Map Warped', fontsize=20)

    ax4.imshow(color_binary, cmap='gray')
    ax4.set_title('Color Map Unwarped', fontsize=20)

    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    f3,(ax5,ax6) = plt.subplots(1,2,figsize=(16, 6))
    ax5.imshow(gray, cmap='gray')
    ax5.set_title('Gray', fontsize=20)
    ax6.imshow(l_channel, cmap='gray')
    ax6.set_title('l-channel', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    f4,(ax7,ax8) = plt.subplots(1,2,figsize=(16, 6))
    ax7.imshow(s_channel, cmap='gray')
    ax7.set_title('s-channel', fontsize=20)
    ax8.imshow(h_channel, cmap='gray')
    ax8.set_title('h-channel', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    return

def testWarp(src,dst):
    
    return

def checkLines(LLine,side):
    print("For " + side + " Line:")
    print("Detected: " + str(LLine.detected))
    print("Number of Failed Detections: " + str(LLine.detected_fails))
    print("Number of xFit iterations: " + str(len(LLine.recent_xfitted)))
    print("Number of poly coef fit iterations: " + str(len(LLine.recent_fits)))
    print("Current Fit: " + str(LLine.current_fit))
    print("Best Fit: " + str(LLine.best_fit))
    print("Difference in Fit: " + str(LLine.diffs))
    print("Radius of Curvature: " + str(LLine.radius_of_curvature))
    print("Offset: " + str(LLine.line_base_pos)+"\n")
    #print('\n')
    return

def testPrior(vidFolder,fStart,fEnd,LeftL,RightL,M,Minv,mtx,dist):
    print('Checking Line Class before operation: ')
    checkLines(LeftL,"Left")
    checkLines(RightL,"Right")
    
    #fList = os.listdir(vidFolder+"/")
    #fList.sort()
    #myList = fList[(fStart-1):(fEnd)]
    #print(fList)
    #print(myList)
    
    #nframe = fEnd-fStart+1
    currentF = fStart    
    while(currentF <= fEnd):
        frame = cv2.imread(vidFolder+"/"+"frame"+str(currentF)+".jpg")
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        undist = cam.un_dis(img,mtx,dist)
        binary_warped = gw.getWarpedImg(undist,M)

        fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.tight_layout()
        
        print("FRAME " + str(currentF) + ":")
        
        out_img = gl.fit_polynomial(binary_warped,undist,Minv,LeftL,RightL)
        
        ## Visualization
        ax1.imshow(undist)
        title1 = 'Frame ' + str(currentF) + ' Original'
        ax1.set_title(title1, fontsize=20)    
        
        ax2.imshow(out_img)
        title2 = 'Frame ' + str(currentF) + ' Output'
        ax2.set_title(title2, fontsize=20)   
        ## End Visualization 
        
        print('Checking Lines For Frame ' + str(currentF))
        checkLines(LeftL,"Left")
        checkLines(RightL,"Right")
        
        print('\n')

        
        currentF = currentF + 1
    return

def process_image(image,mtx,dist,LeftL,RightL):
    
    # Undistort Image
    undist = cam.un_dis(image,mtx,dist)
    
    # Create Warped Bianry Image
    binary_warped = gw.getWarpedImg(undist,M)
    
    # Process Binary Waprped Image and Produce Processed Image
    processed_img = gl.fit_polynomial(binary_warped,undist,Minv,LeftL,RightL)
    
    return processed_img

def test_binary(undist):
    binary = gw.getBinary(undist)
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    f.tight_layout()
    
    ax1.imshow(undist)
    ax1.set_title('Original Undistorted Image', fontsize=20)
    
    ax2.imshow(binary,cmap='gray')
    ax2.set_title('Binary Thresholded Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    
    return

### TEST BLOCKS ###
    
## [1] UNDISTORT IMAGE [COMPLETE]
undist = cam.un_dis(img,mtx,dist)                          # Unwarped Image
#testUndistort(img,undist)
#print("Image Displayed")

## [2] PERSPECTIVE TRANSFORM TO BIRDS EYE VIEW [COMPLETE]
src = np.float32([[200,719],[580,460],[700,460],[1100,719]]) # guess source pts
dst = np.float32([[327,719],[327,0],[971,0],[971,719]]) # determine destination pts
# main impact is in choice of destination points, especially how far out
# y is. 

## [3] BINARY THRESHOLD (GRADIENTS, COLOR, ETC.) [COMPLETE]
#src = np.float32([[200,719],[580,460],[700,460],[1100,719]])
#src = np.float32([[204,719],[578,460],[704,460],[1100,719]])
src = np.float32([[195,719],[578,460],[704,460],[1115,719]])
dst = np.float32([[327,719],[327,0],[971,0],[971,719]])
#testBinThresh(undist,src,dst)
M,Minv = w.calcM(src,dst)
binary_warped = gw.getWarpedImg(undist,M)

#test_binary(undist)

## [4] DETERMINE LEFT AND RIGHT LANE PIXLES AND FIT TO POLYNOMIAL [COMPLETE]
LeftL = Line()   # create left line  object
RightL = Line()  # create right line object

out_img = gl.fit_polynomial(binary_warped,undist,Minv,LeftL,RightL)
plt.imshow(out_img)

# test this on a few frames [COMPLETE]
fStart = 1
fEnd = 3
#testPrior(outfolder,fStart,fEnd,LeftL,RightL,M,Minv,mtx,dist)

# NEED TO IMPROVE PERSPECTIVE TRANSFORM FOR BETTER RESULTS

## [5] RADIUS OF CURVATURE AND POSITION OF VEHICLE WITH RESPECT TO CENTER
# do within generateLine


## [6] PLOTTING LANES BACK ON ORIGINAL UNDISTORTED IMAGE

############## PIPELINE VIDEO #############

## MODIFY PIPELINE TO WORK FOR VIDEO 

## [1] INSTEAD OF JUST USING SLIDING WINDOW, USE THE EXISTING LINE IF PREVIOUS
# LANE IS KNOWN [COMPLETE]


## [2] IMPLEMENT LINE() CLASS TO KEEP TRACK OF PRIOR INFORMATION FOR LEFT AND 
#  RIGHT LANE LINES [COMPLETE]

## [3] CREATE CHECK TO VARIFY THAT A LANE LINE WAS DETECTED AND WHAT TO DO IF 
# ONE WAS NOT. [COMPLETE]

## [4] IMPLEMENT WAY FOR VIDEO TO BE RUN THROUGH PIPLELINE FRAME BY FRAME AND 
# FOR THE PLOTTED LINES TO BE SHOWN FRAME BY FRAME. LOOK AT PREVIOUS PROJECT 
# CODE TO DO THIS. 

white_output = 'test_vid_outputs/project_video_processed.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)

#clip1 = VideoFileClip("project_video.mp4") #.subclip(0,5)
##NOTE: this function expects color images!!
#white_clip = clip1.fl_image(lambda image: process_image(image,mtx,dist,LeftL,RightL)) 
#white_clip.write_videofile(white_output, audio=False) # I think this should work


#red_output = 'test_vid_outputs/challenge_video_processed.mp4'
#clip1 = VideoFileClip("challenge_video.mp4") #.subclip(0,5)
##NOTE: this function expects color images!!
#red_clip = clip1.fl_image(lambda image: process_image(image,mtx,dist,LeftL,RightL)) 
#red_clip.write_videofile(red_output, audio=False) # I think this should work



