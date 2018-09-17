# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 12:29:06 2018

@author: Christian Welling
"""

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
#%matplotlib qt


##  Calculates calibration coefficients 
def calib_cam():
# prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners (ONLY FOR TESTING)
            #img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    #print(objpoints)
    #print(imgpoints)
    cv2.destroyAllWindows()

    # Calibrate Camera
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img.shape[1::-1],None,None)
    
    return mtx,dist

## Undistorts image given calibration coeifficients 
def un_dis(img,mtx,dist):
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    return undist