# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 13:39:21 2018

@author: Christian Welling
"""

import cv2

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