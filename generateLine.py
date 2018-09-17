# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:13:58 2018

@author: Christian Welling
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
from Line import Line              # Line Class

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    return left_fit,right_fit,left_fitx, right_fitx, ploty

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    # Create an output image to draw on and visualize the result
    # NOTE, BINARY_WARPED HAS RANGE 0-255, WHITE POINTS HAVE 255, BLACK HAVE 0
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
        
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  
        win_xleft_high = leftx_current+margin 
        win_xright_low = rightx_current-margin
        win_xright_high = rightx_current+margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & 
        (nonzeroy < win_y_high) & (nonzeroy >= win_y_low)).nonzero()[0]
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & 
        (nonzeroy < win_y_high) & (nonzeroy >= win_y_low)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, margin, out_img

def search_around_poly(binary_warped,LineL,LineR):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    
    # Grab priror line fit
    left_fit = LineL.current_fit
    right_fit = LineR.current_fit
    
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox < (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]+margin)) & 
                     (nonzerox >= (left_fit[0]*nonzeroy**2 + left_fit[1]*nonzeroy + left_fit[2]-margin)))
    right_lane_inds = ((nonzerox < (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]+margin)) & 
                      (nonzerox >= (right_fit[0]*nonzeroy**2 + right_fit[1]*nonzeroy + right_fit[2]-margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    
    return leftx, lefty, rightx, righty, margin, out_img

def vizA(out_img,lefty,leftx,righty,rightx,left_fitx,right_fitx,ploty):
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return out_img

def vizB(out_img,lefty,leftx,righty,rightx,left_fitx,right_fitx,ploty,margin):
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    return result

def vizC(out_img,lefty,leftx,righty,rightx,LineL,LineR,ploty,margin):
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(out_img)
    
    # Color in left and right line pixels
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    
     # Pull from Line object
    left_fitx = LineL.bestx
    right_fitx = LineR.bestx

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##
    return result
    

def vizD(lefty,leftx,righty,rightx,left_fitx,right_fitx,ploty,margin,
         undist,binary_warped,Minv):
    
    # Create a blank image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Draw the detected left and right lane line points    
    color_warp2[lefty, leftx] = [255, 0, 0]
    color_warp2[righty, rightx] = [0, 0, 255]
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    newwarp2 = cv2.warpPerspective(color_warp2, Minv, (undist.shape[1], undist.shape[0])) 


    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(result,1,newwarp2,0.7,0)
    
    return result

def vizSmooth(lefty,leftx,righty,rightx,LineL,LineR,ploty,undist,
              binary_warped,Minv):
    
    # Pull from Line object
    left_fitx = LineL.bestx
    right_fitx = LineR.bestx
    
    # Create a blank image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    color_warp2 = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Draw the detected left and right lane line points    
    color_warp2[lefty, leftx] = [255, 0, 0]
    color_warp2[righty, rightx] = [0, 0, 255]
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 
    newwarp2 = cv2.warpPerspective(color_warp2, Minv, (undist.shape[1], undist.shape[0])) 


    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    result = cv2.addWeighted(result,1,newwarp2,0.7,0)
    
    ## Add Text For Curvature and Offset
    # determine offset
    if (LineL.detected_fails > LineR.detected_fails):
        offset = LineR.line_base_pos
    else:
        offset = LineL.line_base_pos
    
    if (offset <= 0):
        side = "left"
    else:
        side = "right"
        
    offText = "Vehicle is %.2f(m) " % np.absolute(offset) + side + " of center"
    
    # determine curvature
    LL_rC = LineL.radius_of_curvature
    LR_rC = LineR.radius_of_curvature
    
    nL = 1/(1+LineL.detected_fails)
    nR = 1/(1+LineR.detected_fails)
    
    rC = (1-(nR/(nL+nR)))*LL_rC + (nR/(nL+nR))*LR_rC
    
    rCText = "Radius of Curvature = %.0f(m)" % rC
    
    # print text on 
    #txt = rCText + "\n" + offText
    cv2.putText(result,rCText,(100,100),cv2.FONT_HERSHEY_SIMPLEX,2,
                (255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,offText,(100,170),cv2.FONT_HERSHEY_SIMPLEX,2,
                (255,255,255),2,cv2.LINE_AA)
    
    return result

def calcCurvature(ploty,xvals,yvals):
    # Conversion from pixels to meters 
    ym_per_pix = 30/720                # [meter/pixel]
    xm_per_pix = 3.7/700               # [meter/pixel]
    
    # Find polynomial coefficients using metric units
    fit_cr = np.polyfit((ym_per_pix*yvals),(xm_per_pix*xvals),2)
    
    # Generate y values to evaluate at
    y_eval = ploty*ym_per_pix
    
    # calculate curvature
    rC = ((1+(2*fit_cr[0]*y_eval+fit_cr[1])**2)**1.5)/np.absolute(2*fit_cr[0])

    rC = np.average(rC)
    return rC

def calcOffset(widthPix,left_fitx,right_fitx):
    # Conversion from pixels to meters 
    ym_per_pix = 30/720                # [meter/pixel]
    xm_per_pix = 3.7/700               # [meter/pixel]
    
    # Center of Car
    carCenter = (xm_per_pix*widthPix)/2
    
    # Center of Lane
    laneCenter = np.average([left_fitx[-1],right_fitx[-1]])*xm_per_pix
    
    # Calculate offset
    # positive value: car is to right of lane center
    # negative value car is to left of lane center
    offset = carCenter-laneCenter
    
    return offset

def maxDmin(A,B):    
    return np.max([A,B])/np.min([A,B])

def verifyDetection(LLine,LLineO,rC,rC_other,offset,threshRc,threshOfs):
    ratioRc= 2
    ratioRc_p = 1.5
    # pull some values to save line space
    LL_rC = LLine.radius_of_curvature
    LLO_rC = LLineO.radius_of_curvature
    LL_ofs = LLine.line_base_pos
    
    verified = True
    
    # if both lines have not been detected before or failed detections for
    # 3 or more times, results must be verified. Stop here if true.
    if not (LLine.detected | LLineO.detected):
#        print("Both Lines Not Detected")
        return verified
    
    # Check offset
    # if offset from prior is outside of threshold.
    elif(np.absolute(LL_ofs - offset)>threshOfs):
#        print("Offset Failure, Previous offset: " + str(LL_ofs) + ", Current offset: " + str(offset))
        verified = False
    
#    # Threshold Check radius of curvature consistancy and line parallelism
#    # if not (curvature consistancy or ((other lane curvature consistancy) and 
#    #(parallel lanes))), then is not verified 
#    elif not ((np.absolute(LL_rC-rC) < threshRc)|
#            ((np.absolute(LLO_rC-rC_other) < threshRc) &
#             (np.absolute(rC-rC_other) < threshRc))): 
#        print("Curvature and Parallelism Failure")
#        print("Previous Curvature: " + str(LL_rC) + ", Current Curvature: " + str(rC))
#        print("Other Line Previous Curvature: " + str(LLO_rC) + ", Other Line Current Curvature: " + str(rC_other))
#        verified = False
        
    # Ratio Check radius of curvature consistancy and line parallelism
    # if not (curvature consistancy or ((other lane curvature consistancy) and 
    #(parallel lanes))), then is not verified 
    elif not ((maxDmin(LL_rC,rC) < ratioRc)|
            ((maxDmin(LLO_rC,rC_other) < ratioRc) & 
             (maxDmin(rC,rC_other) < ratioRc_p))): 
#        print("Curvature and Parallelism Failure")
#        print("Previous Curvature: " + str(LL_rC) + ", Current Curvature: " + str(rC))
#        print("Other Line Previous Curvature: " + str(LLO_rC) + ", Other Line Current Curvature: " + str(rC_other))
        verified = False
        
    return verified 

def fit_polynomial(binary_warped,undist,Minv,LineL,LineR):
    # Find our lane pixels first
    
    # USE EITHER SLIDING WINDOW OR PRIOR HERE
    # if both left and right lane lines are not detected. 
    if not ((LineL.detected)|(LineR.detected)):
        leftx, lefty, rightx, righty, margin, out_img = find_lane_pixels(binary_warped)
    else: 
        leftx, lefty, rightx, righty, margin, out_img = search_around_poly(binary_warped,LineL,LineR)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    # Fit new polynomials
    left_fit,right_fit,left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, 
                                                               rightx, righty)
    
    ### CALCULATE CURVATURE + OFFSET
    # curvature
    left_rC = calcCurvature(ploty,leftx,lefty)
    right_rC = calcCurvature(ploty,rightx,righty)
    
    # offset
    offset = calcOffset(binary_warped.shape[1],left_fitx,right_fitx)
    
    ## Verify that new line was detected 
    threshRc = 500
    threshOffset = 0.15
    
#    print("Left Line Verification: ")    
    verLeft = verifyDetection(LineL,LineR,left_rC,right_rC,offset,threshRc,
                              threshOffset)
    ## COMMENT OUT AFTER TESTING
#    if (verLeft):
#        print("Verified Left")
#    else:
#        print("Unverified Left")
#    
#    print("Right Line Verification: ")        
    verRight = verifyDetection(LineR,LineL,right_rC,left_rC,offset,threshRc,
                              threshOffset)
    ## COMMENT OUT AFTER TESTING
#    if (verRight):
#        print("Verified Right")
#    else:
#        print("Unverified Right")
    
    ## UPDATE LINE
    if (verLeft):
        LineL.update(leftx,lefty,left_fit,left_fitx,ploty,left_rC,offset)
#        print("Verified Left")
    else:
        LineL.fail()
#        print("Unverified Left")
    
    if (verRight):
        LineR.update(rightx,righty,right_fit,right_fitx,ploty,right_rC,offset)
#        print("Verified Right")
    else:
        LineR.fail()
#        print("Unverified Right")
        
    # ASSUME TRUE FOR NOW
    
    ## UPDATE LINE
    #LineL.update(leftx,lefty,left_fit,left_fitx,ploty,left_rC,offset)
    #LineR.update(rightx,righty,right_fit,right_fitx,ploty,right_rC,offset)

    ## Visualization ##
    viz_option = 4
    if(viz_option == 0):
        final_img = vizA(out_img,lefty,leftx,righty,rightx,left_fitx,
                         right_fitx,ploty)
    elif(viz_option == 1):
        final_img = vizB(out_img,lefty,leftx,righty,rightx,left_fitx,
                         right_fitx,ploty,margin)   
    elif(viz_option == 2):
        final_img = vizC(out_img,lefty,leftx,righty,rightx,LineL,LineR,ploty,
                         margin)
    elif(viz_option == 3):
        final_img = vizD(lefty,leftx,righty,rightx,left_fitx,
                         right_fitx,ploty,margin,undist,binary_warped,Minv)
    else:
        final_img = vizSmooth(lefty,leftx,righty,rightx,LineL,LineR,ploty,
                              undist,binary_warped,Minv)
    ## End visualization steps ##

    return final_img
