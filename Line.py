# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 19:29:41 2018

@author: Christian Welling
"""

import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        
        # number of failed detections
        self.detected_fails = 0
        
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        
        #polynomial coefficients for last n fits
        self.recent_fits = []
        
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None 
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        #x values for detected line pixels
        self.allx = None  
        
        #y values for detected line pixels
        self.ally = None  
        
    def update(self,xvals,yvals,fit,xfit,yfit,rC,offset):
        # Update detected state
        self.detected = True
        
        # Update number of detected fails
        self.detected_fails = 0
        
        # Update current x and y values for detected line pixles 
        self.allx = xvals
        self.ally = yvals
        
        # Update offset
        self.line_base_pos = offset
        
        # Update radius of curvature
        self.radius_of_curvature = rC
        
        # Update x values of last n fits, average x values, difference in fit,
        # recent fit coefficient, fit coefficients of last n fits, and 
        # average fit coefficients
        n = 3                           #  number of iterations for averaging
        self.updateFit(n,xfit,yfit,fit)
        
        return
    
    def updateFit(self,n,xfit,yfit,fit):
        # Update difference in fit coefficients
        newFit = fit #np.array(fit,dtype='float')
        if (len(self.recent_fits) < 1):
            self.diffs = newFit
        else:
            self.diffs = newFit - self.current_fit
        
        # Update new current fit
        self.current_fit = newFit
        
        # Append current_fit and current xfits
        self.recent_xfitted.append(xfit)
        self.recent_fits.append(newFit)
        
        # Pop Lists if too long (LIFO)
        if (len(self.recent_fits) > n):
            self.recent_fits.pop(0)
            self.recent_xfitted.pop(0)
        
        if (len(self.recent_fits) > 1):
            # Update average x values of the fitted line over last n iterations
            self.bestx = np.average(self.recent_xfitted,0)
        
            # Update average polynomial coefficients over last n iterations
            self.best_fit = np.average(self.recent_fits,0)
        else:
            self.bestx = xfit
            self.best_fit = newFit
        return
    def fail(self):
        nDF = 3
        
        self.detected_fails = self.detected_fails + 1
        
        if self.detected_fails >= nDF:
            self.detected = False
            
        return
    