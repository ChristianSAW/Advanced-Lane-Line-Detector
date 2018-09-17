# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 11:42:16 2018

@author: Christian Welling
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


A = np.array([1,2,3,4,5])
print(A[-1])
B = np.array([11,12,13,14,15])
C = np.average([A,A,B],0)
D = [A,B,A]
#E = np.average([[],A],0)
print(C)
#print(E)

A = 10
B = 15
C = np.average([A,B])
print(C)

#image = cv2.imread('test_images/test2.jpg')
#img = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#
#image2 = cv2.imread('test_images/test1.jpg')
#img2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
#
#print("Now doing first image")
##plt.figure()
#f, (ax1,ax2) = plt.subplots(1,2, figsize=(16, 6))
#
#ax1.imshow(img)
#ax1.set_title('Original 1', fontsize=20)
#
#ax2.imshow(img)
#ax2.set_title('Original 2', fontsize=20)
#
#print("Now Doing 2nd Image")
#
##plt.figure()
#f, (ax1,ax2) = plt.subplots(1,2, figsize=(16, 6))
#
#ax1.imshow(img2)
#ax1.set_title('Original 1', fontsize=20)
#
#ax2.imshow(img2)
#ax2.set_title('Original 2', fontsize=20)

#fig, axs = plt.subplots(nframe, 2, figsize=(16, 6))
#fig.tight_layout()

## Testing Conditional Logic

# What I was originally using
AL = [True,True,False,False]
BL = [True,False,True,False]

print("Actual Results")
for i in range(0,4,1):
    A = AL[i]
    B = BL[i]
    print("A: " + str(A) + ", B: " + str(B))
    if ~(A|B):
        print("Passed")
    else:
        print("Failed")

print("Desired Results: ")
print("Failed")
print("Failed")
print("Failed")
print("Passed")

# Result Failed 

# What I was originally usin
AL = [True,True,False,False]
BL = [True,False,True,False]

print("\n Attempt 2 \n")
print("Actual Results")
for i in range(0,4,1):
    A = AL[i]
    B = BL[i]
    print("A: " + str(A) + ", B: " + str(B))
    if not (A | B):
        print("Passed")
    else:
        print("Failed")

print("Desired Results: ")
print("Failed")
print("Failed")
print("Failed")
print("Passed")

A = 1000
B = 500
C = np.min([A,B])
print(C)

side = "right"
offset = 1.343457

offText = "Vehicle is %.2fm " % np.absolute(offset) + side + " of center"

print(offText)