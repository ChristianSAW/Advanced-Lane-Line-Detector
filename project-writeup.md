## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output/original_vs_undistorted.png "Original And Undistorted Images Compared"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output/original_vs_binary.jpg "Original Undistorted Image And Binary Thresholded Image"
[image4]: ./output/perspective_transform.jpg "Before And After Warped Image"
[image5]: ./output/fit_polynomial.jpg "Fit Visual With Sliding Histogram"
[image6]: ./output/test2_output.jpg "Example Output"
[video1]: ./test_vid_outputs/project_video_processed.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `cameraCalibrate.py`

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

See the aformentioned image to see an original distorted image and the resultant undistorted image after the undistortion process is applied to it.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines #89 through #169 in `getWarped.py`).  Here's an example of my output for this step. To summarise what thresholding was used: color thresholding of the s and h channels, gradient thresholding of the l channel in the x and y direction, and gradient thresholding of the s channel in the x and y direction. 

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 17 through 22 in the file `warper.py`. The `warper()` function takes as inputs an image (`img`) as well as the image matrix M. M is calculated with `cv2.getPerspectiveTransform()` using source (`src`) and destination (`dst`) points as inputs.  I chose the hardcode the source and destination points using a trial and error method with given test images. The following were the final acceptable results:

```python
src = np.float32([[195,719],[578,460],[704,460],[1115,719]])
dst = np.float32([[327,719],[327,0],[971,0],[971,719]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 195, 719      | 327, 719      | 
| 578, 460      | 327, 0        |
| 704, 460      | 971, 0        |
| 1115, 719     | 971, 719      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for the identification takes place within `generateLine.py`. Function `fit_polynomial()` is the main function which takes in a binary warped input, and then identifies the lane lines. 

Left and right lane pixles are determined one of two ways: sliding histogram or based off prior. Function `find_lane_pixels()` details how the sliding histogram method works and function `search_around_poly()` details how the left and right lane pixles are determined from the prior lane line estimation.

once the left and right lane pixles are determiend, they are fit to left and right lines in the `fit_poly()` function. This function approximates a second order line for each left and right lane using `numpy.polyfit()`.

An image with the fit lane lines looks like this: 

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines #436 through #440 allong with functions `calcCurvature()` and `calcOffset()`in my code in `generateLine.py`. 

For each line, curvature is calculated using the curvature formula in line #347 of `generateLine.py`. The curvature formula uses polynomial coefficeints that were generated using metric units instead of pixles (as in line #341), and calculates the curvature along the y axis (coverted to metric units). Then the average curvature allong the y axis is used. 

When reporting the curvature for the vehicle, a weighted average of the curvature for each line is used. The weighted average weighs more heavily towards the lane line (left or right) which is more consistantly detected. See lines #316 to #322 for how this is implemented.

In calculating the position of the vehicle with respect to the center (i.e. offset), the difference between the car center (x center of image) and the lane center (average x position of the bottom points of the left and right lanes). This calculation is seen in lines #358 to #366. 

Assuming a lane length is approximately 30 meters and takes up 720 pixles, and that the lane width is approximately 3.7 meters and takes up 700 pixles, the following conversion rates are used,

```python
ym_per_pix = 30/720                # [meter/pixel] conversion for y axis 
xm_per_pix = 3.7/700               # [meter/pixel] conversion for x axis 
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #63 through #71 in my code in `laneLineFinder_SingleFrame.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_vid_outputs/project_video_processed.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took for the video implementation was as such: 1) genrate line objects LeftLane, RightLane, 2) pass line objects and each frame to process_image() function, 3) process image and update line objects, 4) save processed frame and repeat. 

See `Line.py` for what each line object stored. The image processign pipeline went as follows: 

 1) generate the undistorted image, 2) generate the binary warped image, 3) generate left and right line points, 3) fit left and right lines to 2nd order polynomial, 4) calculate offset and curvature, 5) check if generated lines are valid and update line objects appropriatly, 6) display curvature, offset, and left and right lane lines based off information in the line objects.

 For a more detailed overview, follow the function `process_image()` in `laneLineFinder_Video.py`. 

 In creating the binary image, I used various color and gradiant thresholding techniques. I spent a lot of time optimizing the thresholds to work for the test images and the project_video. 
 
 To improve results, I think I could generate a binary image on a warped image instead of warping after the binary image is created. This would help increase the fidelity of relevant points in the image which are further away from the camera origin.

 Additionally, when I determine the source and destination points for the perspective transform, I assumes the car is in the center of the lane when in reality it is not. Thus, even if I get a good transformation for one image, other image perspective transformations will not be as good since the car offset from the center of the lane will change. The effect of this can be seen when using a perspective transform on two diffent test images with straight lines. One can tune the source and destination points for one image to get a transformation with straight lines, but then when using those same source and destination points on another test image, the straight lane lines will be slightly curved in the warped image. I'm not entirely sure how to improve this. 

 Another problem with my approach is that I assumes that both left and right lanes will be in the image, which is not always the case. Additionally, the generation of the left and right line points assumes that the left and right lane lines are far away enough from other lines which would fall within the search margin. The search margin is currently a fixed value, but depending on the lines being detected, the current value may not work. lines that do not correspond to the lane lines might be picked up instead with the current margin. Making the margin smaller might improve this.

If I were to pursue this further, I would find a way to improve my lane line verification method. I would additionally look into changing my process such that if only one lane line is detected, this is sufficient to determine a lane assuming prior data contained two lane lines. Currently this pipeline works well with project_video.mp4 but fails with challenge_video.mp4 and harder_challenge_video.mp4. 
