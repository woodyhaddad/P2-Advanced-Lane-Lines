## Project: Advanced Lane Lines

### In this project, the goal is to write a software pipeline to identify the lane boundaries in a video. 

---

** Pipeline steps: **

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undistorted_Calibration_Image.jpg "Undistorted Calibration"
[image2]: ./output_images/Undistorted_Test_Image.jpg "Undistorted Test"
[image3]: ./output_images/Combined_Gradient.jpg "Combined Gradient"
[image4]: ./output_images/Combined_Color.jpg "Combined Color"
[image5]: ./output_images/Color_and_gradient_thresholded.jpg "Color and Gradient Thresholded"
[image6]: ./output_images/perspective_transform.jpg "Warped Image"
[image7]: ./output_images/Sliding_Window_polynomial.jpg "Sliding Window"
[image8]: ./output_images/identified_lane_image.jpg "Identified lane"
[video1]: ./Project_Video_Output.mp4 "Project Video"
[video2]: ./Challenge_Video_Output.mp4 "Challenge Video Output.mp4"
[video3]: ./Harder_Challenge_Video_Output.mp4 "Harder Challenge Video Output.mp4"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

#### First of all, of course, I imported all the libraries I will be working with: ‘matplotlib’, ‘numpy’, ‘OpenCV cv2’, ‘os’, and ‘pylab’

### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook ‘P2.ipynb’ located in the main directory of the project folder.  

I started by defining the path locally and setting up the images. In order to use open cv’s ‘cv2.calibrateCamera’ to calibrate our camera, I need to find ‘objpts’, which are hypothetical chess board corners that are found based on a 2D representation of a chessboard of the same size (in columns and rows) as the one in the images I am using in order to calibrate the camera (cell 3, lines 16-18). I also need to find ‘imgpoints’ which are the physical corners on the images used for calibration. There is a function ‘cv2.findChessboardCorners()’ that computes the corners of a chess grayscale image.   

Next, all calibration images in the ‘camera_cal’ folder are iterated through and all ‘objpts’ and ‘imgpoints’ are extracted. ‘cv2.calibrateCamera()’ is ready to use. Note that camera calibration needs to happen only once for this entire project since the same camera is used everywhere. ‘cv2.undistort’ takes as inputs and image, the camera matrix and distortion coefficients matrix both of which are outputs of ‘cv2.calibrateCamera’. ‘cv2.undistort’ undistorts a given image. Example of an undistorted calibration image is below 

![alt text][image1]

### Pipeline (single images)

#### 1. Distortion correcting test images.

The following cell applies distortion correction to the test images in the ‘test_images’ folder. Here’s what an example image should look like after distortion correction:

![alt text][image2]

#### 2. Creating a thresholded binary image via gradients and color detection.

I used a combination of color and gradient thresholds to generate a binary image (thresholding code at the next 3 cells in the Jupyter notebook). I defined a function ‘abs_sobel_thresh()’ that takes in an image and calculates directional gradient based on the Sobel operator. Orientation of the gradient is in the ‘x’ direction by default but can be changed to ‘y’. Output is a thresholded image (thresholds are specified in the inputs). ‘mag_thresh()’ is similar to ‘abs_sobel_thresh()’ but it creates the thresholded image based on the magnitude of the directional gradient (using the magnitude equation for a right triangle). ‘dir_threshold’ is similar, but the thresholds are dependent on the direction (angle) of the gradient. Cell 5, lines 71 till the end of the cell, I found a thresholded image based on all the previous gradient functions. Here’s what the output, ‘Combined_Gradient’ looks like:

![alt text][image3]

The following defines a function ‘hls_saturation’ which applies thresholding based on the Saturation and Lightness channels of an image. The input should be RGB and the function will convert to HLS and apply the necessary steps to output a thresholded binary image based on the input image and threshold values given. See ‘Combined_Color’ below for a representation of the function output:

![alt text][image4]

The following cell defines a function ‘Combined_colorgradient’ that only needs an RGB image as input, and outputs a thresholded binary image based on both color and gradient thresholds. As you can see in the image ‘Color_and_gradient_thresholded’ below, combining both color and gradient methods provides more reliable thresholding.

![alt text][image5]

#### 3. Perspective transform.

The following cell performs a perspective transform in order to warp the undistorted image and get a “top view” of the lane lines. I used a ‘try’ ‘except’ clause to make it easy to apply the pipeline to ‘process_image()’, the function used for video processing. ‘src’ is the variable storing the source points for the transform, which are the four points on the original image that I know should represent a rectangle. The way to go about this is to use a straight lane image and pick 4 points that form a trapezoid with top and bottom horizontal and the sides parallel to the lane lines. ‘dst’ are destination points. These are more easily found as there are quite a few options to go with here. The main requirement to find these is that the 4 points need to form a rectangle.
‘perspective_transform’ is a binary image of the warped view just explained. An easy way to know that the transform is appropriate is to check if the lane lines are roughly parallel (as they should be)

![alt text][image6]

#### 4. Lane line pixel identification and polynomial fit

The next cell uses the sliding window approach discussed in the class lessons in order to identify the left and right lanes. ‘find_lane_pixels’ takes in a binary warped image and outputs left lane activated pixel coordinates (x and y) and an RGB image with both lanes colored (left lane in red, right lane in blue) and with the sliding windows visualized. This function serves as prep for the next function, ‘fit_polynomial’, which takes in a binary warped function, applies ‘find_lane_pixels’ to it and uses numpy’s ‘polyfit’ function in order to fit a 2nd order polynomial to each of the two lanes. The function also plots the fit so that the user sees if it is suitable. As you can see in ‘Sliding_Window_polynomial’, the polynomials appear in the appropriate locations.

![alt text][image7]

#### 5. Radius of curvature and vehicle offset from center of lane

The following cell defines the function ‘measure_curvature_real’ that takes in a binary warped image and outputs the left line radius of curvature, and the right line radius of curvature in units of meters. The function uses numpy’s polyfit and finds the radius of curvature of the polynomial at the bottom position (closest to the car). ‘ym_per_pix’ and ‘xm_per_pix’ are the conversion factors necessary in order to convert the radius from pixels to meters.

The cell after that defined the function ‘vehicle_offset’, which also takes as input a binary warped image and outputs the difference in meters between the middle of the camera i.e, the middle of the car, and the middle of the lane, which is simply the midpoint between the two polynomial evaluated at the bottom of the image.

#### 6. Unwarped image with lane lines identified.

This last step (following cell) visualizes the values calculated in step 5 (radius or curve and distance from center of lane) on an output image. The output image also plots the lane found from the operations performed on the warped image. The function ‘unwarp_image’ takes in a binary warped image and an undistorted, RGB image and outputs an image with the lanes identified and with text that states the values of the radius of curvature and distance offset of the car from the center of the lane. The warped image was “unwarped” using the same OpenCV function ‘getPerspectiveTransform’, but the ‘dst’ and ‘src’ inputs were reversed. This way, we will get ‘Minv’ (line 27), the inverse matrix of the transformation that will transform the warped image back to an unwarped state. The radius of curvature is the average between left and right lane. Lines 40 to 42 define whether we should type ‘left’ or ‘right’ on the output image (see image ‘identified_lane_image’ below). ‘cv2.fillpoly’ was used on the warped image in order to create a shaded area to represent the lane, and once unwarped, ‘cv2.addWeighted’ is a good function to use in order to visualize the lane lines on the original, undistorted image (line 32).

![alt text][image8]

---

### Pipeline (video)

#### 1. Project Video Results
Here's a [link to my video result](./test_videos_output/Project_Video_Output.mp4)

---

### Discussion

#### 1. My pipeline works pretty well on the project video, however, it will not work on the challenge videos. The easier challenge video cannot iterate through all the frames and will actually crash at a certain frame. This is because at that frame, one (or both) of the lanes did not identify any pixels so a TypeError was raised inside the ‘polyfit’ function. Another problem I faced was with the harder challenge video, where the lane lines were simply too far off due to poor lighting and camera position conditions, amongst other things (lanes are hidden by leaves sometimes, etc). A first step that can be taken in order to improve this code is to actually keep record of previous frames and average the values from previous frames and/ or use the “search around line” method in order to find lane lines. 
