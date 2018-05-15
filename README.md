# Vehicle Detection

![Final Result Gif](./output_images/VehDetection.gif)

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Intro
In this project, our goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product is a detailed writeup of the project. 

## Goals 
The goals / steps of this project are the following:

    *Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
    *Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
    *Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
    *Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
    *Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers 
     and follow detected vehicles.
    *Estimate a bounding box for vehicles detected.


Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and examples extracted from the project video itself. You are welcome and encouraged to take advantage of the recently released [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) to augment your training data.  

[//]: # (Image References)
[image1]: ./output_images/HOG.png
[image2]: ./output_images/HOG2.png
[image3]: ./output_images/SlidingWindow.png
[image4]: ./output_images/HOGsubsampling.png
[image5]: ./output_images/HeatMap.png
[image6]: ./output_images/FinalResult.png
[video1]: ./test_video_out.mp4
[video2]: ./project_video_out.mp4


## Steps

Below are the steps and their explanations that I consider to have a complete pipeline to detect vehicles:

### Histogram of Oriented Gradients (HOG)
I began by loading all of the vehicle and non-vehicle image paths from the provided dataset. The figure below shows a random sample of images from both classes of the dataset.

![alt text][image1]

The code for extracting HOG features from an image is defined by the method `get_hog_features`. The figure below shows a comparison of a car image and its associated histogram of oriented gradients, as well as the same for a non-car image.

![alt text][image2]

I defined parameters for HOG feature extraction and extract features for the entire dataset, also add color transform and append binned color features . These feature sets are combined and a label vector is defined (1 for cars, 0 for non-cars). The features and labels are then shuffled and split into training and test sets in preparation to be fed to a linear support vector machine (SVM) classifier. 

### Settled the final choice of Histogram of Oriented Gradients (HOG), color transform and binned color features
I tried several parameters to get the optimum performance. The below is the parameter sets that I used to create the pipeline:

*color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
*orient = 9  # HOG orientations
*pix_per_cell = 8 # HOG pixels per cell
*cell_per_block = 2 # HOG cells per block
*hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
*spatial_size = (32, 32) # Spatial binning dimensions
*hist_bins = 32    # Number of histogram bins
*spatial_feat = True # Spatial features on or off
*hist_feat = True # Histogram features on or off
*hog_feat = True # HOG features on or off 


### Train the classifier
I trained a linear SVM with the default classifier parameters and using HOG features, color transform and append binned color features and was able to achieve a test accuracy of 99.13%.

### Sliding Window Search
The final find_cars method got adapted from this sliding window search. Here the HOG feature is performed on each window. Later in the 'find_cars' the HOG features are extracted for the entire image (or a selected portion of it) and then these full-image features are subsampled according to the size of the window and then fed to the classifier.

![alt text][image3]

### Hog Sub-sampling Window Search and Heat Map

![alt text][image4]

Here HOG feature are extracted for the entire image which is faster than the sliding window search. Here is one of the test images and I'm showing all the bounding boxes for where my classifier reported positive detections. You can see that overlapping detections exist for each of the two vehicles, and in two of the frames, I find a false positive detection on the guardrail to the left.

To make a heat-map, I simply to add "heat" (+=1) for all pixels within windows where a positive detection is reported by the classifier. The individual heat-maps for the above image look like this:

![alt text][image5]

And the final detection area is set to the extremities of each identified label:

![alt text][image6]

### Video Implementation

#### link to video
Here is the [link to the final video][video2]

#### filter implementation
I used moving average filter on Heat Map to reduce the shaky motion of the rectangles. the WINDOW_SIZE is 3 in this case.

#### Y axis Start/Stop and Scaling
I had use different start-stop for Y axis along with scaling changes.

| Configuration 	|  Y start   |  Y Stop  |   Scale    |
| :-------------------: | :--------: | -------: | ---------: |
| 1                     |   400      |   464    |   1.0      |
| 2                     |   416      |   480    |   1.0      |
| 3                     |   400      |   496    |   1.5      |
| 4                     |   432      |   528    |   1.5      |
| 5                     |   400      |   528    |   2.0      |
| 6                     |   432      |   560    |   2.0      |
| 7                     |   400      |   596    |   3.5      |
| 8                     |   464      |   660    |   3.5      |


## Problems faced during implementation

I faced mainly two issues here- detecting white the car and false positive. I had to use color space as YCrCb and increase the Heat Map threshold to achieve that. But still, I can see some false positive in the final video. If I increase the threshold further I can see it started failing to detect the white car. I stopped here. 

I would say 80% of my time I spent tuning the parameter to get the optimum result. I play with scaling a bit (lower than 1) to detect the vehicle at a far distance. It gives more false positive. I think in this scenerio ,deep learning architecture (YOLO) might give good performace. I would really like give it a try on this in the future. 


