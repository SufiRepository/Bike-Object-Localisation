# Bike-Object-Localisation

This is a deep learning group projects about bike detection using object detection CNN.

# Introduction

Cycling is a low impact aerobic exercise that offers a wealth of benefits. It also varies in intensity, making it suitable for all levels. Vehicle and pedestrian detection attracted lots of attention in the course of improving transportation safety. In recent, the safety of cyclists is of a greatest concern because cyclist accidents are gradually increasing in numbers. Our group decided to implement computer vision and machine learning detection methods, tracking algorithms and trajectory analysis for cyclists in traffic video data and developing an efficient system for cyclist counting.

# Problem Statement

In recent years, the protection of vulnerable road users, mainly pedestrians and cyclists has become one of the significant importance of department of transportation. Due to the growing number of cyclist accidents on roads, methods for collecting information on cyclists are of significant importance to the Ministry of Transportation.

# Objective

- To describe the locations of bikes in the image using a bounding box.
- To give awareness to driver on the road for any bicycle.
- To give the data on enforcers for illegal cyclists on the public road.

# Project Idea

In this work, a vision-based method for gathering cyclist count data at intersections and road segments is developed. First, we develop methodology for an efficient detection and tracking of cyclists. The combination of classification features along with motion based properties are evaluated to detect cyclists in the test video data. A Convolutional Neural Network (CNN) based detector called You Only Look Once (YOLO) is implemented to increase the detection accuracy.

# Limitation

Factors contributing to the complexity of the problem include appearance similarity of the upper body of a pedestrian and a cyclist that may lead to misdetection and subsequently to the wrong count in the real road environment. Other factors include variety of poses in the field of view depending on the camera location, illumination changes under day/night and environmental changes, and occlusion of the target objects. The lack of a sufficient image resolution can cause misdetections and may lead to failure of tracking. 

# Possible Improvement

For improving the performance of YOLO detector, the detector should be trained with a larger dataset. Therefore, it is better to collect more cyclist data for future training. Monitoring systems presented in the literature are designed to work under the day light conditions. Enhancing the ability of the system to detect and track cyclists at night by analyzing thermal images.

# Dataset  : 

https://www.kaggle.com/johnmdennis/mountain-bikes?select=download+%2841%29.jpg  (mountain bike)

https://drive.google.com/file/d/1_x_Nqpjtvxp22ADExyumrRbawC9O4FCE/view?usp=sharing   (general bicycle)

# Reference

https://github.com/brandonjabr/darknet-YOLO-V2-example
https://www.researchgate.net/profile/Farideh_Foroozandeh_Shahraki2/publication/341256204_Cyclist_Detection_Tracking_and_Trajectory_Analysis_in_Urban_Traffic_Video_Data/links/5eb5fbeb92851cd50da3934c/Cyclist-Detection-Tracking-and-Trajectory-Analysis-in-Urban-Traffic-Video-Data.pdf
https://carpenterlab.broadinstitute.org/blog/annotating-images-with-cellprofiler-and-gimp
https://www.researchgate.net/publication/337464355_OBJECT_DETECTION_AND_IDENTIFICATION_A_Project_Report

