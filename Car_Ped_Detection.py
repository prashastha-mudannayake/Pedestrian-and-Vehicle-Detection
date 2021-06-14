# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 00:46:04 2021

@author: prashastha
"""

import cv2

# importing cascade classifier file with haar features
car_class = 'car_classifier.xml'
ped_class = 'ped_classifier.xml'

# importing video
video = cv2.VideoCapture('vehicles_and_pedestrian.mp4')

# generating cascade classifier 
car_tracker = cv2.CascadeClassifier(car_class)
ped_tracker = cv2.CascadeClassifier(ped_class)

# loop through video
while True:
    
    # read each frame
    (read_successful, frame) = video.read()
    
    if read_successful:
        # convert each frame to greyscale
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break
    
    # apply classifier for identification
    # will be returning a list of arrays (x_coordinate, y_coordinate,
    # width, height)
    cars = car_tracker.detectMultiScale(grayscale_frame)
    peds = ped_tracker.detectMultiScale(grayscale_frame)
    
    # green bounding box
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # red bounding box    
    for (x, y, w, h) in peds:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
    cv2.imshow('Pedestrian and Vehicle Detection Device', frame)
    cv2.waitKey(4)
    
cv2.waitKey(1)
cv2.destroyAllWindows()