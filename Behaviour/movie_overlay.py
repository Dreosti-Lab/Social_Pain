# -*- coding: utf-8 -*-
"""
Make overlay demo video from tracking and raw movie
"""
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import glob
import cv2

# Specify paths

movie_file = '/Users/alizeekastler/Desktop/Social.avi'
tracking_file = '/Users/alizeekastler/Desktop/tracking3.npz'
save_file = '/Users/alizeekastler/Desktop/social_overlay_2.avi'

# Set crop limits
left = 34
top = 262
right = 741
bottom = 376
width = right-left
height = bottom - top

# Set overlay color, size, thickness
color = (0, 0, 255) # Blue, Green, Red
radius = 1
thickness = 2

# Set frame range
start = 78000
end = 86000
step = 1

# Load tracking data
data = np.load(tracking_file)
tracking = data['tracking']
fx = tracking[:,0] 
fy = tracking[:,1]
bx = tracking[:,2]
by = tracking[:,3]
ex = tracking[:,4]
ey = tracking[:,5]
area = tracking[:,6]
ort = tracking[:,7]
motion = tracking[:,8]

# Open raw Video
raw_vid = cv2.VideoCapture(movie_file)

# Open save Video
save_vid = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc('F','M','P','4'), 100, (width, height))

# Overlay tracking
#start = 100
#end = 200
overlay = []
for i, f in enumerate(range(start, end, step)):
    
    # Read next frame and crop
    ret = raw_vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = raw_vid.read()
    crop = frame[top:bottom, left:right, :]

    # Load tracking position
    x = fx[f]
    y = fy[f]
    overlay.insert(0, (x-left,y-top))

    # Draw overlay
    for o in overlay:
        ret = cv2.circle(crop, (int(o[0]), int(o[1])), radius, color, thickness)

    # Write overlay image
    ret = save_vid.write(crop)


# Release videos
raw_vid.release()  
save_vid.release()  
