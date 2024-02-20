# -*- coding: utf-8 -*-
"""
Make overlay demo video from tracking and raw movie
"""

# Set Library Path - Social_Pain Repos
#lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'/Users/alizeekastler/Desktop/Project_Pain_Social/Behaviour_Heat_Gradient'
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'
 
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import glob
import cv2


# Import local modules

import SP_utilities as SPU
import SP_Analysis as SPA
import SP_video_TRARK as SPV
import BONSAI_ARK

# Specify paths

NS_folder = '/NewChamber/Isolated_Habituation_NewChamber/Experiments/2022_05_19/Fish1_28dpf/Non_Social'
movie_file = NS_folder + '/Non_Social.avi'
save_file = NS_folder +  '/Users/Tracking_Overlay.avi'

# Set frame range
start = 78000
end = 86000
step = 1


# Open raw Video
raw_vid = cv2.VideoCapture(movie_file)

# Open save Video
save_vid = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc('F','M','P','4'), 100, (1000,1000))


# Analyze and plot each Fish
for i in range(0,6):

    # Extract tracking data (NS)     
    tracking_file_NS = NS_folder + r'/tracking' + str(i+1) +'.npz'
    fx_NS,fy_NS,bx_NS, by_NS, ex_NS, ey_NS, area_NS, ort_NS, motion_NS = SPU.getTracking(tracking_file_NS)
    

# Overlay tracking

overlay = []
for i, f in enumerate(range(start, end, step)):
    
    # Read next frame and crop
    ret = raw_vid.set(cv2.CAP_PROP_POS_FRAMES, f)
    ret, frame = raw_vid.read()

    # Load tracking position
    bx = bx_NS[f,i]
    by = by_NS[f,i]
    ex = ex_NS[f,i]
    ey = ey_NS[f,i]
    heading = ort_NS[f, i] 
    
    overlay.insert(0, (bx, by, ex, ey, heading)

    # Draw overlay
    for o in overlay:
        
        ret = cv2.circle(ret, (int(o[0]), int(o[1])), 1, (0, 0, 255), 1)
        ret = cv2.circle(ret, (int(o[2]), int(o[3])), 1, (0, 255, 255), 1)
        start_line = (int(o[0]), int(o[1]))
        heading_radians = 2.0 * np.pi * (heading/360.0)
        end_line = (int(o[0]) + int(30 * np.cos(heading_radians)), int(o[1]) + int(-30 * np.sin(heading_radians)))
        ret = cv2.line(ret, start_line, end_line, (0,255,0), 1)
     

    # Write overlay image
    ret = save_vid.write(ret)


# Release videos
raw_vid.release()  
save_vid.release()  
