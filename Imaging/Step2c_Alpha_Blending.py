#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:03:42 2023
This script is used to overlay raw RGB tiff of cfos for example onto a white DAPI background. 
@author: alizeekastler
"""

import os
import numpy as np
from PIL import Image
import cv2
from tifffile import imread, imsave

# Build paths
base_folder = 'F:/ATLAS/CFOS/Heat/14_10_23/'
positive_stack = base_folder + 'Positive_green.tif'
negative_stack = base_folder + 'Negative_RGB.tif'
DAPI_stack = base_folder +  'DAPI_RGB.tif'
output_folder = base_folder + '/output/'

# Create output folder (if it does not exist)
if not os.path.exists(output_folder):
   os.makedirs(output_folder)

# Load postive and negative stacks
positive_data = np.float32(imread(positive_stack))
negative_data = np.float32(imread(negative_stack))
num_frames, height, width, channels = np.shape(positive_data)

# Load DAPI stack
DAPI_data = np.float32(imread(DAPI_stack))
num_frames, height, width,channels = np.shape(DAPI_data)

# Blend alpha (positive)
alpha = np.zeros(np.shape(DAPI_data), dtype=np.float32)
beta = np.clip(positive_data[:,:,:,1] / (255.0 / 1.0), 0.0, 1.0)
alpha[:,:,:,0] = beta
alpha[:,:,:,1] = beta
alpha[:,:,:,2] = beta
front = alpha * positive_data
back = (1-alpha) * DAPI_data
blended_stack = np.uint8(front + back)

# # Blend alpha (negative)
# alpha = np.zeros(np.shape(DAPI_data), dtype=np.float32)
# beta = np.clip(negative_data[:,:,:,0] / (210.0 / 1.0), 0.0, 1.0)
# alpha[:,:,:,0] = beta
# alpha[:,:,:,1] = beta
# alpha[:,:,:,2] = beta
# front = alpha * negative_data
# back = (1-alpha) * blended_stack
# blended_stack = np.uint8(front + back)

# Save the result as a TIFF stack
imsave(output_folder + 'Overlayed_green.tiff', blended_stack)

