

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:03:42 2023

@author: alizeekastler
"""

import os
import numpy as np
from PIL import Image
import cv2
from tifffile import imread, imsave

# Build paths
base_folder = '/Volumes/T7/ATLAS/CFOS'
positive_stack = '/Volumes/T7/ATLAS/stacks/CCK/CCK_HEAT_RGB.tif'
negative_stack = '/Volumes/T7/ATLAS/stacks/NPY/Negative_NPY.tif'
DAPI_stack = base_folder +  '/AITC/03_11_23/DAPI_RGB.tif'
output_folder = '/Volumes/T7/ATLAS/stacks/CCK'

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

alpha = np.zeros(np.shape(DAPI_data), dtype=np.float32)

# Blend alpha (positive)
alpha = np.zeros(np.shape(DAPI_data), dtype=np.float32)
beta = np.clip(positive_data[:,:,:,2] /110.0/1, -0.1, 1.0)
alpha[:,:,:,0] = beta
alpha[:,:,:,1] = beta
alpha[:,:,:,2] = beta
front = alpha * positive_data
back = (1-alpha) * DAPI_data
blended_stack = np.uint8(front + back)


# # Blend alpha (negative)
# alpha = np.zeros(np.shape(DAPI_data), dtype=np.float32)
# beta = np.clip(negative_data[:,:,:,0] / 200.0/1, 0.0, 1.0)
# alpha[:,:,:,0] = beta
# alpha[:,:,:,1] = beta
# alpha[:,:,:,2] = beta
# front = alpha * negative_data
# back = (1-alpha) * blended_stack
# blended_stack = np.uint8(front_b + front_r + back)

# Save the result as a TIFF stack
imsave(output_folder + '/Overlay_Heat_CCK.tiff', blended_stack)

