#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 21:01:51 2023

@author: alizeekastler
"""


import numpy as np
import cv2
from tifffile import imread, imsave

base_folder = 'F:/ATLAS/CFOS/Heat/27_06_23/'


# Load the TIFF stacks as NumPy arrays
dapi_stack = np.array(imread(base_folder + 'DAPI_2.tif'))
negative_stack = np.array(imread(base_folder + 'Negative_2.tif'))
positive_stack = np.array(imread(base_folder+ 'Positive_2.tif'))

# Ensure the stacks have the same dimensions
assert dapi_stack.shape[1:] == negative_stack.shape[1:] == positive_stack.shape[1:]

# Define alpha values for blending
alpha = 0.3

# Perform alpha blending for each frame
overlayed_stack = dapi_stack.copy()
for i in range(len(dapi_stack)):
    # DAPI channel (completely opaque)
    blended_frame = dapi_stack[i].copy()

    # Negative and Positive channels (blended with transparency)
    blended_frame = cv2.addWeighted(blended_frame, 1, negative_stack[i], alpha, 0)
    blended_frame = cv2.addWeighted(blended_frame, 1, positive_stack[i], alpha, 0)

    overlayed_stack[i] = blended_frame

# Save the result as a TIFF stack
imsave(base_folder + 'Overlayed.tiff', overlayed_stack)