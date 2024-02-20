#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 17:40:05 2023

@author: alizeekastler
"""

import os
import cv2
import numpy as np
from PIL import Image

# Specify diff name, basediff, and scale values
diff_name = 'Diff_Stack'


# Build paths
base_folder = '/Volumes/T7 Touch/CFOS_Areas/CFOS_Areas/'
stack_path = base_folder + '/Noxious/512_2_2/'+ diff_name + '.tif'
mask_path = base_folder + '/DAPI_512_2_MASK.tif'
pos_output_folder = base_folder + '/Noxious/512_2_2/' + diff_name + '_pos'
neg_output_folder = base_folder + '/Noxious/512_2_2/' + diff_name + '_neg'

output_folder = base_folder + '/Noxious/512_2_2/' + diff_name + '_both'
# Create output folders (if they do not exist)
if not os.path.exists(pos_output_folder):
   os.makedirs(pos_output_folder)
if not os.path.exists(neg_output_folder):
   os.makedirs(neg_output_folder)

# Load diff stack
diff_data = Image.open(stack_path)
height, width = np.shape(diff_data)
num_frames = diff_data.n_frames

# Load mask stack
mask_data = Image.open(mask_path)

# Create a single empty container for the result
result_container = np.zeros((num_frames, width, height, 4), dtype=np.uint8)

# Set the threshold values for blue (positive) and red (negative)
blue_threshold = 0.65
red_threshold = -0.5

# Create empty containers for positive and negative results
pos_result_container = []
neg_result_container = []

for i in range(num_frames):
    diff_data.seek(i)
    mask_data.seek(i)
    data = np.asarray(diff_data) * np.asarray(mask_data)

    # Assign colors based on pixel values and set transparency
    blue_pixels = data > blue_threshold
    red_pixels = data < red_threshold
    gray_pixels = ~blue_pixels & ~red_pixels

    result_container_frame = np.zeros((width, height, 4), dtype=np.uint8)

    result_container_frame[blue_pixels] = [0, 0, 255, 255]  # Blue and opaque
    result_container_frame[red_pixels] = [255, 0, 0, 255]  # Red and opaque

    # Calculate grayscale values
    gray_values = ((data + 2) * 127.5).astype(np.uint8)

    # Apply grayscale values to all non-blue and non-red pixels
    result_container_frame[gray_pixels, :3] = gray_values[gray_pixels][:, None]  # Reshape for broadcasting
    result_container_frame[gray_pixels, 3] = 128  # Semi-transparent

    # Apply Gaussian blur to the red and blue areas
    result_container_frame[:, :, :3] = cv2.GaussianBlur(result_container_frame[:, :, :3], (5, 5), 0)

    # Update the result_container for the current frame
    result_container[i] = result_container_frame
    
    
    # Check if pixel values are positive or negative
    positive_frame = np.zeros((width, height, 4), dtype=np.uint8)
    negative_frame = np.zeros((width, height, 4), dtype=np.uint8)

    positive_frame[blue_pixels] = [0, 0, 255, 255]  # Blue and opaque (positive)
    negative_frame[red_pixels] = [255, 0, 0, 255]  # Red and opaque (negative)

    # Calculate grayscale values
    gray_values = ((data + 2) * 127.5).astype(np.uint8)

    # Apply grayscale values to all non-blue and non-red pixels
    positive_frame[gray_pixels, :3] = gray_values[gray_pixels][:, None]  # Reshape for broadcasting
    negative_frame[gray_pixels, :3] = gray_values[gray_pixels][:, None]  # Reshape for broadcasting

    # Apply Gaussian blur to the red and blue areas
    positive_frame[:, :, :3] = cv2.GaussianBlur(positive_frame[:, :, :3], (5, 5), 0)
    negative_frame[:, :, :3] = cv2.GaussianBlur(negative_frame[:, :, :3], (5, 5), 0)

    # Append positive and negative frames to their respective containers
    pos_result_container.append(positive_frame)
    neg_result_container.append(negative_frame)


# Create and save the result images outside the loop
for i in range(num_frames):
    result_output = Image.fromarray(result_container[i])
    result_path = output_folder + '/' + diff_name + str(i).zfill(4) + '.png'
    result_output.save(result_path)
    
    pos_result_output = Image.fromarray(pos_result_container[i])
    neg_result_output = Image.fromarray(neg_result_container[i])

    pos_result_path = os.path.join(pos_output_folder, f"{diff_name}{i:04}_pos.tiff")
    neg_result_path = os.path.join(neg_output_folder, f"{diff_name}{i:04}_neg.tiff")

    pos_result_output.save(pos_result_path)
    neg_result_output.save(neg_result_path)
