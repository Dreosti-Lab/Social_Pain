# -*- coding: utf-8 -*-
"""
This script converts the diff stack generated in Step2a into a display-scaled version.
- Black = min, 0  = 128/gray, White = max
- use the output MIN and MAX for the scale bar

@author: Dreosti Lab
"""

# -----------------------------------------------------------------------------
# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
import os
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


# Set Input stack
stackFolder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/NOXIOUS/512_2'
stackFile = stackFolder + r'/Diff_Stack.nii.gz'

# Set Mask Path
mask_path =  'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/mask/DAPI_512_2_MASK.tif'
mask_slice_range_start = 0
mask_slice_range_stop = 316


# Load mask
mask_data = SPCFOS.load_mask(mask_path, transpose=True)
mask_data[:,:,:mask_slice_range_start] = 0
mask_data[:,:,mask_slice_range_stop:] = 0
num_mask_voxels = np.sum(np.sum(np.sum(mask_data)))

# Load stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stackFile, normalized=True)
masked_values = cfos_data[mask_data == 1]
n_stack_rows = np.size(cfos_data, 0)
n_stack_cols = np.size(cfos_data, 1)
n_stack_slices = np.size(cfos_data, 2)
display_data = np.zeros((n_stack_rows, n_stack_cols, n_stack_slices), dtype = np.float32)    

# Compute stats
min_val = np.min(cfos_data[:])
max_val = np.max(cfos_data[:])

# Measure histogram values
histogram, bin_edges  = np.histogram(masked_values, bins = 10000, range=[-10, 10]);
bin_width = (bin_edges[1]-bin_edges[0])/2
bin_centers = bin_edges[:-1] + bin_width

# Find lower 0.25% bin
bot_count = np.sum(histogram) / 250
bot_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - bot_count))).astype(np.uint)
bot_val = bin_centers[bot_bin]

# Find upper 0.25% bin
top_count = 249 * np.sum(histogram) / 250
top_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - top_count))).astype(np.uint)
top_val = bin_centers[top_bin]

# Adjust stack
neg_vals = cfos_data < 0.0
pos_vals = cfos_data > 0.0
display_data[neg_vals] = cfos_data[neg_vals] / np.abs(bot_val)
display_data[pos_vals] = cfos_data[pos_vals] / np.abs(top_val)


# Save Display stack
image_affine = np.eye(4)
displayFile = stackFolder + r'\Diff_Stack_DISPLAY_MIN' + format(bot_val, '.3f') + '_MAX' + format(top_val, '.3f') + '.nii.gz'
SPCFOS.save_nii(displayFile, display_data, image_affine)


# FIN
