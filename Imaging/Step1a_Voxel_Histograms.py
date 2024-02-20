# -*- coding: utf-8 -*-
"""
This script loads a registered cfos stack and computes a voxel histogram for all voxells within the defined mask 
- Saves the corresponding histogram: .png .npz
- Calculates offset, bin centers, bottom and top decile, mode and median values

@author: Alizee Kastler
"""



#---------------------------------------------------------------------------

# Specify the  Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
import os
sys.path.append(lib_path)

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


#---------------------------------------------------------------------------


# Set cfos Stack Path
folder_path = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Peptides/512_2/CART/23_11_01/fish1'
stack_path = folder_path + '/DAPI_CART_02_reg_Warped.nii.gz'

# Set DAPI Mask Path
mask_path =  'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/mask/DAPI_512_2_MASK.tif'
mask_slice_range_start = 0
mask_slice_range_stop = 320

# Load mask
mask_data = SPCFOS.load_mask(mask_path, transpose=True)
mask_data[:,:,:mask_slice_range_start] = 0
mask_data[:,:,mask_slice_range_stop:] = 0
num_mask_voxels = np.sum(np.sum(np.sum(mask_data)))

# ------------------------------------------------------------------

    
# Load original (warped) cFos stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stack_path, normalized = False)

# Remove starurated (peak in large values)
#cfos_data[cfos_data > 32768] = 0

# Find voxels belonging to the Mask 
masked_values = cfos_data[mask_data == 1]
    
#Specifiy bin range and compute histogram of masked_values
histogram, bin_edges  = np.histogram(masked_values, bins = 1000, range=[0,20000]);        
bin_width = (bin_edges[1]-bin_edges[0])/2
bin_centers = bin_edges[:-1] + bin_width

#Find Offset
offset_bin =np.where(histogram>100)[0][0].astype(np.uint)
offset=bin_centers[offset_bin]
    
# Find median bin
half_count = np.sum(histogram) / 2
median_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - half_count))).astype(np.uint)
median = bin_centers[median_bin]

# Find lower quartile bin
bot_decile_count = np.sum(histogram) / 10
bot_decile_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - bot_decile_count))).astype(np.uint)
bot_decile = bin_centers[bot_decile_bin]

# Find lower quartile bin
top_decile_count = 9 * np.sum(histogram) / 10
top_decile_bin = np.round(np.argmin(np.abs(np.cumsum(histogram) - top_decile_count))).astype(np.uint)
top_decile = bin_centers[top_decile_bin]

# Find mode bin
mode_bin = np.argmax(histogram)
mode = bin_centers[mode_bin]


# Plot histogram with markers for median(black dot), mode(black cross), bottom (blue) and top decile (red) 
voxel_hist = plt.figure()

plt.plot(bin_centers, histogram)
plt.plot(median, histogram[median_bin], 'ko')
plt.plot(mode, histogram[mode_bin], 'k+')
plt.plot(bot_decile, histogram[bot_decile_bin], 'bo')
plt.plot(top_decile, histogram[top_decile_bin], 'ro')

#Add labels and legend
plt.xlabel('voxel intensity')
plt.ylabel ('frequency')
plt.legend()

# Save histogram as png file to check immediately and npz to store vallues
voxel_hist.savefig(folder_path + '/voxel_histogram.png', dpi=300, bbox_inches='tight')

histogram_file = folder_path + '/voxel_histogram.npz'      
np.savez(histogram_file, 
         histogram=histogram, 
         bin_centers=bin_centers,
         offset=offset,
         median=median,
         bot_decile = bot_decile,
         top_decile = top_decile,
         mode=mode)
 

                          
# FIN