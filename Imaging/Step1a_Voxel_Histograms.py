# -*- coding: utf-8 -*-
"""
This script loads a rgeistered cfos stack and computes a voxel histogram for all voxells within the defined mask 
- Saves the corresponding histogram: .png .npz
- Calculates offset, bin_centers, bottom and top decile, mode and median values

@author: Alizee Kastler
"""



#---------------------------------------------------------------------------

# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
import os
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


#---------------------------------------------------------------------------



# Set Stack Path
folder_path = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Habituation/22_10_27/fish18'
stack_path = folder_path + '/DAPI_CFOS_02_reg_Warped.nii.gz'

# Set Mask Path
mask_path =  'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/mask/DAPI_MASK.tif'
mask_slice_range_start = 100
mask_slice_range_stop = 300

# Load mask
mask_data = SPCFOS.load_mask(mask_path, transpose=True)
mask_data[:,:,:mask_slice_range_start] = 0
mask_data[:,:,mask_slice_range_stop:] = 0
num_mask_voxels = np.sum(np.sum(np.sum(mask_data)))

# ------------------------------------------------------------------

    
# Load original (warped) cFos stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stack_path, normalized = False)

# Remove starurated (peak in large values)
cfos_data[cfos_data > 32768] = 0

# Find voxels belonging to the Mask 
masked_values = cfos_data[mask_data == 1]
    
#Histogram of masked_values
histogram, bin_edges  = np.histogram(masked_values, bins = 100, range=[100,5000]);        
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



# Plot histogram
voxel_hist = plt.figure()

plt.plot(bin_centers, histogram)
plt.plot(median, histogram[median_bin], 'ko')
plt.plot(mode, histogram[mode_bin], 'k+')
plt.plot(bot_decile, histogram[bot_decile_bin], 'bo')
plt.plot(top_decile, histogram[top_decile_bin], 'ro')

# Save histogram
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