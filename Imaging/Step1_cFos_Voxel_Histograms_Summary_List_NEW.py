# -*- coding: utf-8 -*-

# Set Library Paths
lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

import sys
import os

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Stack Path
stack_path = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1_Server/Registration/Heat_Gradient/Average/Reg_CFOS_GRADIENT_22_02_04_fish1.nii'
# Set Mask Path
mask_path =  'S:/WIBR_Dreosti_Lab/Alizee/LSZ1_Server/Registration/mask/DAPI_MASK.nii'
mask_slice_range_start = 130
mask_slice_range_stop = 230

# Use the normalized stacks?
normalized = False

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Load mask
mask_data, mask_affine, mask_header = SPCFOS.load_nii(mask_path, normalized=False)
mask_data = mask_data[:,:,:,0]
mask_data[:,:,:mask_slice_range_start] = 0
mask_data[:,:,mask_slice_range_stop:] = 0
num_mask_voxels = np.sum(np.sum(np.sum(mask_data)))


# Measure cFOS in Mask (normalize to "background")
plt.figure()
    
# Load original (warped) cFos stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stack_path, normalized = False)

# Remove starurated (peak in large values)
cfos_data[cfos_data > 65400] = 0

# Apply mask
masked_values = cfos_data[mask_data == 1]

# Histogram
histogram, bin_edges  = np.histogram(masked_values, bins = 1024, range=[-31072, 100000]);        
bin_width = (bin_edges[1]-bin_edges[0])/2
bin_centers = bin_edges[:-1] + bin_width

# Reverse search the histogram for the first local maximum
smoothing = 5
reverse_histogram = np.flip(histogram)
running_max = 0
best_index = 0
for index in range(len(reverse_histogram[smoothing:-smoothing])):
    window_start = index - smoothing
    window_stop = index + smoothing
    average = np.mean(reverse_histogram[window_start:window_stop])
    print(average)
    if average > running_max:
        running_max = average
        best_index = index

    # Check for dip
    if average < ((running_max * 0.95) and(average > 100000)):
        break

# Find mode (of signal peak)
mode_index = 1024 - best_index
mode = bin_centers[mode_index]

# Plot histogram
plt.plot(bin_centers, histogram)
plt.plot(bin_centers[mode_index], histogram[mode_index], 'ro')
plt.xlim(-1000, 70000)
plt.ylim(0, 700000)
plt.show()

# Save histogram
if(normalized):
    histogram_file = os.path.dirname(stack_path) + '/voxel_histogram_normalized.npz'
else:
    histogram_file = os.path.dirname(stack_path) + '/voxel_histogram.npz'        
np.savez(histogram_file, 
            histogram=histogram, 
            bin_centers=bin_centers, 
            mode=mode)
                              
# FIN
