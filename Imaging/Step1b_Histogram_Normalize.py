# -*- coding: utf-8 -*-
"""

This Srcript reads the Histogram generated step 1a (.npz file)
- Substracts the histogram offset from all the voxels
- Normalizes each voxel to background (divide by the mode)
- Outputs a Normalized Stack as a .nii file

"""


# -----------------------------------------------------------------------------

# Specify Library Path
lib_path = r'C:/Repos/Social_Pain/libs'
import os
import sys
sys.path.append(lib_path)
  

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS

# -----------------------------------------------------------------------------


# Set cfos Stack Path
folder_path =  'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Peptides/512_2/CART/23_11_01/fish1'
stack_path = folder_path + '/DAPI_CART_02_reg_Warped.nii.gz'


# Load histogram data from the .npx file generated in step1a
histogram_file = folder_path + r'/voxel_histogram.npz'
npzfile = np.load(histogram_file)
histogram = npzfile['histogram']
bin_centers = npzfile['bin_centers']
offset = npzfile['offset']
mode = npzfile['mode']

# Load original (warped) cfos stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stack_path, normalized=False)

# Normalize: subtract Histogram offset and scale (divide) by mode
backsub=cfos_data - offset
normalized = backsub / (mode-offset)


# Specifiy path and save normlaized stack as .nii file
save_path = folder_path+ '/Reg_Warped_normalized.nii.gz'      
SPCFOS.save_nii(save_path, normalized, cfos_affine, cfos_header)

                          
# FIN