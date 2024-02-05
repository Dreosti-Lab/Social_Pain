# -*- coding: utf-8 -*-
"""

This Srcript reads the Histogram generated step 1a: .npz
- Substracts the histogram offset from all the voxels and normalizes each voxel to background (divide by the mode)
- Outputs a Normalized Stack : .nii

"""


# -----------------------------------------------------------------------------


# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'
import os
import sys
sys.path.append(lib_path)
  

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


# -----------------------------------------------------------------------------


# Set Stack Path
folder_path =  'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Peptides/512_2/CART/23_11_01/fish1'
stack_path = folder_path + '/DAPI_CART_02_reg_Warped.nii.gz'


# Read histogram npz (Step1a)
histogram_file = folder_path + r'/voxel_histogram.npz'
npzfile = np.load(histogram_file)
histogram = npzfile['histogram']
bin_centers = npzfile['bin_centers']
offset = npzfile['offset']
mode = npzfile['mode']

# Load original (Registered) cFos stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stack_path, normalized=False)
# # Remove starurated (peak in large values)
# cfos_data[cfos_data > 70000] = 0

# Subtract Histogram offset and scale (divide) by mode
backsub=cfos_data - offset
normalized = backsub / (mode-offset)


# Save normlaized NII stack...
save_path = folder_path+ '/Reg_Warped_normalized.nii.gz'      
SPCFOS.save_nii(save_path, normalized, cfos_affine, cfos_header)

                          
# FIN