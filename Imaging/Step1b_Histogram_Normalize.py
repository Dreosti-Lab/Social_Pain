# -*- coding: utf-8 -*-
"""
This script loads and normalizes a cFos folder list: .nii images and behaviour
@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:/Repos/Social_Pain/libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import os
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


#---------------------------------------------------------------------------

# Set Stack Path
folder_path ='S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Habituation/22_10_27/fish16'
stack_path = folder_path + '/DAPI_CFOS_02_reg_Warped.nii.gz'
# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

# Subtract histogram offset and scale (divide) by mode

# Read histogram npz
histogram_file = folder_path + r'/voxel_histogram.npz'
npzfile = np.load(histogram_file)
histogram = npzfile['histogram']
bin_centers = npzfile['bin_centers']
offset = npzfile['offset']
mode = npzfile['mode']

# Load original (warped) cFos stack
cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(stack_path, normalized=False)
# # Remove starurated (peak in large values)
# cfos_data[cfos_data > 70000] = 0


backsub=cfos_data - offset

# Normalize to histogram mode
normalized = backsub / (mode-offset)
# Save normlaized NII stack...
save_path = folder_path+ '/Reg_Warped_normalized.nii.gz'      
SPCFOS.save_nii(save_path, normalized, cfos_affine, cfos_header)

                          
# FIN