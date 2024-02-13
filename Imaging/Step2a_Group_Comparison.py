# -*- coding: utf-8 -*-
"""
This script loads and processes two different groups of cfos stacks from an excel folder 
- Computes the mean and std of each group
- Computes the t-score differences between each group for each voxel in the stack 
- Saves mean, std, Diff and T-Stacks

@author: Dreosti Lab
"""


# -----------------------------------------------------------------------------
# Specify Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
import os
sys.path.append(lib_path)

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


# -----------------------------------------------------------------------------

# Set Summary List
summaryListFile = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Cfos_Summary/Cfos_Summary_512_2.xlsx'

# Set analysis path
analysis_folder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/AITC/512_2/03_11_23'

# Read summary list
cfos_paths,group_names = SPCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
group_names = np.array(group_names)


# Assign metrics/paths for each group
group_A = (group_names == 'AITC')
group_B = (group_names == 'DMSO')

cfos_paths_A = cfos_paths[group_A]
cfos_paths_B = cfos_paths[group_B]
n_A = len(cfos_paths_A)
n_B = len(cfos_paths_B)



# Spatial smoothing factor
smooth_factor = 4
	
# Measure mean and std stacks for group A and B
mean_stack_A, std_stack_A = SPCFOS.summary_stacks(cfos_paths_A, smooth_factor, normalized=True)
mean_stack_B, std_stack_B = SPCFOS.summary_stacks(cfos_paths_B, smooth_factor, normalized=True)


# Compute t-score stack for (B - A)
# - Subtract meanB - meanA = Diff_Mean
# - Estimate combined STD for A and B: sqrt(stdA^2/nA + stdB^2/nB)

diff_mean = mean_stack_A - mean_stack_B 
both_std = np.sqrt( ((std_stack_A*std_stack_A)/n_A) + ((std_stack_B*std_stack_B)/n_B) )
t_stack = diff_mean/both_std


# ------------------------------------------------------------------
# Display and Save Results

# Make plots
plt.figure()
plt.subplot(1,2,1)
plt.imshow(diff_mean[:,:,50])
plt.subplot(1,2,2)
plt.imshow(t_stack[:,:,50])


# Save NII stack of results
image_affine = np.eye(4)
save_path = analysis_folder + r'/T_Stack.nii.gz'
SPCFOS.save_nii(save_path, t_stack, image_affine)

save_path = analysis_folder + r'/Diff_Stack.nii.gz'
SPCFOS.save_nii(save_path, diff_mean, image_affine)

save_path = analysis_folder + r'/Mean_Stack_A.nii.gz'
SPCFOS.save_nii(save_path, mean_stack_A, image_affine)

save_path = analysis_folder + r'/Mean_Stack_B.nii.gz'
SPCFOS.save_nii(save_path, mean_stack_B, image_affine)

save_path = analysis_folder + r'/STD_Stack_A.nii.gz'
SPCFOS.save_nii(save_path, std_stack_A, image_affine)

save_path = analysis_folder + r'/STD_Stack_B.nii.gz'
SPCFOS.save_nii(save_path, std_stack_B, image_affine)

save_path = analysis_folder + r'/STD_Stack.nii.gz'
SPCFOS.save_nii(save_path, both_std, image_affine)

# FIN
