# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:54 2023

@author: Alizee Kastler
"""

# -*- coding: utf-8 -*-
"""
This script loads and processes a cFos folder list: .nii images and behaviour
@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:/Repos/Social_Pain/libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Summary List
summaryListFile = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Cfos_Summary/Cfos_Summary_2.xlsx'

# Set ROI Path
roi_path = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/mask/cH.tif'
roi_name = r'cH'

#Set analysis folder and filename
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'+ roi_name


analysis_path_A = analysis_folder + '/group_A_cFos.npz'
analysis_path_B = analysis_folder + '/group_B_cFos.npz'
analysis_path_C = analysis_folder + '/group_C_cFos.npz'


# Read summary list
cfos_paths, group_names = SPCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
group_names = np.array(group_names)

# Assign metrics/paths for each group
group_A = (group_names == 'Baseline')
group_B = (group_names == 'Social')
group_C = (group_names == 'Noxious')


cfos_paths_A = cfos_paths[group_A]
cfos_paths_B = cfos_paths[group_B]
cfos_paths_C = cfos_paths[group_C]

n_A = len(cfos_paths_A)
n_B = len(cfos_paths_B)
n_C = len(cfos_paths_C)

# Load ROI mask
roi_stack = SPCFOS.load_mask(roi_path, transpose=True)
num_roi_voxels = np.sum(np.sum(np.sum(roi_stack)))

# ------------------------------------------------------------------
# cFos Analysis
# ------------------------------------------------------------------

# Measure (normalized) cFOS in Mask ROI
cFos_values_A = np.zeros(n_A)
for i in range(n_A):
    
    # Load original (warped) cFos stack
    cfos_data_A, cfos_affine_A, cfos_header_A = SPCFOS.load_nii(cfos_paths_A[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value_A = np.sum(np.sum(np.sum(roi_stack * cfos_data_A)))/num_roi_voxels
                             
    # Append to list
    cFos_values_A[i] = cFos_value_A
    
    print(str(i+1) + ' of ' + str(n_A) + ':\n' + cfos_paths_A[i] + '\n')

np.savez(analysis_path_A, cFos_values=cFos_values_A, group_name=group_A, roi_name=roi_name)

# Measure (normalized) cFOS in Mask ROI
cFos_values_B = np.zeros(n_B)
for i in range(n_B):
    
    # Load original (warped) cFos stack
    cfos_data_B, cfos_affine_B, cfos_header_B = SPCFOS.load_nii(cfos_paths_B[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value_B = np.sum(np.sum(np.sum(roi_stack * cfos_data_B)))/num_roi_voxels
                             
    # Append to list
    cFos_values_B[i] = cFos_value_B
    
    print(str(i+1) + ' of ' + str(n_B) + ':\n' + cfos_paths_B[i] + '\n')

np.savez(analysis_path_B, cFos_values=cFos_values_B, group_name=group_B, roi_name=roi_name)


# Measure (normalized) cFOS in Mask ROI
cFos_values_C = np.zeros(n_C)
for i in range(n_C):
    
    # Load original (warped) cFos stack
    cfos_data_C, cfos_affine_C, cfos_header_C = SPCFOS.load_nii(cfos_paths_C[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value_C = np.sum(np.sum(np.sum(roi_stack * cfos_data_C)))/num_roi_voxels
                             
    # Append to list
    cFos_values_C[i] = cFos_value_C
    
    print(str(i+1) + ' of ' + str(n_C) + ':\n' + cfos_paths_C[i] + '\n')

np.savez(analysis_path_C, cFos_values=cFos_values_C, group_name=group_C, roi_name=roi_name)
# FIN