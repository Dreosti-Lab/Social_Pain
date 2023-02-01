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
import SP_cfos as SZCFOS


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Summary List
summaryListFile = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Cfos_Summary/Cfos_fish_data_1.xlsx'

# Set Group
group = 1;
group_name = r'Susceptible'

# Set VPI thresholds
VPI_min = 0
VPI_max = 40

# Set ROI Path
roi_path = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/mask/PoA_MASK.tif'
roi_name = r'PoA'

# Set analysis folder and filename
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/GRADIENT/Cfos_Values'
analysis_path = analysis_folder + '/' + group_name + '_' + roi_name + '_cFos.npz'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, group_names ,behaviour_metrics = SZCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)

# Assign metrics/paths for each group
group_correct_id = (behaviour_metrics[:,0] == group)
group_metric_in_range = (behaviour_metrics[:,3] > VPI_min) * (behaviour_metrics[:,3] <= VPI_max)
group_indices = np.where(group_correct_id * group_metric_in_range)[0].astype(np.uint)
cfos_paths = cfos_paths[group_indices]
n = len(group_indices)

# Load ROI mask
roi_stack = SZCFOS.load_mask(roi_path, transpose=True)
num_roi_voxels = np.sum(np.sum(np.sum(roi_stack)))

# ------------------------------------------------------------------
# cFos Analysis
# ------------------------------------------------------------------

# Measure (normalized) cFOS in Mask ROI
cFos_values = np.zeros(n)
for i in range(n):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SZCFOS.load_nii(cfos_paths[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value = np.sum(np.sum(np.sum(roi_stack * cfos_data)))/num_roi_voxels
                             
    # Append to list
    cFos_values[i] = cFos_value
    
    print(str(i+1) + ' of ' + str(n) + ':\n' + cfos_paths[i] + '\n')

# ------------------------------------------------------------------
# Save cFos values
# ------------------------------------------------------------------
np.savez(analysis_path, cFos_values=cFos_values, group_name=group_name, roi_name=roi_name)
#print("Saved cFos Values: Mean - " + np.mean(cFos_values) + ' +/- STD ' + np.std(cFos_values))

# FIN