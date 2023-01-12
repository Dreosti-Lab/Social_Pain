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
import SP_cfos as SPCFOS


#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set Summary List
summaryListFile = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Cfos_Summary/Cfos_fish_data_2.xlsx'

# Set ROI Path
roi_path = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/mask/Vv_MASK.tif'
roi_name = r'Ventral_Telencephalon'

# Set analysis folder and filename
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/GRADIENT/Cfos_Values'
analysis_path = analysis_folder + '/' + roi_name + '/' + roi_name +'_cFos.npz'

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Read summary list
cfos_paths, group_names,behaviour_metrics = SPCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
behaviour_metrics = np.array(behaviour_metrics)
n = len(cfos_paths)

# Load ROI mask
roi_stack = SPCFOS.load_mask(roi_path, transpose=True)
num_roi_voxels = np.sum(np.sum(np.sum(roi_stack)))

# ------------------------------------------------------------------
# cFos Analysis
# ------------------------------------------------------------------

# Measure (normalized) cFOS in Mask ROI
cFos_values = np.zeros(n)
for i in range(n):
    
    # Load original (warped) cFos stack
    cfos_data, cfos_affine, cfos_header = SPCFOS.load_nii(cfos_paths[i], normalized=True)

    # Measure average signal level in mask ROI
    cFos_value = np.sum(np.sum(np.sum(roi_stack * cfos_data)))/num_roi_voxels
                             
    # Append to list
    cFos_values[i] = cFos_value
    
    print(str(i+1) + ' of ' + str(n) + ':\n' + cfos_paths[i] + '\n')

# ------------------------------------------------------------------
# Save cFos values
# ------------------------------------------------------------------
np.savez(analysis_path, cFos_values=cFos_values, behaviour_metrics=behaviour_metrics, roi_name=roi_name)

# FIN