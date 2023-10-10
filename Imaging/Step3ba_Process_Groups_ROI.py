# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:54 2023
This code reads a summary list file and loads a specific ROI 
- calculate the cfos_value of a specific ROI according to a defined behaviour group
- save values as npz 
@author: Alizee Kastler
"""


# -----------------------------------------------------------------------------

# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS

# -----------------------------------------------------------------------------


# Set Summary List
summaryListFile = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Cfos_Summary/Cfos_Summary_512_2.xlsx'

# Set ROI Path
roi_path = 'F:/ATLAS/MASK_CFOS/PTN.tif'
roi_name = r'PTN'

#Set analysis folder and filename
analysis_folder ='S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'+ roi_name

# Create output folder (if doesn't exist)
if not os.path.exists(analysis_folder):
   os.makedirs(analysis_folder)


analysis_path_A = analysis_folder + '/Baseline_cfos.npz'
analysis_path_B = analysis_folder + '/Heat_cfos.npz'
analysis_path_C = analysis_folder + '/AITC_cfos.npz'
analysis_path_D = analysis_folder + '/Social_cfos.npz'


# -----------------------------------------------------------------------------

# Read summary list
cfos_paths, group_names = SPCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
group_names = np.array(group_names)


group_A = (group_names == 'Baseline')
cfos_values_A = SPCFOS.cfos_value_ROI(group_A, cfos_paths, roi_path)
np.savez(analysis_path_A, cfos_values=cfos_values_A, group_name=group_A, roi_name=roi_name)

group_B = (group_names == 'Heat')
cfos_values_B = SPCFOS.cfos_value_ROI(group_B, cfos_paths, roi_path)
np.savez(analysis_path_B, cfos_values=cfos_values_B, group_name=group_B, roi_name=roi_name)

group_C = (group_names == 'AITC')
cfos_values_C = SPCFOS.cfos_value_ROI(group_C, cfos_paths, roi_path)
np.savez(analysis_path_C, cfos_values=cfos_values_C, group_name=group_C, roi_name=roi_name)

group_D = (group_names == 'Social')
cfos_values_D = SPCFOS.cfos_value_ROI(group_D, cfos_paths, roi_path)
np.savez(analysis_path_D, cfos_values=cfos_values_D, group_name=group_D, roi_name=roi_name)



#FIN