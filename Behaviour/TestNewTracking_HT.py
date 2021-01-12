# -*- coding: utf-8 -*-
"""
Add comments here
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Pain\libs'

# Set Library Paths
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
base_path = r'V:/WIBR_Dreosti_Lab/Hande/Behaviours/Dual_heat_plate'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import glob, os

# Import local modules
import SP_video as SPV
import SP_utilities as SPU
import BONSAI_ARK

# Reload important libraries
import importlib
importlib.reload(SPV)

#---------------------------------------------------------------------------------
## Set experiment path for bulk analysis 
Exps_folder = base_path + r'/Experiment_1'
#---------------------------------------------------------------------------------

# Test Fish Tracking 
avi_paths_list = glob.glob(Exps_folder +'/*/*/*/*/')
#for the entire path with avi      avi_paths_list = glob.glob(Exps_folder +'/*/*/*.avi')

for avis in avi_paths_list:
    input_folder = avis
    output_folder = input_folder
    bonsai_file = input_folder + r'\Bonsai_ROI_Analysis.bonsai'
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsai_file)

    # Run more improved tracking
    SPV.improved_fish_tracking(input_folder, output_folder, ROIs)


# ## Conspecific Fish Tracking 
# Conspecifics_avi_paths_list = glob.glob(Exps_folder +'/*/*/*/Social_1')
# for avis in Conspecifics_avi_paths_list:
#     input_folder = avis
#     output_folder = input_folder + r'/Social_Fish'
#     cons_bonsai_file = output_folder + r'\Bonsai_ROI_Analysis.bonsai'
#     cons_ROIs = BONSAI_ARK.read_bonsai_crop_rois(cons_bonsai_file)  
    
#         # Run more improved tracking
#     SPV.improved_fish_tracking(input_folder, output_folder, cons_ROIs)
    
    
# FIN
    