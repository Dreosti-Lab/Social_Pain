# -*- coding: utf-8 -*-
"""
Video Fish Tracking after defining ROIs 
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social_Pain Repos
lib_path = r'C:/Repos/Dreosti-Lab/Social_Pain/libs'

# Set Library Path
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
base_path = r'V:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats

# Import local modules
import SP_video as SPV
import BONSAI_ARK
import SP_utilities as SPU

# Reload important libraries
import importlib
importlib.reload(SPV)

# # Read Folder List
# folderListFile = base_path + r'\Experiment_2\Folderlist\Exp_2.txt'

# groups, ages, folderNames = SPU.read_folder_list(folderListFile)

# # Bulk analysis of all folders
# for idx,folder in enumerate(folderNames):
    
#     # Get Folder Names
#     NS_folder, S_folder = SPU.get_folder_names(folder)

#     input_folder = NS_folder, S_folder
#     output_folder = input_folder
#     bonsai_file = input_folder + r'/Bonsai_ROI_Analysis.bonsai'
#     ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsai_file)

# # Run more improved tracking in SP_video module
# SPV.improved_fish_tracking(input_folder, output_folder, ROIs)



# Set path video + ROI + output folder
input_folder = base_path + r'/Experiment_3/2020_12_27/Behaviours/Fish5_26dpf/Social_1'
output_folder = input_folder
bonsai_file = input_folder + r'/Bonsai_ROI_Analysis.bonsai'
ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsai_file)
    
# Run more improved tracking in SP_video module
SPV.improved_fish_tracking(input_folder, output_folder, ROIs)


# FIN
    