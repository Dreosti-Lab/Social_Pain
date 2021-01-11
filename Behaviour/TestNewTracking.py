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
base_path = r'S:/WIBR_Dreosti_lab/Alizee/Behaviour_Heat_Gradient'

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

# Reload important libraries
import importlib
importlib.reload(SPV)

# Set path video + ROI + output folder
input_folder = base_path + r'/Experiment_2/2020_12_11/behaviours/Fish2_23dpf/Non_Social_1'
output_folder = input_folder
bonsai_file = input_folder + r'/Bonsai_ROI_Analysis'
ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsai_file)
    
# Run more improved tracking in SP_video module
SPV.improved_fish_tracking(input_folder, output_folder, ROIs)

# FIN
    