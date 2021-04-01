# -*- coding: utf-8 -*-
"""
Video Fish Tracking after defining ROIs e
"""                        

# -----------------------------------------------------------------------------
# Set "Library Path" - Social_Pain Repos
lib_path = r'C:/Repos/Social_Pain/libs'

# Set Library Path
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'


# Import useful libraries
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import scipy.misc as misc
from scipy import stats



# Import local modules
import SP_video as SPV
import BONSAI_ARK
import SP_utilities as SPU

# Reload important libraries
import importlib
importlib.reload(SPV)

# Read Folder List
FolderlistFile = base_path + r'\Experiment_14\Folderlist\Exp_14.txt'

groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Process Video (NS)
    input_folder = NS_folder
    output_folder = input_folder
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    # Run more improved tracking in SP_video module
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SPV.improved_fish_tracking(input_folder, output_folder, ROIs)

    # Save Tracking (NS)
    for i in range(0,6):
        # Save NS
        filename = NS_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)
    
    #---------------------
    # Process Video (S)
    input_folder = S_folder
    output_folder = input_folder
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    #Run more improved tracking in SP_video module
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SPV.improved_fish_tracking(input_folder, output_folder, ROIs)

    # Save Tracking (S)
    for i in range(0,6):
        filename = S_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)
    
     # Close Plots
    plt.close('all')
    


# # Set path video + ROI + output folder
# input_folder = base_path + r'/Experiment_2/Behaviours/2020_12_11/Fish2_23dpf/Non_Social_1'
# output_folder = input_folder
# bonsai_file = input_folder + r'/Bonsai_ROI_Analysis.bonsai'
# ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsai_file)
    

# # # Run more improved tracking in SP_video module
# fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SPV.improved_fish_tracking(input_folder, output_folder, ROIs)

# #Save Tracking as NPZ file
# for i in range(0,6):
#     filename = output_folder + r'/tracking'+ str(i+1) + '.npz'
#     fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
#     np.savez(filename, tracking=fish.T)




# FIN
    