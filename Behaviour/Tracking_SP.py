# -*- coding: utf-8 -*-
"""
Video Fish Tracking after defining ROIs 
"""                        
# Set Library Path - Social_Pain Repos
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'
base_path = r'S:/WIBR_Dreosti_Lab/Tom/Behaviour_Lesion_Social'

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
import SP_Video as SPV
import BONSAI_ARK
import SP_Utilities as SPU

# Read Folder List
FolderlistFile = base_path + r'/Experiment_1/Folderlist_Sham.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Process Video (NS)
    bonsaiFiles = NS_folder + r'/Bonsai_ROI_Analysis.bonsai'
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    # Run more improved tracking in SP_video module
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SPV.fish_tracking(NS_folder, NS_folder, ROIs)

    # Save Tracking (NS)
    for i in range(0,6):
        filename = NS_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)
    
    #---------------------
    # Process Video (S)
    bonsaiFiles = S_folder + r'/Bonsai_ROI_Analysis.bonsai'
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    #Run more improved tracking in SP_video module
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SPV.fish_tracking(S_folder, S_folder, ROIs)

    # Save Tracking (S)
    for i in range(0,6):
        filename = S_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)
    
    #Close Plots
    plt.close('all')
    
# FIN
    