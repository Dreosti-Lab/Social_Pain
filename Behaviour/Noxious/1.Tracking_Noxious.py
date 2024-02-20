# -*- coding: utf-8 -*-
"""
Fish tracking offline from a video and after defining ROIs in Bonsai
Extracts x and y coordinates of eye, body and heading orientation 

"""                        
# Set Library Path 
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'S:\WIBR_Dreosti_Lab\Tom\Videos_Alizee'
base_path =  r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient/NewChamber'

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
import SP_video_TRARK as SPV
import BONSAI_ARK
import SP_utilities as SPU

# Read Folder List
FolderlistFile = base_path + '/Heat_36/23_07_19/Folderlist.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Divisor and closing kernel parameters (thresholding for mask and background)
divisor=7
kernelWidth=3

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder,S_folder = SPU.get_folder_name(folder)

    # ---------------------
    # Process Video (Non_Social)
    bonsaiFiles = NS_folder + r'/Bonsai_ROI_Analysis.bonsai'
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    # Run tracking in SP_video 
    fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS = SPV.fish_tracking(NS_folder,S_folder, ROIs, divisor=divisor, kSize=kernelWidth)

    # Save Tracking (Non_Social)
    for i in range(0,6):
        filename = NS_folder + r'/tracking'+ str(i+1) + '.npz'
        fish = np.vstack((fxS[:,i], fyS[:,i], bxS[:,i], byS[:,i], exS[:,i], eyS[:,i], areaS[:,i], ortS[:,i], motS[:,i]))
        np.savez(filename, tracking=fish.T)         
    
    #Close Plots
    plt.close('all')
    
    
# FIN
    