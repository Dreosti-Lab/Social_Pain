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
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats

# Import local modules
import SP_video as SPV
import SP_utilities as SPU

# Read Folder List
folderListFile = base_path + r'\Python_ED\Social _Behaviour_Setup\PreProcessing'
folderListFile = folderListFile + r'\SocialFolderList_PreProcessing_2017_08_25_subset.txt'

control = False
groups, ages, folderNames, fishStatus = SZU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, C_folder = SZU.get_folder_names(folder)

    # ----------------------
            
    # Process Video (NS)
    SZV.pre_process_video_summary_images(NS_folder, False)
    
    # Check if this is a control experiment
    if control:
        # Process Video (NS_2) - Control
        SZV.pre_process_video_summary_images(C_folder, False)

    else:
        # Process Video (S)
        SZV.pre_process_video_summary_images(S_folder, True)
#    print (cv2.__version__) 
        # Process Video (D-dark)
       
    # Report Progress
 #print groups[idx]
    
# FIN
    