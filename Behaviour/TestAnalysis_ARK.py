# -*- coding: utf-8 -*-
"""
Test Analysis
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social_Pain Repos
#lib_path = r'C:/Repos/Dreosti-Lab/Social_Pain/libs'
lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Pain/libs'

# Set Library Path
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
#base_path = r'V:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'
base_path = r'/media/kampff/Elements/Adam_Alizee'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import seaborn as sns

# Import local modules
import SP_utilities as SPU

# Read folder list file
FolderlistFile = base_path + '/FolderList.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# XMs
XMs = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis_folder = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
        fx_NS = tracking[:,0] 
        fy_NS = tracking[:,1]
        bx_NS = tracking[:,2]
        by_NS = tracking[:,3]
        ex_NS = tracking[:,4]
        ey_NS = tracking[:,5]
        area_NS = tracking[:,6]
        ort_NS = tracking[:,7]
        motion_NS = tracking[:,8]
   
        # Extract tracking data (S)
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking'] 
        fx_S = tracking[:,0] 
        fy_S = tracking[:,1]
        bx_S = tracking[:,2]
        by_S = tracking[:,3]
        ex_S = tracking[:,4]
        ey_S = tracking[:,5]
        area_S = tracking[:,6]
        ort_S = tracking[:,7]
        motion_S = tracking[:,8] 

        # Filter out bad data
        min_x = 250
        max_x = 1000
      
        if fish_number == 1:
            min_y = 120
            max_y = 165
        if fish_number == 2:
            min_y = 260
            max_y = 305
        if fish_number == 3:
            min_y = 400
            max_y = 445
        if fish_number == 4:
            min_y = 540
            max_y = 585
        if fish_number == 5:
            min_y = 680
            max_y = 725
        if fish_number == 6:
            min_y = 820
            max_y = 865    
 
        # Find good tracking (NS)
        num_total_frames_NS = len(fx_NS)
        good_frame_NS = (fx_NS > min_x) * (fx_NS < max_x) * (fy_NS > min_y) * (fy_NS < max_y)
        num_good_frames_NS = np.sum(good_frame_NS)
        lost_frame_NS = num_total_frames_NS-num_good_frames_NS
        
        # Find good tracking (S)         
        num_total_frames_S = len(fx_S)
        good_frame_S = (fx_S > min_x) * (fx_S < max_x) * (fy_S > min_y) * (fy_S < max_y)
        num_good_frames_S = np.sum(good_frame_S)
        lost_frame_S = num_total_frames_S-num_good_frames_S         

        # All lost frames
        lost_frame = lost_frame_S + lost_frame_NS
       
        # Store XMs
        XMs.append([np.mean(fx_NS[good_frame_NS]), np.mean(fx_S[good_frame_S])])

    # Report
    print("Next File: {0}".format(idx))

# Crude calibration: 0 = 28 deg, 900 = 36 deg (900/8) pixels per degree
XM_values = np.array(XMs)/112.5
TTSs = XM_values[:,1] - XM_values[:,0]

# Stats
s, pvalue_rel = stats.ttest_rel(XM_values[:,1], XM_values[:,0])
s, pvalue_1samp = stats.ttest_1samp(TTSs, 0)

# Make plot
mean_TTS = np.mean(XM_values[:,1] - XM_values[:,0])
sem_TTS = np.std(XM_values[:,1] - XM_values[:,0])/np.sqrt(len(TTSs)-1)
plt.vlines(0, 0, 8, 'k')
plt.hist(TTSs, bins=25)
plt.xlabel('Tolerated Temperature Shift (degrees C)')
plt.ylabel('Fish Count')
plt.title('Tolerated Temerpature Shift (TSS) During Social Cue Viewing\nMean TTS +/- SEM: {0:0.2f} +/- {1:0.2f}\n(p-value: {2:0.4f})'.format(mean_TTS, sem_TTS, pvalue_rel))
plt.ylim([0, 7.5])
plt.text(4.5, 7, 'n={0}'.format(len(TTSs)))
plt.show()


#FIN