# -*- coding: utf-8 -*-
"""
Test Analysis
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social_Pain Repos
lib_path = r'C:/Repos/Social_Pain/libs'
#lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Pain/libs'

# Set Library Path
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'
#base_path = r'/media/kampff/Elements/Adam_Alizee'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import seaborn as sns
import pandas as pd
import pylab as pl


# Import local modules
import SP_utilities as SPU

# Read folder list file
FolderlistFile = base_path + '/Folderlist_New.txt' 
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# XMs
XMs = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

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
#=============================================================================
      
        if fish_number == 1:
            min_y = 150
            max_y = 250
        if fish_number == 2:
            min_y = 270
            max_y = 370
        if fish_number == 3:
            min_y = 390
            max_y = 490
        if fish_number == 4:
            min_y = 510
            max_y = 610
        if fish_number == 5:
            min_y = 630
            max_y = 730
        if fish_number == 6:
            min_y = 750
            max_y = 850    
# =============================================================================
 
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
        
        # plt. figure ()
        # plt.plot(fx_NS[good_frame_NS], fy_NS[good_frame_NS], 'b.', alpha = 0.15)
        # plt.plot(fx_S[good_frame_S], fy_S[good_frame_S], 'm.', alpha = 0.15)  
        # plt.title("Fish #{0}- Lost Frames {1}".format(fish_number, lost_frame))
     
        # for i in range(0,6):
        #     plt.savefig(Analysis + '/tracking_summary'+ str(fish_number) + '.png', dpi=300)
       
        # Store XMs
        XMs.append([np.mean(fx_NS[good_frame_NS]), np.mean(fx_S[good_frame_S])])

    # Report
    print("Next File: {0}".format(idx))

# Crude calibration: 300 = 28 deg, 800 = 36 deg (900/8) pixels per degree
XM_values = np.array(XMs)/62.5
TTSs = XM_values[:,1] - XM_values[:,0]

# Stats: paired Ttest mean position of each fish in NS vs S
s, pvalue_rel = stats.ttest_rel(XM_values[:,1], XM_values[:,0])
s, pvalue_1samp = stats.ttest_1samp(TTSs, 0)

mean_TTS = np.mean(XM_values[:,1] - XM_values[:,0])
sem_TTS = np.std(XM_values[:,1] - XM_values[:,0])/np.sqrt(len(TTSs)-1)



# Make a Plot using Color Gradient Function 
#cm = plt.cm.get_cmap('plasma') #Choose color map 

# # Plot histogram
# plt.figure()
# np, bins, patches = plt.hist(TTSs, 15)
# bin_centers = 0.5 * (bins[:-1] + bins[1:])

# # scale values to interval [0,1]
# col = bin_centers - min(bin_centers)
# col /= max(col)

# for c, p in zip(col, patches):
#     plt.setp(p, 'facecolor', cm(c))

# plt.vlines(0, 0, 12.5, 'k',)
# plt.xlabel('Tolerated Temperature Shift (°C)')
# plt.ylabel('Fish Count')
# plt.title('Tolerated Temperature Shift (TTS) During Social Cue Viewing\nMean TTS +/- SEM: {0:0.2f} +/- {1:0.2f}\n(p-value: {2:0.4f})'.format(mean_TTS, sem_TTS, pvalue_rel))
# plt.ylim([0, 12.5])
# plt.xlim([-2,6])
# plt.text(4.5, 7, 'n={0}'.format(len(TTSs)))
# sns.despine()
# plt.tight_layout() 

# plt.savefig(base_path + '/Figures/TTS_Histplot.png')
# plt.savefig(base_path + '/Figures/TTS_Histplot.eps')

#Plot unique Color Histogram
plt.vlines(0, 0, 18, 'k')
plt.hist(TTSs, bins=20, color='Steelblue')
plt.xlabel('Tolerated Temperature Shift (°C)')
plt.ylabel('Fish Count')
plt.title('Mean position of Test fish in Social vs Non_Social trials\nMean TTS +/- SEM: {0:0.2f} +/- {1:0.2f}\n(p-value: {2:0.4f})'.format(mean_TTS, sem_TTS, pvalue_rel))
plt.ylim([0, 18])
plt.xlim([-4,8])
plt.text(4.5, 14, 'n={0}'.format(len(TTSs)))
sns.despine()
plt.show()

plt.savefig(base_path + '/Figures/TTS_Histplot.png')
plt.savefig(base_path + '/Figures/TTS_Histplot.eps')


#Make Distribution Plot 

plt.figure()
plt.vlines(0, 0, 0.75, 'k',)
sns.distplot(TTSs, bins=15)
plt.xlabel('Tolerated Temperature Shift (°C)')
plt.title('Tolerated Temperature Shift (TTS) During Social Cue Viewing\nMean TTS +/- SEM: {0:0.2f} +/- {1:0.2f}\n(p-value: {2:0.4f})'.format(mean_TTS, sem_TTS, pvalue_rel))
plt.text(4.5, 0.45, 'n={0}'.format(len(TTSs)))
sns.despine()
plt.tight_layout() 

plt.savefig(base_path + '/Figures/TTS_Distplot.png')
plt.savefig(base_path + '/Figures/TTS_Distplot.eps')





# #FIN