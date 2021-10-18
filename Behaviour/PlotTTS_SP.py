# -*- coding: utf-8 -*-
"""
Plot Tracking data
"""

# Set Library Path - Social_Pain Repos
lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)
# Set Base Path
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import glob

# Import local modules
import SP_Utilities as SPU
import BONSAI_ARK

# Read folder list file
FolderlistFile = base_path + '/Experiment_62/Folderlist/Exp_62.txt' 
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# XMs
XMs = []
XM_NS = []
XM_S = []

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)
    
    #Load Crop regions (NS and S are the same)
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    ROIs[:,:]

    # ---------------------
    # Get Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file_NS = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        fx_NS,fy_NS,bx_NS, by_NS, ex_NS, ey_NS, area_NS, ort_NS, motion_NS = SPU.getTracking(tracking_file_NS)
       
        # Extract tracking data (S)
        tracking_file_S = S_folder + r'/tracking' + str(fish_number) +'.npz'
        fx_S,fy_S,bx_S, by_S, ex_S, ey_S, area_S, ort_S, motion_S = SPU.getTracking(tracking_file_S)
        
        #Filter out bad data 
        min_x = 264
        max_x = 890
        
        #=============================================================================
        if fish_number == 1:
            min_y = 180
            max_y = 280
        if fish_number == 2:
            min_y = 300
            max_y = 400
        if fish_number == 3:
            min_y = 430
            max_y = 530
        if fish_number == 4:
            min_y = 550
            max_y = 650
        if fish_number == 5:
            min_y = 680
            max_y = 780
        if fish_number == 6:
            min_y = 800
            max_y = 900 
#=============================================================================
 
        # Find good tracking (NS)
        numFrames_NS = len(fx_NS)
        good_frame_NS = (fx_NS > min_x) * (fx_NS < max_x) * (fy_NS > min_y) * (fy_NS < max_y)
        num_good_frames_NS = np.sum(good_frame_NS)
        bad_frame_NS = numFrames_NS-num_good_frames_NS
        XM_NS.append(np.mean(fx_NS[good_frame_NS]))
        NS_values = np.array(XM_NS)
        
        # Find good tracking (S)         
        numFrames_S = len(fx_S)
        good_frame_S = (fx_S > min_x) * (fx_S < max_x) * (fy_S > min_y) * (fy_S < max_y)
        num_good_frames_S = np.sum(good_frame_S)
        bad_frame_S = numFrames_S-num_good_frames_S         
        XM_S.append(np.mean(fx_S[good_frame_S]))
        S_values = np.array(XM_S)
        
        # All lost frames
        lost_frames = bad_frame_S + bad_frame_NS
        
        # Store XMs
        XMs.append([np.mean(fx_NS[good_frame_NS]), np.mean(fx_S[good_frame_S])])
        
        # plt.figure(figsize=(8,2), dpi=600)
        # plt.xlim([160,810])
        # #plt.plot(fx_NS, fy_NS, 'lightsteelblue', alpha=1,  linewidth=1)
        # plt.plot(fx_S, fy_S, 'steelblue', linewidth=1)  
        # plt.title("Fish #{0}".format(fish_number))
        
        # for i in range(0,6):
        #     plt.savefig(Analysis + '/tracking_summary'+ str(fish_number) + '.png', dpi=600)
        

        # NS_values = np.array(XM_NS)
        # S_values = np.array(XM_S)
        XM_values = np.array(XMs)
        TTSs = XM_values[:,1] - XM_values[:,0]


# Plot histogram
plt.figure(figsize = (6,4))
sns.histplot(NS_values, bins=20, color ='white')
     
plt.title('Mean Position Non Social n='+ format(len(NS_values)))
plt.ylabel('Fish Count')
plt.xlim([280,860])
plt.ylim([0,6])
plt.xticks([])
sns.despine()
plt.show()


# Plot histogram
plt.figure(figsize = (6,4))
sns.histplot(S_values, bins=20, color ='white')

plt.title('Mean Position Social n='+ format(len(S_values)))
plt.ylabel('Fish Count')
plt.xlim([280,860])
plt.xticks([])
sns.despine()
plt.show()


# Crude calibration: 280 = 28 deg, 850 = 36 deg (900/8) pixels per degree
XM_values = np.array(XMs)
TTSs = XM_values[:,1] - XM_values[:,0]

# Stats: paired Ttest mean position of each fish in NS vs S
s, pvalue_rel = stats.ttest_rel(XM_values[:,1], XM_values[:,0])
s, pvalue_1samp = stats.ttest_1samp(TTSs, 0)

mean_TTS = np.mean(XM_values[:,1] - XM_values[:,0])
sem_TTS = np.std(XM_values[:,1] - XM_values[:,0])/np.sqrt(len(TTSs)-1)

#Plot TTS Histogram
plt.figure(figsize=(10,8), dpi=300)
plt.vlines(0, 0, 18, 'k')
sns.histplot(TTSs, bins=20, color = 'xkcd:royal', kde=True,line_kws={"linewidth":3})
plt.xlabel('Tolerated Temperature Shift (°C)')
plt.ylabel('Fish Count')
plt.title('Mean position of Test fish in Social vs Non_Social trials\nMean TTS +/- SEM: {0:0.2f} +/- {1:0.2f}\n(p-value: {2:0.4f})'.format(mean_TTS, sem_TTS, pvalue_rel))
plt.ylim([0, 18])
plt.xlim([-4,8])
plt.text(4.5, 14, 'n={0}'.format(len(TTSs)))
sns.despine()
plt.show()

plt.savefig(base_path + '/Figures/TTS_Histplot.png')


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


    


#FIN