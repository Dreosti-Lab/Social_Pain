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

# Import local modules
import SP_Utilities as SPU

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
    # Analyse Tracking for each fish 
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
    
XM_values = np.array(XMs)

plt.figure(figsize=(10,10))
plt.axis([250,900,250,900])
sns.scatterplot(x=XM_values[:,0], y=XM_values[:,1])
plt.xlabel('Non_Social')
plt.ylabel('Social')
plt.title('Mean Position Non_Social vs Social', size=16)
plt.show()
    

# Crude calibration: 300 = 28 deg, 800 = 36 deg (900/8) pixels per degree
XM_values = np.array(XMs)/62.5
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
#plt.savefig(base_path + '/Figures/TTS_Histplot.eps')



#FIN