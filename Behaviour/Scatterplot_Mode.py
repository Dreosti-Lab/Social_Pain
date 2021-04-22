
"""
Created on Wed Apr  7 19:04:21 2021

@author: alizeekastler
"""

# -----------------------------------------------------------------------------
# Set Library Path - Social Pain Repo
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)
# Set Base Path
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'
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
from scipy.stats import mode
import statistics as st


# Import local modules
import SP_utilities as SPU


# Read folder list file
Folderlist_Heat = base_path + '/Folderlist_New.txt' 

groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Heat)

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
        XMs.append([st.mode(fx_NS[good_frame_NS]), st.mode(fx_S[good_frame_S])])

    # Report
    print("Next File: {0}".format(idx))


XM_values = np.array(XMs)
Position_Heat = pd.DataFrame(data = XM_values, columns = ["Non_Social","Social"])
Position_Heat['condition']='Heat'

# Read folder list file

Folderlist_Control= base_path + '/Folderlist_Control.txt' 

groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Control)

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
        XMs.append([st.mode(fx_NS[good_frame_NS]), st.mode(fx_S[good_frame_S])])

    # Report
    print("Next File: {0}".format(idx))

XM_values = np.array(XMs)
Position_Control = pd.DataFrame(data = XM_values, columns = ["Non_Social","Social"])
Position_Control['condition']='control'


# Read folder list file

Folderlist_Control= base_path + '/Folderlist_Lidocaine.txt' 

groups, ages, folderNames, fishStatus = SPU.read_folder_list(Folderlist_Control)

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
        XMs.append([st.mode(fx_NS[good_frame_NS]), st.mode(fx_S[good_frame_S])])

    # Report
    print("Next File: {0}".format(idx))

XM_values = np.array(XMs)
print(XM_values)
Position_Lidocaine = pd.DataFrame(data = XM_values, columns = ["Non_Social","Social"])
Position_Lidocaine['condition']='Lidocaine'


Position = Position_Heat.append([Position_Control, Position_Lidocaine])
print(Position)


plt.figure(figsize=(10,10), dpi=300)
plt.axis([250,900,250,900])
ax=sns.scatterplot(data=Position, x='Non_Social', y='Social', hue='condition', palette=['steelblue', 'springgreen','coral'])
ax.set(xlabel='Non_Social XM(px)', ylabel='Social XM(px)')
plt.title('Most Current Position in  Non_Social vs Social', size=16)
plt.show()