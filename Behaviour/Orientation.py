#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:00:40 2021

@author: alizeekastler
Bouts
"""                        
# Set Library Path - Social_Pain Repos
#lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'/Users/alizeekastler/Desktop'
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Import local modules

import SP_Utilities as SPU
import SP_Ort as SPO
import BONSAI_ARK


# Set threshold
long_freeze_threshold = 1000
short_freeze_threshold = 300
motionStartThreshold = 0.02
motionStopThreshold = 0.002 

analysisFolder = base_path + '/Analysis_Control_New' 
# Read folder list
FolderlistFile = base_path + '/Folderlist_Control_New.txt' 
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)


# Get Folder Names
for idx,folder in enumerate(folderNames):
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    #Load Crop regions NS
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    NS_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        
    #Load Crop regions S
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    S_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
       
    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
    # Analyze and plot each Fish
    for i in range(0,6):
        
        # Only use "good" fish
        if fishStat[i] == 1:
            
            # Extract tracking data (NS)     
            tracking_file_NS = NS_folder + r'/tracking' + str(i+1) +'.npz'
            fx_NS,fy_NS,bx_NS, by_NS, ex_NS, ey_NS, area_NS, ort_NS, motion_NS = SPU.getTracking(tracking_file_NS)
            # Extract tracking data (S)
            tracking_file_S = S_folder + r'/tracking' + str(i+1) +'.npz'
            fx_S,fy_S,bx_S, by_S, ex_S, ey_S, area_S, ort_S, motion_S = SPU.getTracking(tracking_file_S)

            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_NS, Pauses_NS = SPO.analyze_bouts_and_pauses(fx_NS, fy_NS,ort_NS, motion_NS, motionStartThreshold, motionStopThreshold)
             
            #Orientation 
            Median_ort_NS = Pauses_NS[:,3]
            #Median_ort_NS = Median_ort_NS[Pauses_NS[:,8]> short_freeze_threshold]
            Stop_ort_NS = Pauses_NS[:,7]
            #Stop_ort_NS = Stop_ort_NS[Pauses_NS[:,8]> short_freeze_threshold]
            
            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_S, Pauses_S = SPO.analyze_bouts_and_pauses(fx_S, fy_S,ort_S, motion_S, motionStartThreshold, motionStopThreshold)
            
            #Orientation 
            Median_ort_S = Pauses_S[:,3]
            #Median_ort_S = Median_ort_S[Pauses_S[:,8]> short_freeze_threshold]
            Stop_ort_S = Pauses_S[:,7]
            #Stop_ort_S = Stop_ort_S[Pauses_S[:,8]> short_freeze_threshold]


            plt.scatter(Median_ort_NS, Stop_ort_NS, c ='lightsteelblue')
            plt.scatter(Median_ort_S, Stop_ort_S, c = 'steelblue')
            
            plt.title('Difference in Orientation during Freezing')
            plt.xlabel('Median Orientation when freezing')
            plt.ylabel('Orientation when freeze stops')
         
            
    # Report Progress
    print (idx)

#FIN




        
        
        
        
        
        
    