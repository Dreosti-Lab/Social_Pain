#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:00:52 2023

@author: alizeekastler
"""
                      
# Set Library Path - Social_Pain Repos
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient/NewChamber'
base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'

# Import useful libraries
import glob
import numpy as np

# Import local modules

import SP_utilities as SPU
import SP_Analysis as SPA
import BONSAI_ARK
import cv2



# Set threshold
freeze_threshold = 200
motionStartThreshold = 0.03
motionStopThreshold = 0.015


AnalysisFolder = base_path + '/NoStim_15min/Analysis'
# Read folder list
FolderlistFile = base_path + '/NoStim_15min/Folderlist.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)


# Get Folder Names
for idx,folder in enumerate(folderNames):
    S_folder = folder + '/Non_Social'

    #Load Crop regions NS
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    S_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        

    x=S_ROIs[:,0]
    y=S_ROIs[:,1]
    width=S_ROIs[:,2]
    height=S_ROIs[:,3]

    

    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
    # Analyze and plot each Fish
    for i in range(0,6):
        
        # Only use "good" fish
       if fishStat[i] == 1:

            # Extract tracking data (NS)     
            tracking_file_S = S_folder + r'/tracking' + str(i+1) +'.npz'
            fx_S,fy_S,bx_S, by_S, ex_S, ey_S, area_S, ort_S, motion_S = SPU.getTracking(tracking_file_S)
            
                
            #15min Movie
            fx_S = fx_S[0:90000]
            fy_S = fy_S[0:90000]
            bx_S = bx_S[0:90000]
            by_S = by_S[0:90000]
            ex_S = ex_S[0:90000]
            ey_S = ey_S[0:90000]
            area_S = area_S[0:90000]
            ort_S = ort_S[0:90000]
            motion_S = motion_S[0:90000]
          
        
            numFrames = 90000  
            FPS= 100

            
            fx_S_mm, fy_S_mm = SPU.convert_mm(fx_S, fy_S, S_ROIs[i])
            
#--------------------------------------------------------------------------------------------------------            
            
            #Smooth Motion
            Smotion_S =SPU.smoothSignal(motion_S,N=3)
        
            # Analyze "Bouts" and "Pauses" 
            Bouts_S, Pauses_S = SPA.analyze_bouts_and_pauses(fx_S_mm, fy_S_mm,ort_S, Smotion_S, S_ROIs[i,1],motionStartThreshold, motionStopThreshold)
            
            # Compute BPS 
            BPS_S = SPA.measure_BPS(Smotion_S, Bouts_S[:,0])
            
            Binned_Bouts_S = SPA.Binning(Bouts_S[:,0], 100, 16, 1) 
            
            # Compute Distance Travelled 
            DistanceT_S = SPA.distance_traveled(fx_S_mm,fy_S_mm,len(fx_S_mm))
            
            avgdistPerBout_S = np.mean(Bouts_S[:,10])
           
            #Analyze Bouts
            B_labels_S, Bout_Angles_S = SPA.label_bouts(Bouts_S[:,9])
          
            
            Turns_S =(np.sum(B_labels_S.Turn))/len(B_labels_S) 
            FSwim_S = (np.sum(B_labels_S.FSwim))/len(B_labels_S)
            

            Percent_Moving_S = (100 * np.sum(Bouts_S[:,8]))/(len(motion_S))
            Percent_Paused_S = (100 * np.sum(Pauses_S[:,8]))/(len(motion_S))
        
        
            # Compute percent time freezing in one minute bins
            moving_frames_S = SPA.fill_bouts(Bouts_S, FPS)
            Binned_PTM_S = SPA.bin_frames(moving_frames_S, FPS)
           
            pausing_frames_S = SPA.fill_pauses(Pauses_S, FPS, freeze_threshold)
        
            Binned_PTF_S = SPA.bin_frames(pausing_frames_S, FPS)
        
        
            # Count Freezes
            Freezes_S, numFreezes_S = SPA.analyze_freezes(Pauses_S, freeze_threshold)
            Binned_Freezes_S = SPA.Binning(Freezes_S[:,0], 100, 16, 1)
    

#-----------------------------------------------------------------------------------------
            

            # Save Analyzed Summary Data
            filename = AnalysisFolder + '/' + str(int(groups[idx])) + 'SUMMARY' + str(i+1) + '.npz'
            np.savez(filename,BPS_S = BPS_S,
                      Bouts_S = Bouts_S, Binned_Bouts_S = Binned_Bouts_S,
                      avgdistPerBout_S=avgdistPerBout_S, 
                      Turns_S=Turns_S, FSwim_S=FSwim_S,
                      Pauses_S = Pauses_S,
                      Percent_Moving_S = Percent_Moving_S, 
                      Percent_Paused_S = Percent_Paused_S, 
                      Freezes_S = Freezes_S, numFreezes_S = numFreezes_S, 
                      Binned_Freezes_S = Binned_Freezes_S, 
                      Binned_PTM_S = Binned_PTM_S, Binned_PTF_S = Binned_PTF_S,
                      DistanceT_S = DistanceT_S)
                     
    
    # Report Progress
    print (idx)
    

    
#FIN


