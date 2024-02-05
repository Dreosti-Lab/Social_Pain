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
freeze_threshold = 300
motionStartThreshold = 0.03
motionStopThreshold = 0.015


AnalysisFolder = base_path + '/Gradient/Analysis'
# Read folder list
FolderlistFile = base_path + '/Gradient/Folderlist.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)


# Get Folder Names
for idx,folder in enumerate(folderNames):
    NS_folder = folder + '/Social'

    #Load Crop regions NS
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    NS_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        

    x=NS_ROIs[:,0]
    y=NS_ROIs[:,1]
    width=NS_ROIs[:,2]
    height=NS_ROIs[:,3]

    

    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
    # Analyze and plot each Fish
    for i in range(0,6):
        
        # Only use "good" fish
       if fishStat[i] == 1:

            # Extract tracking data (NS)     
            tracking_file_NS = NS_folder + r'/tracking' + str(i+1) +'.npz'
            fx_NS,fy_NS,bx_NS, by_NS, ex_NS, ey_NS, area_NS, ort_NS, motion_NS = SPU.getTracking(tracking_file_NS)
            
                
            #15min Movie
            fx_NS = fx_NS[0:90000]
            fy_NS = fy_NS[0:90000]
            bx_NS = bx_NS[0:90000]
            by_NS = by_NS[0:90000]
            ex_NS = ex_NS[0:90000]
            ey_NS = ey_NS[0:90000]
            area_NS = area_NS[0:90000]
            ort_NS = ort_NS[0:90000]
            motion_NS = motion_NS[0:90000]
          
        
            numFrames = 90000  
            FPS= 100

            
            fx_NS_mm, fy_NS_mm = SPU.convert_mm(fx_NS, fy_NS, NS_ROIs[i])
            
#--------------------------------------------------------------------------------------------------------            
            
            #Smooth Motion
            Smotion_NS =SPU.smoothSignal(motion_NS,N=3)
        
            # Analyze "Bouts" and "Pauses" 
            Bouts_NS, Pauses_NS = SPA.analyze_bouts_and_pauses(fx_NS_mm, fy_NS_mm,ort_NS, Smotion_NS, NS_ROIs[i,1],motionStartThreshold, motionStopThreshold)
            
            # Compute BPS 
            BPS_NS = SPA.measure_BPS(Bouts_NS[:,0])
            
            Binned_Bouts_NS = SPA.Binning(Bouts_NS[:,0], 100, 16, 1) 
            
            # Compute Distance Travelled 
            DistanceT_NS = SPA.distance_traveled(fx_NS_mm,fy_NS_mm,len(fx_NS_mm))
            
            avgdistPerBout_NS = np.mean(Bouts_NS[:,10])
            #Speed = dist/time
            avgSpeedPerBout_NS = np.mean(Bouts_NS[:,10]/(Bouts_NS[:,8]/30))
            
            
            #Analyze Bouts
            B_labels_NS, Bout_Angles_NS = SPA.label_bouts(Bouts_NS[:,9])
          
            
            Turns_NS =(np.sum(B_labels_NS.Turn))/len(B_labels_NS) 
            FSwim_NS = (np.sum(B_labels_NS.FSwim))/len(B_labels_NS)
            

            Percent_Moving_NS = (100 * np.sum(Bouts_NS[:,8]))/(len(motion_NS))
            Percent_Paused_NS = (100 * np.sum(Pauses_NS[:,8]))/(len(motion_NS))
        
            # Count Freezes
            Freezes_NS, numFreezes_NS = SPA.analyze_freezes(Pauses_NS, freeze_threshold)
            Binned_Freezes_NS = SPA.Binning(Freezes_NS[:,0], 100, 16, 1)
            
            # Compute percent time freezing in one minute bins
            moving_frames_NS = SPA.fill_bouts(Bouts_NS, FPS)
            Binned_PTM = SPA.bin_frames(moving_frames_NS, FPS)
           
            freezing_frames_NS = SPA.fill_pauses(Pauses_NS, FPS, freeze_threshold)
            Binned_PTF = SPA.bin_frames(freezing_frames_NS, FPS)
        
    
            avgBout_interval_NS = np.mean(Pauses_NS[:,8])/100

#-----------------------------------------------------------------------------------------
            

            # Save Analyzed Summary Data
            filename = AnalysisFolder + '/' + str(int(groups[idx])) + 'SUMMARY' + str(i+1) + '.npz'
            np.savez(filename,BPS_NS = BPS_NS,
                      Bouts_NS = Bouts_NS, Binned_Bouts_NS = Binned_Bouts_NS,
                      avgdistPerBout_NS=avgdistPerBout_NS,
                      avgSpeedPerBout_NS = avgSpeedPerBout_NS,
                      Turns_NS=Turns_NS, FSwim_NS=FSwim_NS,
                      Pauses_NS = Pauses_NS,
                      Percent_Moving_NS = Percent_Moving_NS, 
                      Percent_Paused_NS = Percent_Paused_NS, 
                      Freezes_NS = Freezes_NS, numFreezes_NS = numFreezes_NS, 
                      Binned_Freezes_NS = Binned_Freezes_NS, 
                      Binned_PTM_NS = Binned_PTM, Binned_PTF_NS = Binned_PTF,
                      DistanceT_NS = DistanceT_NS, 
                      avgBout_interval_NS=avgBout_interval_NS)
                     
    
    # Report Progress
    print (idx)
    

    
#FIN


