#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:00:40 2021

@author: alizeekastler
Bouts
"""                        
# Set Library Path - Social_Pain Repos
#lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'/Users/alizeekastler/Desktop'
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'

# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Import local modules

import SP_utilities as SPU
import SP_Analysis as SPA
import BONSAI_ARK

plot = True 


# Set threshold
freeze_threshold = 500
motionStartThreshold = 0.03
motionStopThreshold = 0.01 

analysisFolder = base_path + '/Analysis' 
# Read folder list
FolderlistFile = base_path + '/Folderlist_New.txt' 
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)
 
# Get Folder Names
for idx,folder in enumerate(folderNames):
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    #Load Crop regions NS
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    NS_ROIs = ROIs[:,:]
    
    #Load Crop regions S
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    S_ROIs = ROIs[:,:]
    
    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
    
    # Analyze and plot each Fish
    for i in range(0,6):
        
        # Only use "good" fish
        if fishStat[i] == 1:
            
            if plot:
                plt.figure(figsize=(10, 12), dpi=300)
            
            # ---------------------
            
            # Extract tracking data (NS)     
            tracking_file = NS_folder + r'/tracking' + str(i+1) +'.npz'
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
            
            
            # Compute BPS (NS)
            BPS_NS = SPA.measure_BPS(motion_NS, motionStartThreshold, motionStopThreshold)
           
            # Compute Distance Traveled (NS)
            #DistanceT_NS = SPA.distance_traveled(bx_NS, by_NS, NS_ROIs)
           
            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_NS, Pauses_NS = SPA.analyze_bouts_and_pauses(bx_NS, by_NS,ort_NS, motion_NS, motionStartThreshold, motionStopThreshold)
            Percent_Moving_NS = 100 * np.sum(Bouts_NS[:,8])/len(motion_NS)
            Percent_Paused_NS = 100 * np.sum(Pauses_NS[:,8])/len(motion_NS)
            # Count Freezes(NS)
            Freezes_NS = np.array(np.sum(Pauses_NS[:,8] > freeze_threshold))
            
            if plot: 
                
                plt.subplot(2,2,1)
                plt.title('BPS_NS: ' + format(BPS_NS, '.3f')+',%Moved:'+ format(Percent_Moving_NS, '.2f')+ ',%Paused:'+ format(Percent_Paused_NS, '.2f'))
                motion_NS[motion_NS == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion_NS)
            
            
                plt.subplot(2,2,3)
                plt.title('Area_NS')
                plt.plot(area_NS, c='teal')
                
        
                
            # ------------------------------
            
            # Extract tracking data (S)     
            tracking_file = S_folder + r'/tracking' + str(i+1) +'.npz'
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
    
            #Compute BPS (S)
            BPS_S = SPA.measure_BPS(motion_S, motionStartThreshold, motionStopThreshold)
            
            # Compute Distance Traveled (S)
            #DistanceT_S = SPA.distance_traveled(bx_S, by_S, S_ROIs)
            
            # Analyze "Bouts" and "Pauses" (S)
            Bouts_S, Pauses_S = SPA.analyze_bouts_and_pauses(bx_S, by_S,ort_S, motion_S, motionStartThreshold, motionStopThreshold)
            Percent_Moving_S = 100 * np.sum(Bouts_S[:,8])/len(motion_S)
            Percent_Paused_S = 100 * np.sum(Pauses_S[:,8])/len(motion_S)
            # Count Freezes(S)
            Freezes_S = np.array(np.sum(Pauses_S[:,8] > freeze_threshold))
            
            if plot:
                
                plt.subplot(2,2,2)
                plt.title('BPS_S: ' + format(BPS_S, '.3f')+',%Moved:'+ format(Percent_Moving_S, '.2f')+ ',%Paused:'+ format(Percent_Paused_S, '.2f'))
                motion_S[motion_S == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion_S)
            
            
                plt.subplot(2,2,4)
                plt.title('Area_S')
                plt.plot(area_S, c='teal')
                
                

            #----------------------------
            
            # Save figure and data for each fish
            if plot: 
                filename = analysisFolder + '/' + str(np.int(groups[idx])) + 'Motion' + str(i+1) + '.png'  
                plt.savefig(filename, dpi=300)
                plt.close('all')

            # Save Analyzed Summary Data
            filename = analysisFolder + '/' + str(np.int(groups[idx])) + 'SUMMARY' + str(i+1) + '.npz'
            np.savez(filename,  
                      BPS_NS=BPS_NS,
                      BPS_S=BPS_S,
                      Bouts_NS = Bouts_NS, 
                      Bouts_S = Bouts_S,
                      Pauses_NS = Pauses_NS,
                      Pauses_S = Pauses_S,
                      Freezes_NS = Freezes_NS, 
                      Freezes_S = Freezes_S )
                      #DistanceT_NS = DistanceT_NS,
                      #DistanceT_S = DistanceT_S)
            
    
    # Report Progress
    print (idx)


#FIN




        
        
        
        
        
        
    