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
import SP_Analysis as SPA
import BONSAI_ARK

plot = False


# Set threshold
long_freeze_threshold = 1000
short_freeze_threshold = 300
motionStartThreshold = 0.02
motionStopThreshold = 0.002 

analysisFolder = base_path + '/Analysis_Lidocaine' 
# Read folder list
FolderlistFile = base_path + '/Folderlist_Lidocaine.txt' 
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
    
    Time_3SFreezes_NS = []
    Time_3SFreezes_S = []
    # Analyze and plot each Fish
    for i in range(0,6):
        
        # Only use "good" fish
        if fishStat[i] == 1:
            
            if plot:
                plt.figure(figsize=(10, 12), dpi=300)

            # Extract tracking data (NS)     
            tracking_file_NS = NS_folder + r'/tracking' + str(i+1) +'.npz'
            fx_NS,fy_NS,bx_NS, by_NS, ex_NS, ey_NS, area_NS, ort_NS, motion_NS = SPU.getTracking(tracking_file_NS)
            # Extract tracking data (S)
            tracking_file_S = S_folder + r'/tracking' + str(i+1) +'.npz'
            fx_S,fy_S,bx_S, by_S, ex_S, ey_S, area_S, ort_S, motion_S = SPU.getTracking(tracking_file_S)
        
            # Compute BPS (NS)
            BPS_NS = SPA.measure_BPS(motion_NS, motionStartThreshold, motionStopThreshold)
            #Compute BPS (S)
            BPS_S = SPA.measure_BPS(motion_S, motionStartThreshold, motionStopThreshold)
            
            
            # Compute Distance Traveled (NS)
            DistanceT_NS = SPA.distance_traveled(fx_NS, fy_NS, NS_ROIs[i],len(fx_NS))
            # Compute Distance Traveled (S)
            DistanceT_S = SPA.distance_traveled(fx_S, fy_S, S_ROIs[i], len(fx_S))


            # Analyze "Bouts" amd "Pauses" (NS)
            Bouts_NS, Pauses_NS = SPA.analyze_bouts_and_pauses(fx_NS, fy_NS,ort_NS, motion_NS, motionStartThreshold, motionStopThreshold)
                    
            Percent_Moving_NS = 100 * np.sum(Bouts_NS[:,8])/len(motion_NS)
            Percent_Paused_NS = 100 * np.sum(Pauses_NS[:,8])/len(motion_NS)
            # Count Freezes(NS)
            Long_Freezes_NS = np.array(np.sum(Pauses_NS[:,8]> long_freeze_threshold))
            Short_Freezes_NS = np.array(np.sum(Pauses_NS[:,8]> short_freeze_threshold))
            Short_Freezes_X_NS = Pauses_NS[:,1]
            Short_Freezes_Y_NS = Pauses_NS[:,2]
            Short_Freezes_X_NS = Short_Freezes_X_NS[Pauses_NS[:,8]> short_freeze_threshold]
            Short_Freezes_Y_NS = Short_Freezes_Y_NS[Pauses_NS[:,8]> short_freeze_threshold]
            
            Short_Freezes_Time_NS = Pauses_NS[:,0]
            Short_Freezes_Time_NS = Short_Freezes_Time_NS[Pauses_NS[:,8]> short_freeze_threshold]
            
            # Analyze "Bouts" and "Pauses" (S)
            Bouts_S, Pauses_S = SPA.analyze_bouts_and_pauses(fx_S, fy_S, ort_S, motion_S, motionStartThreshold, motionStopThreshold)
            Percent_Moving_S = 100 * np.sum(Bouts_S[:,8])/len(motion_S)
            Percent_Paused_S = 100 * np.sum(Pauses_S[:,8])/len(motion_S )
            # Count Freezes(S)
            Long_Freezes_S = np.array(np.sum(Pauses_S[:,8] > long_freeze_threshold))
            Short_Freezes_S = np.array(np.sum(Pauses_S[:,8] > short_freeze_threshold))
            Short_Freezes_X_S = Pauses_S[:,1]
            Short_Freezes_Y_S = Pauses_S[:,2]
            Short_Freezes_X_S = Short_Freezes_X_S[Pauses_S[:,8]> short_freeze_threshold]
            Short_Freezes_Y_S = Short_Freezes_Y_S[Pauses_S[:,8]> short_freeze_threshold]
            
            Short_Freezes_Time_S = Pauses_S[:,0]
            Short_Freezes_Time_S = Short_Freezes_Time_S[Pauses_S[:,8]> short_freeze_threshold]
            
            plt.subplot()
            plt.scatter(Short_Freezes_X_NS,Short_Freezes_Y_NS, color= 'lightsteelblue')
            plt.scatter(Short_Freezes_X_S,Short_Freezes_Y_S, color= 'steelblue')
            
            
            #Freezes binning 1min
            movieLength=15 # mins
            FPS=100
            length_fr=movieLength*60*FPS
            binsize=1 # bin size in minutes for freezing time plot
            binsizeFrames=1*60*FPS
            
            #(NS)
            num_pauses=len(Short_Freezes_Time_NS)
            
            Binned_3SFreezes_NS=[]
            for x in range(0, length_fr,binsizeFrames):
                if x>0:
                    boo=Short_Freezes_Time_NS<x
                    Binned_3SFreezes_NS.append(np.sum(boo[Short_Freezes_Time_NS>(x-binsizeFrames)]))                
            Time_3SFreezes_NS.append(Binned_3SFreezes_NS)
            
            #(S)
            num_pauses=len(Short_Freezes_Time_S)
            
            Binned_3SFreezes_S=[]
            for x in range(0, length_fr,binsizeFrames):
                if x >0:
                    boo=Short_Freezes_Time_S<x
                    Binned_3SFreezes_S.append(np.sum(boo[Short_Freezes_Time_S>(x-binsizeFrames)]))                
            Time_3SFreezes_S.append(Binned_3SFreezes_S)
        
        
            #(NS)
            num_pauses=len(Short_Freezes_Time_NS)
            
            Binned_3SFreezes_NS=[]
            for x in range(0, length_fr,binsizeFrames):
                if x>0:
                    boo=Short_Freezes_Time_NS<x
                    Binned_3SFreezes_NS.append(np.sum(boo[Short_Freezes_Time_NS>(x-binsizeFrames)]))                
            Time_3SFreezes_NS.append(Binned_3SFreezes_NS)
            
            #(S)
            num_pauses=len(Short_Freezes_Time_S)
            
            Binned_3SFreezes_S=[]
            for x in range(0, length_fr,binsizeFrames):
                if x >0:
                    boo=Short_Freezes_Time_S<x
                    Binned_3SFreezes_S.append(np.sum(boo[Short_Freezes_Time_S>(x-binsizeFrames)]))                
            Time_3SFreezes_S.append(Binned_3SFreezes_S)
            
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
            
                
                plt.subplot(2,2,2)
                plt.title('BPS_S: ' + format(BPS_S, '.3f')+',%Moved:'+ format(Percent_Moving_S, '.2f')+ ',%Paused:'+ format(Percent_Paused_S, '.2f'))
                motion_S[motion_S == -1.0] = -0.01
                plt.axhline(motionStartThreshold, c="green")
                plt.axhline(motionStopThreshold, c="red")
                plt.plot(motion_S)
            
            
                plt.subplot(2,2,4)
                plt.title('Area_S')
                plt.plot(area_S, c='teal')
                
            # Save figure and data for each fish
            if plot: 
                filename = analysisFolder + '/' + str(np.int(groups[idx])) + 'Motion' + str(i+1) + '.png'  
                plt.savefig(filename, dpi=300)
                plt.close('all')

            # Save Analyzed Summary Data
            filename = analysisFolder + '/' + str(np.int(groups[idx])) + 'SUMMARY' + str(i+1) + '.npz'
            np.savez(filename,BPS_NS = BPS_NS,BPS_S = BPS_S,
                      Bouts_NS = Bouts_NS,Bouts_S = Bouts_S,
                      Pauses_NS = Pauses_NS,Pauses_S = Pauses_S,
                      Long_Freezes_NS = Long_Freezes_NS,Short_Freezes_NS = Short_Freezes_NS,Long_Freezes_S = Long_Freezes_S,Short_Freezes_S=Short_Freezes_S,
                      Short_Freezes_X_NS = Short_Freezes_X_NS,Short_Freezes_Y_NS = Short_Freezes_Y_NS,Short_Freezes_X_S = Short_Freezes_X_S,Short_Freezes_Y_S =  Short_Freezes_Y_S,
                      Binned_3SFreezes_NS = Binned_3SFreezes_NS,Binned_3SFreezes_S = Binned_3SFreezes_S,
                      DistanceT_NS=DistanceT_NS, DistanceT_S=DistanceT_S)
    
    # Report Progress
    print (idx)


    
    # plt.figure()
    # plt.scatter(Short_Freezes_X_S,Short_Freezes_Y_S)
    
    Binned_Freezes_NS = np.sum(Time_3SFreezes_NS, axis=0)
    Binned_Freezes_S = np.sum(Time_3SFreezes_S, axis=0)
    index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15']
    df = pd.DataFrame({'Freezes NS': Binned_Freezes_NS,
                    'Freezes S':Binned_Freezes_S }, index=index)
    df.plot.bar(rot=0, color={"Freezes NS": "lightsteelblue", "Freezes S": "steelblue"}, figsize= (10,6), width=0.8)
    
#FIN




        
        
        
        
        
        
    