# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:58:43 2021

@author: Alizee Kastler
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

motionStartThreshold = 0.02
motionStopThreshold = 0.002 



FolderlistFile = base_path + '/Folderlist_Control.txt' 
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

Time_Bouts_NS  = [] 
Time_Bouts_S  = [] 


for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder) 
    
    #Load Crop regions NS
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    NS_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
        
    #Load Crop regions S
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    S_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
 
    for i in range(0,6):
        fish_number = i + 1
     
        # Extract tracking data (NS)     
        tracking_file_NS = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        fx_NS,fy_NS,bx_NS,by_NS,ex_NS,ey_NS,area_NS,ort_NS,motion_NS = SPU.getTracking(tracking_file_NS)
        
        Bouts_NS, Pauses_NS = SPA.analyze_bouts_and_pauses(fx_NS, fy_NS,ort_NS, motion_NS, motionStartThreshold, motionStopThreshold)
        
        # Extract tracking data (S)
        tracking_file_S = S_folder + r'/tracking' + str(fish_number) +'.npz'
        fx_S,fy_S,bx_S,by_S,ex_S,ey_S,area_S,ort_S,motion_S = SPU.getTracking(tracking_file_S)
        
        Bouts_S, Pauses_S = SPA.analyze_bouts_and_pauses(fx_S, fy_S, ort_S, motion_S, motionStartThreshold, motionStopThreshold)

        #Bouts binning 1min
        movieLength=15 # mins
        FPS=100
        max_frame=movieLength*60*FPS
        binsize=1 # bin size in minutes for freezing time plot
        binning=1*60*FPS
        
        #Temporal Bouts (NS)
        num_bouts = fx_NS.shape[0]
       
        Temporal_Bouts_NS = Bouts_NS[:,0][Bouts_NS[:,8]> motionStartThreshold]
           
        Binned_Bouts_NS=[]
        for x in range(0, max_frame, binning):         
            if x >0 :
                boo = Temporal_Bouts_NS <x
                Binned_Bouts_NS.append(np.sum(boo[Temporal_Bouts_NS>(x-binning)]))
        Time_Bouts_NS.append(Binned_Bouts_NS)    
        
        #Temporal Bouts (S)
        num_bouts = fx_S.shape[0]
       
        Temporal_Bouts_S = Bouts_S[:,0][Bouts_S[:,8]> motionStartThreshold]
           
        Binned_Bouts_S=[]
        for x in range(0, max_frame, binning):         
            if x >0 :
                boo = Temporal_Bouts_S <x
                Binned_Bouts_S.append(np.sum(boo[Temporal_Bouts_S>(x-binning)]))
        Time_Bouts_S.append(Binned_Bouts_S)    
        
        
        # # Compute Distance Traveled (NS)
        # DistanceT_NS = SPA.distance_traveled(fx_NS, fy_NS, NS_ROIs[i],len(fx_NS))

        # Binned_Bouts_S=[]
        # for x in range(0, max_frame, binning):         
        #     if x >0 :
        #         boo = Temporal_Bouts_S <x
        #         Binned_Bouts_S.append(np.sum(boo[Temporal_Bouts_S>(x-binning)]))
        # Time_Bouts_S.append(Binned_Bouts_S)    
        
        
        
        # Compute Distance Traveled (S)
        DistanceT_S = SPA.distance_traveled(fx_S, fy_S, S_ROIs[i], len(fx_S))
 
    Binned_Temporal_Bouts_NS = np.sum(Time_Bouts_NS, axis=0)
    Binned_Temporal_Bouts_S = np.sum(Time_Bouts_S, axis=0)
    index = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '14', '15']
    df = pd.DataFrame({'Bouts NS': Binned_Temporal_Bouts_NS,
                    'Bouts S':Binned_Bouts_S }, index=index)
    df.plot.bar(rot=0, color={"Bouts NS": "lightsteelblue", "Bouts S": "steelblue"}, figsize= (10,6), width=0.8)        
        
    