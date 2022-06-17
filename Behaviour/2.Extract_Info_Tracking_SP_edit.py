#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:00:40 2021

@author: alizeekastler

Extract information from tracking data and save into npz Summary file for each video of 6 fish
"""                        
# Set Library Path - Social_Pain Repos
#lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'/Users/alizeekastler/Desktop/Project_Pain_Social/Behaviour_Heat_Gradient'
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2

# Import local modules

import SP_utilities as SPU
import SP_Analysis as SPA
import SP_video_TRARK as SPV
import BONSAI_ARK


plot = False
filterTracking = False

# Set threshold
freeze_threshold = 400
motionStartThreshold = 0.02
motionStopThreshold = 0

AnalysisFolder = base_path + '/NewChamber/Control_NewChamber38/Analysis' 
# Read folder list
FolderlistFile = base_path + '/NewChamber/Control_NewChamber38/Folderlist_Control.txt'
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
    
    x=NS_ROIs[:,0]
    y=NS_ROIs[:,1]
    width=NS_ROIs[:,2]
    height=NS_ROIs[:,3]

    
    Threshold_Cool = np.mean(x+(width)/6)
    Threshold_Noxious = np.mean(x+(width)*5/6)
    
    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
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
    
            if filterTracking:
                count_S,ort_S =SPU.filterTrackingFlips(ort_S)
                count_NS,ort_NS =SPU.filterTrackingFlips(ort_NS)
            
            # #Conspecific Motion
            # cue_motion_file = S_folder + r'/ROIs/cue_motion' + str(i+1) + '.npz'
            # npzFile = np.load(cue_motion_file)
            # cue_motS = npzFile['cue_motS']
            # avg_cue_motion = np.mean(cue_motS)
            
        
                
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


            avgPosition_NS = np.mean(fx_NS)
            avgPosition_S = np.mean(fx_S)
#--------------------------------------------------------------------------------------------------------            
            # Compute BPS 
            BPS_NS, avgBout_NS = SPA.measure_BPS(motion_NS, motionStartThreshold, motionStopThreshold)
            BPS_S, avgBout_S = SPA.measure_BPS(motion_S, motionStartThreshold, motionStopThreshold)
            
             # Analyze "Bouts" and "Pauses" 
            Bouts_NS, Pauses_NS = SPA.analyze_bouts_and_pauses(fx_NS, fy_NS,ort_NS, motion_NS, NS_ROIs[i,1], motionStartThreshold, motionStopThreshold)
            Bouts_S, Pauses_S = SPA.analyze_bouts_and_pauses(fx_S, fy_S, ort_S, motion_S, S_ROIs[i,1], motionStartThreshold, motionStopThreshold)       
            
            # Compute Distance Travelled 
            DistanceT_NS, Distance_Frame_NS = SPA.distance_traveled(fx_NS, fy_NS,NS_ROIs[i-1], len(fx_NS))
            Binned_DistanceT_NS = SPA.Binning(Distance_Frame_NS, 6000)
            
            DistanceT_S, Distance_Frame_S = SPA.distance_traveled(fx_S, fy_S, S_ROIs[i-1], len(fx_S))
            Binned_DistanceT_S = SPA.Binning(Distance_Frame_S, 6000)

            #Distance Per bout
            distPerBout_NS = SPA.computeDistPerBout(Bouts_NS[:,1],Bouts_NS[:,2], Bouts_NS[:,5], Bouts_NS[:,6], NS_ROIs)
            distPerBout_S = SPA.computeDistPerBout(Bouts_S[:,1],Bouts_S[:,2], Bouts_S[:,5], Bouts_S[:,6], S_ROIs)
            
            avgdistPerBout_NS = np.mean(distPerBout_NS.distT)
            avgdistPerBout_S = np.mean(distPerBout_S.distT)
            
            #Analyze Bouts
            B_labels_NS = SPA.label_bouts(Bouts_NS, ort_NS)
            B_labels_S = SPA.label_bouts(Bouts_NS,ort_NS)
            
            LTurns_NS =len( B_labels_NS[B_labels_NS.LTurn ==True])
            RTurns_NS = len(B_labels_NS[B_labels_NS.RTurn==True])
            FSwim_NS = len(B_labels_NS[B_labels_NS.FSwim==True])
            
            LTurns_S =np.sum(B_labels_NS.LTurn)
            RTurns_S =np.sum(B_labels_NS.RTurn)
            FSwim_S = np.sum(B_labels_NS.FSwim)
            
            LTurns_S =len( B_labels_S[B_labels_S.LTurn ==True])
            RTurns_S = len(B_labels_S[B_labels_S.RTurn==True])
            FSwim_S = len(B_labels_S[B_labels_S.FSwim==True])
            
            
            BoutType_NS = pd.concat([distPerBout_NS, B_labels_NS],axis=1)
            BoutType_S = pd.concat([distPerBout_S, B_labels_S],axis=1)
        

            Percent_Moving_NS = (100 * np.sum(Bouts_NS[:,8]))/(len(motion_NS))
            Percent_Paused_NS = (100 * np.sum(Pauses_NS[:,8]))/(len(motion_NS))
            
            Percent_Moving_S = (100 * np.sum(Bouts_S[:,8]))/(len(motion_S))
            Percent_Paused_S = (100 * np.sum(Pauses_S[:,8]))/(len(motion_S))
            
            # Speed
            Speed_NS, SpeedAngle_NS = SPA.compute_bout_signals(fx_NS, fy_NS, ort_NS)
            Speed_S, SpeedAngle_S = SPA.compute_bout_signals(fx_S, fy_S, ort_S)
            
            # Count Freezes
            Freezes_NS, numFreezes_NS = SPA.analyze_freezes(Pauses_NS, freeze_threshold)
            Binned_Freezes_NS = SPA.Bin_Freezes(Freezes_NS[:,0], 100, 15, 1)
            
            Freezes_S, numFreezes_S = SPA.analyze_freezes(Pauses_S, freeze_threshold)
            Binned_Freezes_S = SPA.Bin_Freezes(Freezes_S[:,0], 100, 15, 1) 
            
            
#----------------------------------------------------------------------------------------------------------            
           
            # Orientation
            OrtHist_NS_Cool = SPA.ort_histogram(ort_NS[fx_NS < Threshold_Cool])
            OrtHist_NS_Hot = SPA.ort_histogram(ort_NS[(fx_NS >= Threshold_Cool) & (fx_NS <= Threshold_Noxious)])
            OrtHist_NS_Noxious = SPA.ort_histogram(ort_NS[fx_NS > Threshold_Noxious])
            OrtHist_S_Cool = SPA.ort_histogram(ort_S[fx_S < Threshold_Cool])
            OrtHist_S_Hot = SPA.ort_histogram(ort_S[(fx_S >= Threshold_Cool) & (fx_S <= Threshold_Noxious)])
            OrtHist_S_Noxious = SPA.ort_histogram(ort_S[fx_S > Threshold_Noxious])

#--------------------------------------------------------------------------------------------------------------            
            # Divide ROIs
            fx_NS[fx_NS < Threshold_Cool] = 1
            fx_NS[(fx_NS >= Threshold_Cool) & (fx_NS <= Threshold_Noxious)] = 2
            fx_NS[fx_NS > Threshold_Noxious] = 4
            
            fx_S[fx_S < Threshold_Cool] = 1
            fx_S[(fx_S >= Threshold_Cool) & (fx_S <= Threshold_Noxious)] = 2
            fx_S[fx_S > Threshold_Noxious] = 4
            
            # Total Frames in each ROI
            Frames_Cool_NS = np.count_nonzero(fx_NS[fx_NS==1])
            Frames_Hot_NS = (np.count_nonzero(fx_NS[fx_NS==2]))/4
            Frames_Noxious_NS = np.count_nonzero(fx_NS[fx_NS==4])
            totFrames_NS = Frames_Cool_NS + Frames_Hot_NS + Frames_Noxious_NS
            
            Frames_Cool_S = np.count_nonzero(fx_S[fx_S==1])
            Frames_Hot_S = (np.count_nonzero(fx_S[fx_S==2])/4)
            Frames_Noxious_S = np.count_nonzero(fx_S[fx_S==4])
            totFrames_S = Frames_Cool_S + Frames_Hot_S + Frames_Noxious_S
            
            #convert to Dataframe
            Cool_NS = pd.Series(Frames_Cool_NS/totFrames_NS, name='Cool')
            Hot_NS = pd.Series(Frames_Hot_NS/totFrames_NS, name='Hot')
            Noxious_NS = pd.Series(Frames_Noxious_NS/totFrames_NS, name='Noxious')
            Position_NS = pd.concat([Cool_NS,Hot_NS,Noxious_NS], axis=1)
            
            Cool_S = pd.Series(Frames_Cool_S/totFrames_NS, name='Cool')
            Hot_S = pd.Series(Frames_Hot_S/totFrames_NS, name='Hot')
            Noxious_S = pd.Series(Frames_Noxious_S/totFrames_NS, name='Noxious')
            Position_S = pd.concat([Cool_S,Hot_S,Noxious_S], axis=1)            
            
#---------------------------------------------------------------------------------------------------        
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
                filename = AnalysisFolder + '/' + str(np.int(groups[idx])) + 'Motion' + str(i+1) + '.png'  
                plt.savefig(filename, dpi=300)
                plt.close('all')


#-----------------------------------------------------------------------------------------
            

            # Save Analyzed Summary Data
            filename = AnalysisFolder + '/' + str(np.int(groups[idx])) + 'SUMMARY' + str(i+1) + '.npz'
            np.savez(filename,BPS_NS = BPS_NS,BPS_S = BPS_S,
                      Bouts_NS = Bouts_NS,Bouts_S = Bouts_S,BoutType_NS=BoutType_NS, BoutType_S =BoutType_S,
                      avgdistPerBout_NS=avgdistPerBout_NS, avgdistPerBout_S=avgdistPerBout_S,
                      LTurns_NS=LTurns_NS, LTurns_S=LTurns_S,RTurns_NS=RTurns_NS, RTurns_S=RTurns_S, FSwim_NS=FSwim_NS,FSwim_S=FSwim_S,
                      Pauses_NS = Pauses_NS,Pauses_S = Pauses_S,
                      Percent_Moving_NS = Percent_Moving_NS, Percent_Moving_S = Percent_Moving_S, 
                      Percent_Paused_NS = Percent_Paused_NS, Percent_Paused_S = Percent_Paused_S, 
                      Freezes_S = Freezes_S, Freezes_NS = Freezes_NS, numFreezes_NS = numFreezes_NS, numFreezes_S = numFreezes_S,
                      Binned_Freezes_NS = Binned_Freezes_NS,Binned_Freezes_S = Binned_Freezes_S,    
                      DistanceT_NS = DistanceT_NS, DistanceT_S = DistanceT_S, Binned_DistanceT_NS= Binned_DistanceT_NS, Binned_DistanceT_S = Binned_DistanceT_S, 
                      Speed_NS = Speed_NS, Speed_S = Speed_S,
                      OrtHist_NS_Cool = OrtHist_NS_Cool,OrtHist_NS_Noxious = OrtHist_NS_Noxious, OrtHist_S_Cool = OrtHist_S_Cool, OrtHist_S_Noxious = OrtHist_S_Noxious,
                      OrtHist_NS_Hot = OrtHist_NS_Hot, OrtHist_S_Hot = OrtHist_S_Hot,
                      Position_NS=Position_NS, Position_S = Position_S, avgPosition_NS = avgPosition_NS, avgPosition_S = avgPosition_S)
                      #avg_cue_motion = avg_cue_motion)
    
    # Report Progress
    print (idx)
    

    
#FIN




        
        
        
        
        
        
    