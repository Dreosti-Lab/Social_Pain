# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:07:19 2022

@author: Tom

Compute and save the Average Motion for the Conspecific ROIs
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


# Read folder list
FolderlistFile = base_path + '/NewChamber/Control_NewChamber38/Folderlist_Control.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)


plot = True

# Get Folder Names
for idx,folder in enumerate(folderNames):
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)
        
    #Load Crop regions S
    bonsaiFiles = glob.glob(S_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    S_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)

    #Extend ROIs to the Social cue
    cue_ROIs = SPU.get_cue_ROI(S_ROIs,offset=5)
    
    saveFolder = S_folder + '//ROIs'
    SPU.cycleMkDir(saveFolder)
    
    if plot: 
        
       # grab the image and Plot ROIs
        aviFiles = glob.glob(S_folder+'/*.avi')
        aviFile = aviFiles[0]
        vid = cv2.VideoCapture(aviFile)
        img=SPU.grabFrame(aviFile,500)
            
        color = (255,0)
        weight=10
        for roi in cue_ROIs:
                
                # find corners
                tl = (int(roi[0]),int(roi[1]))
                tr = (int(roi[0]+roi[2]), int(roi[1]))
                bl = (int(roi[0]), int(roi[1]+roi[3]))
                br = (int(roi[0]+roi[2]),int(roi[1]+roi[3]))
    
                # draw the lines            
                cv2.line(img,tl,tr,color,thickness=weight)
                cv2.line(img,tl,bl,color,thickness=weight)
                cv2.line(img,tr,br,color,thickness=weight)
                cv2.line(img,bl,br,color,thickness=weight)
                
                plt.figure()
                plt.imshow(img)
                savename=saveFolder+'/cue_ROIs.png'
                plt.savefig(savename,dpi=300)
                plt.close()

    # compute motion for all ROIs
    cue_motS,background_ROIs=SPA.compute_motion(S_folder,cue_ROIs,change_threshold=0,stepFrames=1000,bFrames = 100)
    
    # save as seperate npy for each ROI
    for idr,roi in enumerate(cue_ROIs):
        filename= saveFolder+'/cue_motion'+str(idr+1)+'.npz'
        np.savez(filename,cue_motS = np.array(cue_motS[:,idr]))
        
    filename=saveFolder+'/initial_backgrounds.png'
    plt.figure('backgrounds')
    for i in range(0,6):
        plt.subplot(2,3,i+1)
        plt.imshow(background_ROIs[i])

    plt.savefig(filename, dpi=300)
    plt.close('backgrounds')
        
        
       
        
    