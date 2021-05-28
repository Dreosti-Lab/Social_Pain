#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:36:53 2021

@author: alizeekastler
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
import BONSAI_ARK

# Read folder list
FolderlistFile = base_path + r'/Experiment_1/Folderlist_Ablated.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

NS_Cool = []
NS_Hot = []
NS_Noxious = []

S_Cool = []
S_Hot = []
S_Noxious = []

NS_THot_Cool = []
NS_THot_Noxious = []
NS_TNoxious_Hot = []
NS_TCool_Hot = []
 
S_THot_Cool = []
S_THot_Noxious = []
S_TNoxious_Hot = []
S_TCool_Hot = []


# Get Folder Names
for idx,folder in enumerate(folderNames):
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)
 
    # Determine Fish Status       
    fishStat = fishStatus[idx, :]
    
    #Load Crop regions (NS and S are the same)
    bonsaiFiles = glob.glob(NS_folder + '/*.bonsai')
    bonsaiFiles = bonsaiFiles[0]
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    ROIs[:,:]
  
    x=ROIs[:,0]
    y=ROIs[:,1]
    width=ROIs[:,2]
    height=ROIs[:,3]
                      
    Threshold_Cool = np.mean(x+(width)/4)
    Threshold_Noxious = np.mean(x+(width)*3/4)
    
    
    # Analyze and plot each Fish
    for i in range(0,6):

        if fishStat[i] == 1:
            # Extract tracking data (NS)      
            tracking_file_NS = NS_folder + r'/tracking' + str(i+1) +'.npz'
            fx_NS,fy_NS,bx_NS, by_NS, ex_NS, ey_NS, area_NS, ort_NS, motion_NS = SPU.getTracking(tracking_file_NS)
           
            numFrames = len (fx_NS)
            
            fx_NS[fx_NS < Threshold_Cool] = 1
            fx_NS[(fx_NS >= Threshold_Cool) & (fx_NS <= Threshold_Noxious)] = 2
            fx_NS[fx_NS > Threshold_Noxious] = 4
            
            # Total Frames in each ROI
            TotFrames_Cool_NS = (np.count_nonzero(fx_NS[fx_NS==1]))
            TotFrames_Hot_NS = (np.count_nonzero(fx_NS[fx_NS==2]))
            TotFrames_Noxious_NS = (np.count_nonzero(fx_NS[fx_NS==4]))
            
            # Proportion of Frames in each ROI 
            NS_Cool.append(TotFrames_Cool_NS / numFrames)
            NS_Hot.append(TotFrames_Hot_NS / numFrames)
            NS_Noxious.append(TotFrames_Noxious_NS / numFrames)
            
            # Transitions
            # difference vector
            NSDiff=np.diff(fx_NS)
            
            NS_THot_Cool.append(np.count_nonzero(NSDiff[NSDiff==1])) # Hot - Cool
            NS_TCool_Hot.append(np.count_nonzero(NSDiff[NSDiff==-1])) # Cool - Hot
            NS_THot_Noxious.append(np.count_nonzero(NSDiff[NSDiff==-2])) # Hot - Noxious
            NS_TNoxious_Hot.append(np.count_nonzero(NSDiff[NSDiff==2])) # Noxious - Hot
            
            
            #Extract tracking data (S)
            tracking_file_S = S_folder + r'/tracking' + str(i+1) +'.npz'
            fx_S,fy_S,bx_S, by_S, ex_S, ey_S, area_S, ort_S, motion_S = SPU.getTracking(tracking_file_S)
            #fx_S = tracking[:,0][0:60000]
            numFrames_S = len(fx_S)
            
            fx_S[fx_S < Threshold_Cool] = 1
            fx_S[(fx_S >= Threshold_Cool) & (fx_S <= Threshold_Noxious)] = 2
            fx_S[fx_S > Threshold_Noxious] = 4
            
            
            # Total Frames in each ROI
            TotFrames_Cool_S = (np.count_nonzero(fx_S[fx_S==1]))
            TotFrames_Hot_S = (np.count_nonzero(fx_S[fx_S==2]))
            TotFrames_Noxious_S = (np.count_nonzero(fx_S[fx_S==4]))
            
            # Proportion of Frames in each ROI 
            S_Cool.append(TotFrames_Cool_S / numFrames_S)
            S_Hot.append(TotFrames_Hot_S / numFrames_S)
            S_Noxious.append(TotFrames_Noxious_S / numFrames_S)
    
            
             # Transitions
            # difference vector
            SDiff=np.diff(fx_S)
            
            S_THot_Cool.append(np.count_nonzero(SDiff[SDiff==1])) # Hot - Cool
            S_TCool_Hot.append(np.count_nonzero(SDiff[SDiff==-1])) # Cool - Hot
            S_THot_Noxious.append(np.count_nonzero(SDiff[SDiff==-2])) # Hot - Noxious
            S_TNoxious_Hot.append(np.count_nonzero(SDiff[SDiff==2])) # Noxious - Hot
            
#Plot_NS
NS1 = pd.Series(NS_Cool, name='Cool')
NS2 = pd.Series(NS_Hot, name='Hot')
NS3 = pd.Series(NS_Noxious, name='Noxious')
NS_areadist = pd.concat([NS1,NS2,NS3], axis=1)

plt.figure(figsize=(4,8), dpi=300)
plt.ylim(0,1)
#sns.barplot(data=NS_areadist, ci ='sd', palette= ['midnightblue','purple','darkorange'], dodge= False)
sns.barplot(data=NS_areadist, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=NS_areadist,orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray") 
plt.title('Time Spent in each ROI Non Social (n=11)')
ax.set(ylabel= 'Proportion of Frames')
sns.despine() 
plt.show()


NST1 = pd.Series(NS_THot_Cool, name='Hot->Cool')
NST2 = pd.Series(NS_TCool_Hot, name='Cool->Hot')
NST3 = pd.Series(NS_THot_Noxious, name='Hot->Noxious')
NST4 = pd.Series(NS_TNoxious_Hot, name='Noxious->Hot')
NS_T = pd.concat([NST1,NST2,NST3,NST4], axis=1)

# NST1 = pd.Series(NS_THot_Cool, name='2->1')
# NST2 = pd.Series(NS_TCool_Hot, name='1->2')
# NST3 = pd.Series(NS_THot_Noxious, name='2->3')
# NST4 = pd.Series(NS_TNoxious_Hot, name='3->2')
# NS_T = pd.concat([NST1,NST2,NST3,NST4], axis=1)

plt.figure(figsize=(6,8), dpi=300)
sns.barplot(data=NS_T, ci ='sd', palette= ['midnightblue','purple', 'purple','darkorange'], dodge= False)
#sns.barplot(data=NS_T, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=NS_T,orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray") 
ax.set_ylim([0, 250])
plt.title('Transitions Non Social')
ax.set(ylabel= 'Number of T')
sns.despine() 
plt.show()



#Plot_S
S1 = pd.Series(S_Cool, name='1')
S2 = pd.Series(S_Hot, name='2')
S3 = pd.Series(S_Noxious, name='3')
S_areadist = pd.concat([S1,S2,S3], axis=1)

plt.figure(figsize=(4,8), dpi=300)
plt.ylim(0,1)
#sns.barplot(data=S_areadist, ci ='sd', palette= ['midnightblue','purple','darkorange'], dodge= False)
sns.barplot(data=S_areadist, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=S_areadist,orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray") 
plt.title('Time Spent in each ROI Social n=(11)')
ax.set(ylabel= 'Proportion of Frames')
sns.despine() 
plt.show()


ST1 = pd.Series(S_THot_Cool, name='Hot->Cool')
ST2 = pd.Series(S_TCool_Hot, name='Cool->Hot')
ST3 = pd.Series(S_THot_Noxious, name='Hot->Noxious')
ST4 = pd.Series(S_TNoxious_Hot, name='Noxious->Hot')
S_T = pd.concat([ST1,ST2,ST3,ST4], axis=1)

# ST1 = pd.Series(S_THot_Cool, name='2->1')
# ST2 = pd.Series(S_TCool_Hot, name='1->2')
# ST3 = pd.Series(S_THot_Noxious, name='2->3')
# ST4 = pd.Series(S_TNoxious_Hot, name='3->2')
# S_T = pd.concat([ST1,ST2,ST3,ST4], axis=1)

plt.figure(figsize=(6,8), dpi=300)
sns.barplot(data=S_T, ci ='sd', palette= ['midnightblue','purple', 'purple','darkorange'], dodge= False)
#sns.barplot(data=S_T, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=S_T,orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray") 
ax.set_ylim([0, 250])
plt.title('Transitions Social')
ax.set(ylabel= 'Number of T')
sns.despine() 
plt.show()


          
    
# FIN