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
FolderlistFile = base_path + r'/Folderlist_Heat_New.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

analysisFolder = base_path + r'/Analysis_Heat_New'

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
           
            
            #Only look at last 10min of Movie
            fx_NS = fx_NS[30000:90000]
            fy_NS = fy_NS[30000:90000]
            bx_NS = bx_NS[30000:90000]
            by_NS = by_NS[30000:90000]
            ex_NS = ex_NS[30000:90000]
            ey_NS = ey_NS[30000:90000]
            area_NS = area_NS[30000:90000]
            ort_NS = ort_NS[30000:90000]
            motion_NS = motion_NS[30000:90000]
           
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
            
            fx_S = fx_S[30000:90000]
            fy_S = fy_S[30000:90000]
            bx_S = bx_S[30000:90000]
            by_S = by_S[30000:90000]
            ex_S = ex_S[30000:90000]
            ey_S = ey_S[30000:90000]
            area_S = area_S[30000:90000]
            ort_S = ort_S[30000:90000]
            motion_S = motion_S[30000:90000]
            
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
            
            # # Save Analyzed Summary Data
            # filename = analysisFolder + '/' + str(np.int(groups[idx])) + '_ROIs_' + str(i) + '.npz'
            # np.savez(filename,
            #           NS_Cool = NS_Cool,
            #           NS_Hot = NS_Hot,
            #           NS_Noxious = NS_Noxious,
            #           S_Cool = S_Cool, 
            #           S_Hot = S_Hot, 
            #           S_Noxious = S_Noxious)
            
# #Plot_NS
NS1 = pd.Series(NS_Cool, name='Cool')
NS2 = pd.Series(NS_Hot, name='Hot')
NS3 = pd.Series(NS_Noxious, name='Noxious')
NS_areadist = pd.concat([NS1,NS2,NS3], axis=1)

# NS1 = pd.Series(NS_Cool, name='1')
# NS2 = pd.Series(NS_Hot, name='2')
# NS3 = pd.Series(NS_Noxious, name='3')
# NS_areadist = pd.concat([NS1,NS2,NS3], axis=1)

plt.figure(figsize=(4,8), dpi=300)
plt.ylim(0,1.2)
sns.set(style="white", font_scale=1.5)
sns.barplot(data=NS_areadist, ci ='sd', palette= ['midnightblue','purple','darkorange'], dodge= False)
#sns.barplot(data=NS_areadist, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=NS_areadist,orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray") 
plt.title('Non Social n=35', pad=10, fontsize=24)
ax.set_ylabel('Proportion of Frames')
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
#sns.barplot(data=NS_T, ci ='sd', palette= ['midnightblue','purple', 'purple','darkorange'], dodge= False)
sns.barplot(data=NS_T, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=NS_T,orient="v", color= 'dimgrey',size=4, jitter=True, edgecolor="gray") 
ax.set_ylim([0, 250])
plt.title('Transitions Non Social')
ax.set(ylabel= 'Number of T')
sns.despine() 
plt.show()



#Plot_S
S1 = pd.Series(S_Cool, name='Cool')
S2 = pd.Series(S_Hot, name='Hot')
S3 = pd.Series(S_Noxious, name='Noxious')
S_areadist = pd.concat([S1,S2,S3], axis=1)

# S1 = pd.Series(S_Cool, name='1')
# S2 = pd.Series(S_Hot, name='2')
# S3 = pd.Series(S_Noxious, name='3')
# S_areadist = pd.concat([S1,S2,S3], axis=1)

plt.figure(figsize=(4,8), dpi=300)
plt.ylim(0,1.2)
sns.barplot(data=S_areadist, ci ='sd', palette= ['midnightblue','purple','darkorange'], dodge= False)
#sns.barplot(data=S_areadist, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=S_areadist,orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray") 
plt.title('Social n=35', pad=10, fontsize=24)
ax.set_ylabel('Proportion of Frames')
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
ax=sns.stripplot(data=S_T,orient="v", color= 'dimgrey',size=4, jitter=True, edgecolor="gray") 
ax.set_ylim([0, 250])
plt.title('Transitions Social')
ax.set(ylabel= 'Number of T')
sns.despine() 
plt.show()




NS1 = pd.Series(NS_Cool, name='NS_Cool')
S1 = pd.Series(S_Cool, name='S_Cool')

NS2 = pd.Series(NS_Hot, name='NS_Hot')
S2 = pd.Series(S_Hot, name='S_Hot')

NS3 = pd.Series(NS_Noxious, name='NS_Noxious')
S3 = pd.Series(S_Noxious, name='S_Noxious')
areadist = pd.concat([NS1,S1,NS2,S2,NS3,S3], axis=1)


plt.figure(figsize=(16,18), dpi=300)
sns.set(style="white", font_scale=2.5)
plt.ylim(0,1.2)
sns.barplot(data=areadist, ci='sd', palette = ['midnightblue', 'midnightblue', 'purple', 'purple', 'darkorange', 'darkorange'] )
#sns.barplot(data=areadist, ci ='sd', palette= ['midnightblue'], dodge= False)
ax=sns.stripplot(data=areadist,orient="v", color= 'dimgrey',size=8, jitter=True, edgecolor="gray") 
plt.title('Control', pad=10, fontsize=32, y=-0.10)
ax.set_ylabel('Proportion of Frames')
sns.despine() 
plt.show()


# NS_areadist['condition']='Non Social'
# S_areadist['condition']='Social'

# areadist = areadist.groupby(['region']).agg({ ['Cool', 'Hot', 'Noxious'],}).reset_index()


# sns.barplot(data=area,x=, ci ='sd', palette= ['midnightblue','purple','darkorange'], dodge= False)

# labels = ['Cool', 'Hot', 'Noxious']

# NS_list = NS_areadist.values.tolist()
# S_list = S_areadist.values.tolist()
# vector = np.vectorize(np.float)
# NS_list = vector(NS_list)

# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars

# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2,NS_list, width, label='Non_Social')
# rects2 = ax.bar(x + width/2,S_list, width, label='Social')



# FIN