# -*- coding: utf-8 -*-
"""
Compare parameters from the analysis summary across conditions

@author:Alizee
"""
# -----------------------------------------------------------------------------
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'
#base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'


# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd


FigureFolder = base_path + '/Figure_Grad'

# Set analysis folder and label for experiment/condition A
analysisFolder_A = base_path + r'/NoStim_15min/Analysis' 
conditionName_A = "Baseline"

# Set analysis folder and label for experiment/condition A
analysisFolder_B = base_path + r'/Social/Analysis' 
conditionName_B = "Social"

# Set analysis folder and label for experiment/condition B
analysisFolder_C = base_path + r'/Gradient/Analysis' 
conditionName_C = "Gradient"

# Set analysis folder and label for experiment/condition B
analysisFolder_D = base_path + r'/Gradient_Social/Analysis' 
conditionName_D = "Gradient_Social"

# Set analysis folder and label for experiment/condition B
analysisFolder_E = base_path + r'/Gradient_Isolated/Analysis' 
conditionName_E = "Gradient_Isolated"

# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B, analysisFolder_C, analysisFolder_D, analysisFolder_E]
conditionNames = [conditionName_A, conditionName_B, conditionName_C, conditionName_D, conditionName_E]

# Summary Containers
BPS_NS_summary = []
DistanceT_NS_summary = []
Freezes_NS_summary = []
Percent_Moving_NS_summary = []
Percent_Paused_NS_summary = []
avgdistPerBout_NS_summary = []
Binned_Bouts_NS_summary = []
Binned_Freezes_NS_summary = []
Binned_PTF_summary = []
Binned_PTM_summary = []

# Go through each condition (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):
    
    # Freeze time threshold
    freeze_threshold = 300 # more than 3 seconds

    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder +'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data       
    BPS_NS_ALL = np.zeros(numFiles)
    DistanceT_NS_ALL = np.zeros(numFiles)    
    Freezes_NS_ALL = np.zeros((0,4))
    Percent_Moving_NS_ALL = np.zeros(numFiles)
    Percent_Paused_NS_ALL = np.zeros(numFiles)
    avgdistPerBout_NS_ALL = np.zeros(numFiles)
    Binned_Bouts_NS_ALL = np.zeros((numFiles,15))
    Binned_Freezes_NS_ALL = np.zeros((numFiles,15))
    Binned_PTF_ALL = np.zeros((numFiles,15))
    Binned_PTM_ALL = np.zeros((numFiles,15))
    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
    
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file   
        BPS_NS = dataobject['BPS_S'] 
        DistanceT_NS = dataobject['DistanceT_S']   
        Freezes_NS = dataobject['Freezes_S']   
        Percent_Moving_NS = dataobject['Percent_Moving_S'] 
        Percent_Paused_NS = dataobject['Percent_Paused_S']
        avgdistPerBout_NS = dataobject['avgdistPerBout_S']
        Binned_Bouts_NS = dataobject['Binned_Bouts_S']
        Binned_Freezes_NS = dataobject['Binned_Freezes_S']
        Binned_PTF = dataobject['Binned_PTF_S']
        Binned_PTM = dataobject['Binned_PTM_S']
        
        
        # Make an array with all summary stats
        BPS_NS_ALL[f] = BPS_NS
        DistanceT_NS_ALL[f] = DistanceT_NS
        Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
        Percent_Moving_NS_ALL[f] = Percent_Moving_NS
        Percent_Paused_NS_ALL[f] = Percent_Paused_NS
        avgdistPerBout_NS_ALL[f] = avgdistPerBout_NS
        Binned_Bouts_NS_ALL[f,:] = Binned_Bouts_NS
        Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
        Binned_PTF_ALL[f,:] = Binned_PTF
        Binned_PTM_ALL[f,:] = Binned_PTM

    
    # Add to summary lists
    BPS_NS_summary.append(BPS_NS_ALL)
    DistanceT_NS_summary.append(DistanceT_NS_ALL)
    Freezes_NS_summary.append(Freezes_NS_ALL)
    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Paused_NS_summary.append(Percent_Moving_NS_ALL)
    avgdistPerBout_NS_summary.append(avgdistPerBout_NS_ALL)
    Binned_Bouts_NS_summary.append(Binned_Bouts_NS_ALL) 
    Binned_Freezes_NS_summary.append(Binned_Freezes_NS_ALL) 
    Binned_PTF_summary.append(Binned_PTF_ALL) 
    Binned_PTM_summary.append(Binned_PTM_ALL)
    
# Summary plots

# BPS
BPS = plt.figure(figsize = (8,10), dpi=300)
plt.title('BPS')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(BPS_NS_summary[i], name= name)
    series_list.append(s)
    
   
BPS_data = pd.concat(series_list, axis=1)
plt.ylabel('Number of Bouts Per Second (s)')
#plt.ylim(0,5)
ax=sns.boxplot(data=BPS_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=BPS_data, orient="v", palette=['steelblue', 'purple','lightcoral','tomato', 'orangered'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()


BPS.savefig(FigureFolder + '/BPS.png', dpi=300, bbox_inches='tight')


# PTM binned 
Binned_Bouts = plt.figure(figsize=(14,10)) 
plt.title("Number of Bouts(one minute bins)")


m = np.nanmean(Binned_Bouts_NS_summary[0], 0)
std = np.nanstd(Binned_Bouts_NS_summary[0], 0)
valid = (np.logical_not(np.isnan(Binned_Bouts_NS_summary[0])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'steelblue', LineWidth=4)
plt.plot(m, 'steelblue',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'steelblue', LineWidth=1)
plt.plot(m-se, 'steelblue', LineWidth=1)

m = np.nanmean(Binned_Bouts_NS_summary[1], 0)
std = np.nanstd(Binned_Bouts_NS_summary[1], 0)
valid = (np.logical_not(np.isnan(Binned_Bouts_NS_summary[1])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'purple', LineWidth=4)
plt.plot(m, 'purple',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'purple', LineWidth=1)
plt.plot(m-se, 'purple', LineWidth=1)

m = np.nanmean(Binned_Bouts_NS_summary[2], 0)
std = np.nanstd(Binned_Bouts_NS_summary[2], 0)
valid = (np.logical_not(np.isnan(Binned_Bouts_NS_summary[2])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'lightcoral', LineWidth=4)
plt.plot(m, 'lightcoral', Marker='o', MarkerSize=7)
plt.plot(m+se, 'lightcoral', LineWidth=1)
plt.plot(m-se, 'lightcoral', LineWidth=1)

m = np.nanmean(Binned_Bouts_NS_summary[3], 0)
std = np.nanstd(Binned_Bouts_NS_summary[3], 0)
valid = (np.logical_not(np.isnan(Binned_Bouts_NS_summary[3])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'tomato', LineWidth=4)
plt.plot(m, 'tomato',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'tomato', LineWidth=1)
plt.plot(m-se, 'tomato', LineWidth=1)

m = np.nanmean(Binned_Bouts_NS_summary[4], 0)
std = np.nanstd(Binned_Bouts_NS_summary[4], 0)
valid = (np.logical_not(np.isnan(Binned_Bouts_NS_summary[4])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'orangered', LineWidth=4)
plt.plot(m, 'orangered',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'orangered', LineWidth=1)
plt.plot(m-se, 'orangered', LineWidth=1)

#plt.axis([0, 14, 0.0, 0.02])
plt.xlabel('minutes')
plt.ylabel('Bouts')

plt.tight_layout() 
filename = FigureFolder +'/Binned_Bouts.png'
plt.savefig(filename, dpi=300)



# PTF binned 
Binned_Freezes = plt.figure(figsize=(14,10)) 
plt.title("Number of Freezes(one minute bins)")


m = np.nanmean(Binned_Freezes_NS_summary[0], 0)
std = np.nanstd(Binned_Freezes_NS_summary[0], 0)
valid = (np.logical_not(np.isnan(Binned_Bouts_NS_summary[0])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'steelblue', LineWidth=4)
plt.plot(m, 'steelblue',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'steelblue', LineWidth=1)
plt.plot(m-se, 'steelblue', LineWidth=1)

m = np.nanmean(Binned_Freezes_NS_summary[1], 0)
std = np.nanstd(Binned_Freezes_NS_summary[1], 0)
valid = (np.logical_not(np.isnan(Binned_Freezes_NS_summary[1])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'purple', LineWidth=4)
plt.plot(m, 'purple',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'purple', LineWidth=1)
plt.plot(m-se, 'purple', LineWidth=1)

m = np.nanmean(Binned_Freezes_NS_summary[2], 0)
std = np.nanstd(Binned_Freezes_NS_summary[2], 0)
valid = (np.logical_not(np.isnan(Binned_Freezes_NS_summary[2])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'lightcoral', LineWidth=4)
plt.plot(m, 'lightcoral', Marker='o', MarkerSize=7)
plt.plot(m+se, 'lightcoral', LineWidth=1)
plt.plot(m-se, 'lightcoral', LineWidth=1)

m = np.nanmean(Binned_Freezes_NS_summary[3], 0)
std = np.nanstd(Binned_Freezes_NS_summary[3], 0)
valid = (np.logical_not(np.isnan(Binned_Freezes_NS_summary[3])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'tomato', LineWidth=4)
plt.plot(m, 'tomato',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'tomato', LineWidth=1)
plt.plot(m-se, 'tomato', LineWidth=1)

m = np.nanmean(Binned_Freezes_NS_summary[4], 0)
std = np.nanstd(Binned_Freezes_NS_summary[4], 0)
valid = (np.logical_not(np.isnan(Binned_Freezes_NS_summary[4])))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'orangered', LineWidth=4)
plt.plot(m, 'orangered',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'orangered', LineWidth=1)
plt.plot(m-se, 'orangered', LineWidth=1)

#plt.axis([0, 14, 0.0, 0.02])
plt.xlabel('minutes')
plt.ylabel('Freezes(3s)')

plt.tight_layout() 
filename = FigureFolder +'/Binned_Freezes.png'
plt.savefig(filename, dpi=300)



# Distance
DistanceT=plt.figure(figsize= (8,10), dpi=300)
plt.title('Distance Travelled')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(DistanceT_NS_summary[i], name= name)
    series_list.append(s)

distance_data = pd.concat(series_list, axis=1)
plt.ylabel('Total Distance Travelled (mm)')
#plt.ylim(0,20000)
ax=sns.boxplot(data=distance_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=distance_data, orient="v", palette=['steelblue','purple','lightcoral','tomato', 'orangered'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()


DistanceT.savefig(FigureFolder + '/DistanceT_WT.png', dpi=300, bbox_inches='tight')



# Distance
DistBout=plt.figure(figsize= (8,10), dpi=300)
plt.title('Distance Travelled Per Bout')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(avgdistPerBout_NS_summary[i], name= name)
    series_list.append(s)
    
distBout_data = pd.concat(series_list, axis=1)

plt.ylabel('Total Distance Travelled (mm)')
#plt.ylim(0,4)
ax=sns.boxplot(data=distBout_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=distBout_data, orient="v", palette=['steelblue','purple','lightcoral','tomato', 'orangered'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()



DistBout.savefig(FigureFolder + '/DistBout_WT.png', dpi=300, bbox_inches='tight')


# Percent Moving
Moving= plt.figure(figsize= (8,10), dpi=300)
plt.title('Percent Time Moving')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Percent_Moving_NS_summary[i], name=name)
    series_list.append(s)
    
moving_data = pd.concat(series_list, axis=1)
plt.ylabel('% Time Moving')
plt.ylim(0,100)
ax=sns.barplot(data=moving_data,ci=95,  palette=['steelblue','purple','lightcoral','tomato', 'orangered'])
ax=sns.stripplot(data=moving_data, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()


Moving.savefig(FigureFolder + '/Moving_WT.png', dpi=300, bbox_inches='tight')

