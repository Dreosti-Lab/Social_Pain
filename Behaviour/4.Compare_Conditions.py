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


FigureFolder = base_path + '/Habituation_NewChamber38/Figures/Summary'

# Set analysis folder and label for experiment/condition A
analysisFolder_A = base_path + r'/Habituation_NewChamber38/Analysis' 
conditionName_A = "Hab"

# Set analysis folder and label for experiment/condition B
analysisFolder_B = base_path + r'/Habituation_NewChamber38/Susceptible/Analysis' 
conditionName_B = "Low Tolerance"

# Set analysis folder and label for experiment/condition B
analysisFolder_C = base_path + r'/Habituation_NewChamber38/Resilient/Analysis' 
conditionName_C = "High Tolerance"

# # Set analysis folder and label for experiment/condition B
# analysisFolder_D = base_path + r'/NewChamber/Gradient_NewChamber38/Analysis' 
# conditionName_D = "HS_N"

# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B, analysisFolder_C]
conditionNames = [conditionName_A, conditionName_B, conditionName_C]

# Summary Containers
BPS_NS_summary = []
BPS_S_summary = []
DistanceT_NS_summary = []
DistanceT_S_summary = []
Freezes_NS_summary = []
Freezes_S_summary = []
Percent_Moving_NS_summary = []
Percent_Moving_S_summary = []
TTSs_summary = []
stat_TTS_summary = []
avgPosition_NS_summary=[]
avgPosition_S_summary=[]
avgdistPerBout_NS_summary = []
avgdistPerBout_S_summary = []
Binned_Bouts_NS_summary = []
Binned_Bouts_S_summary = []
Binned_Freezes_NS_summary = []
Binned_Freezes_S_summary = []
Binned_PTF_NS_summary = []
Binned_PTF_S_summary = []
Binned_PTM_NS_summary = []
Binned_PTM_S_summary = []

# Go through each condition (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):
    
    # Freeze time threshold
    freeze_threshold = 300 # more than 3 seconds

    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder+'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data       
    BPS_NS_ALL = np.zeros(numFiles)
    BPS_S_ALL = np.zeros(numFiles)
    DistanceT_NS_ALL = np.zeros(numFiles)
    DistanceT_S_ALL = np.zeros(numFiles)    
    Freezes_NS_ALL = np.zeros((0,4))
    Freezes_S_ALL = np.zeros((0,4))
    Percent_Moving_NS_ALL = np.zeros(numFiles)
    Percent_Moving_S_ALL = np.zeros(numFiles)
    avgPosition_NS_ALL = np.zeros(numFiles)
    avgPosition_S_ALL = np.zeros(numFiles)
    avgdistPerBout_NS_ALL = np.zeros(numFiles)
    avgdistPerBout_S_ALL = np.zeros(numFiles)
    Binned_Bouts_NS_ALL = np.zeros((numFiles,15))
    Binned_Bouts_S_ALL = np.zeros((numFiles,15))
    Binned_Freezes_NS_ALL = np.zeros((numFiles,15))
    Binned_Freezes_S_ALL = np.zeros((numFiles,15))
    Binned_PTF_NS_ALL = np.zeros((numFiles,15))
    Binned_PTF_S_ALL = np.zeros((numFiles,15))
    Binned_PTM_NS_ALL = np.zeros((numFiles,15))
    Binned_PTM_S_ALL = np.zeros((numFiles,15))
    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
    
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file   
        BPS_NS = dataobject['BPS_NS']   
        BPS_S = dataobject['BPS_S']
        DistanceT_NS = dataobject['DistanceT_NS']   
        DistanceT_S = dataobject['DistanceT_S']   
        Freezes_NS = dataobject['Freezes_NS']   
        Freezes_S = dataobject['Freezes_S']
        Percent_Moving_NS = dataobject['Percent_Moving_NS']   
        Percent_Moving_S = dataobject['Percent_Moving_S']
        avgPosition_NS = dataobject['avgPosition_NS']
        avgPosition_S = dataobject['avgPosition_S']
        avgdistPerBout_NS = dataobject['avgdistPerBout_NS']
        avgdistPerBout_S = dataobject['avgdistPerBout_S']
        Binned_Bouts_NS = dataobject['Binned_Bouts_NS']
        Binned_Bouts_S = dataobject['Binned_Bouts_S']
        Binned_Freezes_NS = dataobject['Binned_Freezes_NS']
        Binned_Freezes_S = dataobject['Binned_Freezes_S']
        Binned_PTF_NS = dataobject['Binned_PTF_NS']
        Binned_PTF_S = dataobject['Binned_PTF_S']
        Binned_PTM_NS = dataobject['Binned_PTM_NS']
        Binned_PTM_S = dataobject['Binned_PTM_S']
        
        # Make an array with all summary stats
        Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
        Freezes_S_ALL = np.vstack([Freezes_S_ALL, Freezes_S])
        BPS_NS_ALL[f] = BPS_NS
        BPS_S_ALL[f] = BPS_S
        DistanceT_NS_ALL[f] = DistanceT_NS
        DistanceT_S_ALL[f] = DistanceT_S
        Percent_Moving_NS_ALL[f] = Percent_Moving_NS
        Percent_Moving_S_ALL[f] = Percent_Moving_S
        avgPosition_NS_ALL[f] = avgPosition_NS
        avgPosition_S_ALL[f] = avgPosition_S
        avgdistPerBout_NS_ALL[f] = avgdistPerBout_NS
        avgdistPerBout_S_ALL[f] = avgdistPerBout_S
        Binned_Bouts_NS_ALL[f,:] = Binned_Bouts_NS
        Binned_Bouts_S_ALL[f,:] = Binned_Bouts_S
        Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
        Binned_Freezes_S_ALL[f,:] = Binned_Freezes_S
        Binned_PTF_NS_ALL[f,:] = Binned_PTF_NS
        Binned_PTF_S_ALL[f,:] = Binned_PTF_S
        Binned_PTM_NS_ALL[f,:] = Binned_PTM_NS
        Binned_PTM_S_ALL[f,:] = Binned_PTM_S
    
    
    XMs = np.column_stack((avgPosition_NS_ALL, avgPosition_S_ALL))
    XM_values = (np.array(XMs)/6)
    TTSs = XM_values[:,1] - XM_values[:,0]
    # Stats: paired Ttest mean position of each fish in NS vs S
    s, pvalue_rel = stats.ttest_rel(XM_values[:,1], XM_values[:,0])
    
    # Add to summary lists
    BPS_NS_summary.append(BPS_NS_ALL)
    BPS_S_summary.append(BPS_S_ALL)
    
    DistanceT_NS_summary.append(DistanceT_NS_ALL)
    DistanceT_S_summary.append(DistanceT_S_ALL)
    
    Freezes_NS_summary.append(Freezes_NS_ALL)
    Freezes_S_summary.append(Freezes_S_ALL)

    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Moving_S_summary.append(Percent_Moving_S_ALL)
    
    TTSs_summary.append(TTSs)
    stat_TTS_summary.append(pvalue_rel)
    
    avgPosition_NS_summary.append(avgPosition_NS_ALL)
    avgPosition_S_summary.append(avgPosition_S_ALL)
    
    avgdistPerBout_NS_summary.append(avgdistPerBout_NS_ALL)
    avgdistPerBout_S_summary.append(avgdistPerBout_S_ALL)
    
    Binned_Bouts_NS_summary.append(Binned_Bouts_NS_ALL) 
    Binned_Bouts_S_summary.append(Binned_Bouts_S_ALL) 
    Binned_Freezes_NS_summary.append(Binned_Freezes_NS_ALL) 
    Binned_Freezes_S_summary.append(Binned_Freezes_S_ALL) 
    Binned_PTF_NS_summary.append(Binned_PTF_NS_ALL) 
    Binned_PTF_S_summary.append(Binned_PTF_S_ALL) 
    Binned_PTM_NS_summary.append(Binned_PTM_NS_ALL)
    Binned_PTM_S_summary.append(Binned_PTM_S_ALL)
 


    #TTS
XMs = np.column_stack((avgPosition_NS_ALL, avgPosition_S_ALL))
# Crude calibration: 600 pixels / 100mm
XM_values = (np.array(XMs)/6)
TTSs = XM_values[:,1] - XM_values[:,0]

# Stats: paired Ttest mean position of each fish in NS vs S
s, pvalue_rel = stats.ttest_rel(XM_values[:,1], XM_values[:,0])

# Summary plots

# BPS
Pos = plt.figure(figsize = (8,10), dpi=300)
plt.title('Average Position')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(avgPosition_NS_summary[i], name= '1' + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(avgPosition_S_summary[i], name= '2' + name)
    series_list.append(s)
    
Pos_data = pd.concat(series_list, axis=1)
plt.ylabel('Average Position')
plt.ylim(10,120)
ax=sns.boxplot(data=Pos_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=Pos_data, orient="v", palette=['lightsteelblue','lightsteelblue','lightsteelblue','steelblue','steelblue','steelblue'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()

Pos.savefig(FigureFolder + '/Pos.eps', format='eps', dpi=300,bbox_inches= 'tight', transparent =True)    
Pos.savefig(FigureFolder + '/Pos_WT.png', dpi=300, bbox_inches='tight')

test = stats.ttest_ind(series_list[4],series_list[5])




# BPS
BPS = plt.figure(figsize = (8,10), dpi=300)
plt.title('BPS')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(BPS_NS_summary[i], name= '1' + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(BPS_S_summary[i], name= '2' + name)
    series_list.append(s)
    
BPS_data = pd.concat(series_list, axis=1)
plt.ylabel('Number of Bouts Per Second (s)')
plt.ylim(0,5)
ax=sns.boxplot(data=BPS_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=BPS_data, orient="v", palette=['lightsteelblue','lightsteelblue','lightsteelblue','steelblue','steelblue','steelblue'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()

BPS.savefig(FigureFolder + '/BPS.eps', format='eps', dpi=300,bbox_inches= 'tight', transparent =True)    
BPS.savefig(FigureFolder + '/BPS_WT.png', dpi=300, bbox_inches='tight')

test = stats.ttest_ind(series_list[4],series_list[5])


# Distance
DistanceT=plt.figure(figsize= (8,10), dpi=300)
plt.title('Distance Travelled')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(DistanceT_NS_summary[i], name="1" + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(DistanceT_S_summary[i], name="2" + name)
    series_list.append(s)
distance_data = pd.concat(series_list, axis=1)
plt.ylabel('Total Distance Travelled (mm)')
plt.ylim(0,20000)
ax=sns.boxplot(data=distance_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=distance_data, orient="v", palette=['lightsteelblue','lightsteelblue','lightsteelblue','steelblue','steelblue','steelblue'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()

DistanceT.savefig(FigureFolder + '/DistanceT.eps', format='eps', dpi=300,bbox_inches= 'tight', transparent =True)    
DistanceT.savefig(FigureFolder + '/DistanceT_WT.png', dpi=300, bbox_inches='tight')

test = stats.ttest_rel(series_list[2],series_list[5])


# Distance
DistBout=plt.figure(figsize= (8,10), dpi=300)
plt.title('Distance Travelled Per Bout')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(avgdistPerBout_NS_summary[i], name="1" + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(avgdistPerBout_S_summary[i], name="2" + name)
    series_list.append(s)
distBout_data = pd.concat(series_list, axis=1)

plt.ylabel('Total Distance Travelled (mm)')
plt.ylim(0,4)
ax=sns.boxplot(data=distBout_data,color = '#BBBBBB', linewidth=1, showfliers=False)
ax=sns.stripplot(data=distBout_data, orient="v", palette=['lightsteelblue','lightsteelblue','lightsteelblue','steelblue','steelblue','steelblue'],size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()


DistBout.savefig(FigureFolder + '/DistBout.eps', format='eps', dpi=300,bbox_inches= 'tight', transparent =True)    
DistBout.savefig(FigureFolder + '/DistBout_WT.png', dpi=300, bbox_inches='tight')

test = stats.ttest_rel(series_list[2],series_list[5])




# Percent Moving
Moving= plt.figure(figsize= (8,10), dpi=300)
plt.title('Percent Time Moving')
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Percent_Moving_NS_summary[i], name="1" + name)
    series_list.append(s)
    
for i, name in enumerate(conditionNames):
    s = pd.Series(Percent_Moving_S_summary[i], name="2" + name)
    series_list.append(s)
moving_data = pd.concat(series_list, axis=1)
plt.ylabel('% Time Moving')
plt.ylim(0,100)
ax=sns.barplot(data=moving_data,ci=95,  palette=['steelblue','steelblue','steelblue','lightsteelblue','lightsteelblue','lightsteelblue'])
ax=sns.stripplot(data=moving_data, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()


Moving.savefig(FigureFolder + '/Moving_WT.png', dpi=300, bbox_inches='tight')


Shift= plt.figure(figsize= (12,10),dpi=300)
series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(TTSs_summary[i], name="S: " + name)
    series_list.append(s)

sns.swarmplot(data=series_list, orient="v", size=6, color="#919395",  zorder=1)
with plt.rc_context({'lines.linewidth': 1.0}):
    sns.pointplot(data=series_list, orient="v", linewidth=4, ci=68, capsize=0.1, join=False, color='black', zorder=100)
plt.xticks(np.arange(0, 3, step= 1), ('SC', 'HC','HS'), fontsize=18)
plt.hlines(0,-1,3, 'red',linestyles='dotted')
plt.ylabel('Relative Position Shift (mm)')
sns.despine()

Shift.savefig(FigureFolder + '/RPS_WT.png', dpi=300, bbox_inches='tight')

test = stats.ttest_ind(series_list[0], series_list[1], 
                      equal_var=True)


# # cue motion
# cueMotion = plt.figure(figsize=(5,8), dpi=300)
# plt.title('Average Motion, Social Cue', fontsize=18)
# motion_list = []
# for i, name in enumerate(conditionNames):
#     s = pd.Series(cue_motion_summary[i],name=name)
#     motion_list.append(s)

# sns.boxplot(data=motion_list, color = '#BBBBBB', linewidth=2, showfliers=False)
# sns.stripplot(data=motion_list, palette = ['darkblue', 'fuchsia'],size=6, jitter=True, edgecolor="gray")
# plt.ylabel('Average Motion', fontsize=14)
# plt.xticks(np.arange(0, 4, step= 1), ('SC', 'HS','SC', 'HS'), fontsize=12)

# cueMotion.savefig(FigureFolder + '/cueMotion_heat_noheat.png', dpi=300, bbox_inches='tight')


