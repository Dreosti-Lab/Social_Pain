#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:00:40 2021

@author: alizeekastler
Compare Non Social and Social for one condition
"""                        
# Set Library Path - Social_Pain Repos
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'
#base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'


# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from scipy import stats


# Specify Analysis folder

AnalysisFolder = base_path + '/Habituation_NewChamber38/Analysis' 
FigureFolder = base_path + '/Habituation_NewChamber38/Figures' 

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(AnalysisFolder+'/*.npz')

# Calculate how many files
numFiles = np.size(npzFiles, 0)

# Allocate space for summary data
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
avgPosition_NS_ALL = np.zeros(numFiles)
avgPosition_S_ALL = np.zeros(numFiles)
avgdistPerBout_NS_ALL = np.zeros(numFiles)
avgdistPerBout_S_ALL = np.zeros(numFiles)
#avg_cue_motion_ALL = np.zeros(numFiles)
LTurns_NS_ALL = np.zeros(numFiles)
LTurns_S_ALL = np.zeros(numFiles)
RTurns_NS_ALL = np.zeros(numFiles)
RTurns_S_ALL = np.zeros(numFiles)
FSwim_NS_ALL = np.zeros(numFiles)
FSwim_S_ALL = np.zeros(numFiles)
numFreezes_NS_ALL = np.zeros(numFiles)
numFreezes_S_ALL = np.zeros(numFiles)
Binned_Freezes_NS_ALL = np.zeros((numFiles,14))
Binned_Freezes_S_ALL = np.zeros((numFiles,14))
#Binned_DistanceT_NS_ALL = np.zeros((numFiles,15))
#Binned_DistanceT_S_ALL = np.zeros((numFiles,15))
Percent_Moving_NS_ALL = np.zeros(numFiles)
Percent_Moving_S_ALL = np.zeros(numFiles)
Percent_Paused_NS_ALL = np.zeros(numFiles)
Percent_Paused_S_ALL = np.zeros(numFiles)
DistanceT_NS_ALL = np.zeros(numFiles)
DistanceT_S_ALL = np.zeros(numFiles)
Bouts_NS_ALL = np.zeros((0,9))
Bouts_S_ALL = np.zeros((0,9))
#fish_Position_NS_ALL = np.zeros((0,2))
#fish_Position_S_ALL = np.zeros((0,2))
Pauses_NS_ALL = np.zeros((0,9))   
Pauses_S_ALL = np.zeros((0,9))
Freezes_S_ALL = np.zeros((0,4))
Freezes_NS_ALL = np.zeros((0,4))
BoutType_NS_ALL = np.zeros((0,6))
BoutType_S_ALL = np.zeros((0,6))
OrtHist_NS_Cool_ALL = np.zeros((numFiles,36))
OrtHist_NS_Hot_ALL = np.zeros((numFiles,36))
OrtHist_NS_Noxious_ALL = np.zeros((numFiles,36))
OrtHist_S_Cool_ALL = np.zeros((numFiles,36))
OrtHist_S_Hot_ALL = np.zeros((numFiles,36))
OrtHist_S_Noxious_ALL = np.zeros((numFiles,36))
Position_NS_ALL = np.zeros((numFiles,3))
Position_S_ALL = np.zeros((numFiles,3))

                           
# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename, allow_pickle = True)
    
    # Extract from the npz file 
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']
    #fish_Position_NS = dataobject['fish_Position_NS']
    #fish_Position_S = dataobject['fish_Position_S']
    BoutType_NS = dataobject['BoutType_NS']
    BoutType_S = dataobject['BoutType_S']
    LTurns_NS = dataobject['LTurns_NS']
    LTurns_S = dataobject['LTurns_S']
    RTurns_NS = dataobject['RTurns_NS']
    RTurns_S = dataobject['RTurns_S']
    FSwim_NS = dataobject['FSwim_NS']
    FSwim_S = dataobject['FSwim_S']
    Pauses_NS = dataobject['Pauses_NS']   
    Pauses_S = dataobject['Pauses_S']
    Freezes_NS = dataobject['Freezes_NS']
    Freezes_S = dataobject['Freezes_S']
    Percent_Moving_NS = dataobject['Percent_Moving_NS']
    Percent_Moving_S = dataobject['Percent_Moving_S']
    Percent_Paused_NS = dataobject['Percent_Paused_NS']
    Percent_Paused_S = dataobject['Percent_Paused_S']
    numFreezes_NS = dataobject['numFreezes_NS']
    numFreezes_S = dataobject['numFreezes_S']
    Binned_Freezes_NS = dataobject['Binned_Freezes_NS']
    Binned_Freezes_S = dataobject['Binned_Freezes_S']
    DistanceT_NS = dataobject['DistanceT_NS']
    DistanceT_S = dataobject['DistanceT_S']
    #Binned_DistanceT_NS = dataobject['Binned_DistanceT_NS']
    #Binned_DistanceT_S = dataobject['Binned_DistanceT_S']
    OrtHist_NS_Cool = dataobject['OrtHist_NS_Cool']
    OrtHist_NS_Hot = dataobject['OrtHist_NS_Hot']
    OrtHist_NS_Noxious = dataobject['OrtHist_NS_Noxious']
    OrtHist_S_Cool = dataobject['OrtHist_S_Cool']
    OrtHist_S_Hot = dataobject['OrtHist_S_Hot']
    OrtHist_S_Noxious = dataobject['OrtHist_S_Noxious']
    Position_NS = dataobject['Position_NS']
    Position_S = dataobject['Position_S']
    avgPosition_NS = dataobject['avgPosition_NS']
    avgPosition_S = dataobject['avgPosition_S']
    avgdistPerBout_NS = dataobject['avgdistPerBout_NS']
    avgdistPerBout_S = dataobject['avgdistPerBout_S']
    #avg_cue_motion = dataobject['avg_cue_motion']
    
    # Make an array with all summary stats
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    numFreezes_NS_ALL[f] = numFreezes_NS
    numFreezes_S_ALL[f] = numFreezes_S
    Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
    Binned_Freezes_S_ALL[f,:] = Binned_Freezes_S
    #Binned_DistanceT_NS_ALL[f,:] = Binned_DistanceT_NS
    #Binned_DistanceT_S_ALL[f,:] = Binned_DistanceT_S
    Percent_Moving_NS_ALL[f] = Percent_Moving_NS
    Percent_Moving_S_ALL[f] = Percent_Moving_S
    Percent_Paused_NS_ALL[f] = Percent_Paused_NS
    Percent_Paused_S_ALL[f] = Percent_Paused_S
    Position_NS_ALL[f] = Position_NS
    Position_S_ALL[f] = Position_S
    avgPosition_NS_ALL[f] = avgPosition_NS
    avgPosition_S_ALL[f] = avgPosition_S
    avgdistPerBout_NS_ALL[f] = avgdistPerBout_NS
    avgdistPerBout_S_ALL[f] = avgdistPerBout_S
    #avg_cue_motion_ALL[f] = avg_cue_motion
    LTurns_NS_ALL[f] = LTurns_NS
    LTurns_S_ALL[f] = LTurns_S
    RTurns_NS_ALL[f]= RTurns_NS
    RTurns_S_ALL[f]= RTurns_S
    FSwim_NS_ALL[f] = FSwim_NS
    FSwim_S_ALL[f] = FSwim_S
    DistanceT_NS_ALL[f] = DistanceT_NS
    DistanceT_S_ALL[f] = DistanceT_S
    OrtHist_NS_Cool_ALL[f,:] = OrtHist_NS_Cool
    OrtHist_NS_Hot_ALL[f,:] = OrtHist_NS_Hot
    OrtHist_NS_Noxious_ALL[f,:] = OrtHist_NS_Noxious
    OrtHist_S_Cool_ALL[f,:] = OrtHist_S_Cool
    OrtHist_S_Hot_ALL[f,:] = OrtHist_S_Hot
    OrtHist_S_Noxious_ALL[f,:] = OrtHist_S_Noxious
    # Concat all Pauses/Bouts
    Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
    Bouts_S_ALL = np.vstack([Bouts_S_ALL, Bouts_S])
    #fish_Position_NS_ALL = np.vstack([fish_Position_NS_ALL, fish_Position_NS])
    #fish_Position_S_ALL = np.vstack([fish_Position_S_ALL, fish_Position_S])
    Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
    Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S]) 
    Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
    Freezes_S_ALL = np.vstack([Freezes_S_ALL, Freezes_S])
    BoutType_NS_ALL = np.vstack([BoutType_NS_ALL,BoutType_NS])
    BoutType_S_ALL = np.vstack([BoutType_S_ALL,BoutType_S])
    


# Plot Distance Travelled    
s_DistanceT,pvalue_DistanceT = stats.ttest_rel(DistanceT_NS_ALL, DistanceT_S_ALL) 

s1 = pd.Series(DistanceT_NS_ALL, name='Non Social')
s2 = pd.Series(DistanceT_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)

jitter = 0.05
df_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
df_jitter += np.arange(len(df.columns))

DistanceT, ax = plt.subplots(figsize=(4,10), dpi=300)
sns.boxplot(data=df, color = '#BBBBBB', linewidth=1, showfliers=False)
sns.despine()  
ax.set_title('Distance Travelled (n='+ format(numFiles) + ')' + '\n'+'\n p-value:'+ format(pvalue_DistanceT), fontsize =  24, fontweight = 'medium',y=-0.25)
ax.set_xticklabels(df.columns, fontsize=18)
ax.set_ylabel('Total Distance Travelled (mm)', fontsize= 18)
plt.yticks(fontsize=14)
plt.ylim(0,20000)
ax.plot(df_jitter['Non Social'], df['Non Social'],'o',mec='lightsteelblue',mfc='lightsteelblue', ms=8)
ax.plot(df_jitter['Social'], df['Social'], 'o',mec='steelblue',mfc='steelblue', ms=8)

for idx in df.index:
    ax.plot(df_jitter.loc[idx,['Non Social','Social']], df.loc[idx,['Non Social','Social']], color = 'grey', linewidth = 0.5, linestyle = '--', alpha=0.5)
    
DistanceT.savefig(FigureFolder + '/DistanceT.png', dpi=300, bbox_inches='tight')



# # Plot Distance Travelled Time
# mean_DistanceT_NS = np.mean(Binned_DistanceT_NS_ALL, axis=0)
# sem_DistanceT_NS = stats.sem(Binned_DistanceT_NS_ALL)
# std_DistanceT_NS = np.std(Binned_DistanceT_NS_ALL, axis=0)

# mean_DistanceT_S = np.mean(Binned_DistanceT_S_ALL, axis=0)
# sem_DistanceT_S = stats.sem(Binned_DistanceT_S_ALL)
# std_DistanceT_S = np.std(Binned_DistanceT_S_ALL, axis=0)

# Binned_DistanceT = plt.figure(figsize=(25,20), dpi=300)
# plt.suptitle("Distance Travelled over time"+ '\n n='+ format(numFiles), fontweight="bold", fontsize=30, y=0.9) 

# ax = plt.subplot(221)
# plt.title('Non Social', fontweight="bold", fontsize= 25, y=-0.2)
# plt.plot(mean_DistanceT_NS, color = 'lightsteelblue', linewidth= 3)
# plt.fill_between(np.arange(mean_DistanceT_NS.shape[0]), mean_DistanceT_NS + sem_DistanceT_NS,
#                  mean_DistanceT_NS - sem_DistanceT_NS,color= 'lightsteelblue',alpha=0.2)
# plt.ylabel('Distance Travelled (mm)')
# plt.ylim(200,1000)
# plt.xticks(np.arange(0, 15, step= 1), ('1', '2','3','4','5','6','7', '8', '9', '10', '11','12','13','14','15'))
# plt.xlabel('minutes')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

# ax = plt.subplot(222)
# plt.title('Social', fontweight="bold", fontsize= 25, y=-0.2)
# plt.plot(mean_DistanceT_S, color = 'steelblue', linewidth= 3)
# plt.fill_between(np.arange(mean_DistanceT_S.shape[0]), mean_DistanceT_S + sem_DistanceT_S,
#                  mean_DistanceT_S - sem_DistanceT_S,color= 'steelblue',alpha=0.2)
# plt.ylim(200,1000)
# plt.xticks(np.arange(0, 15, step= 1), ('1', '2','3','4','5','6','7', '8', '9', '10', '11','12','13','14','15'))
# plt.xlabel('minutes')
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.get_yaxis().set_ticks([])

# Binned_DistanceT.savefig(FigureFolder + '/BinnedDistanceT.png', dpi=300, bbox_inches='tight', transparent=True)



# Plot Distance Travelled Per Bout   
s_DistBout,pvalue_DistBout = stats.ttest_rel(avgdistPerBout_NS_ALL, avgdistPerBout_S_ALL) 

s1 = pd.Series(avgdistPerBout_NS_ALL, name='Non Social')
s2 = pd.Series(avgdistPerBout_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)

jitter = 0.05
df_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
df_jitter += np.arange(len(df.columns))

DistBout, ax = plt.subplots(figsize=(4,10), dpi=300)
sns.boxplot(data=df, color = '#BBBBBB', linewidth=1, showfliers=False)
sns.despine()  
ax.set_title('Distance Travelled Per Bout (n=' + format(numFiles) +  ')' + '\n' +'\n p-value:'+ format(pvalue_DistBout),fontsize=24,fontweight = 'medium', y=-0.25)
ax.set_xticklabels(df.columns, fontsize = 18)
ax.set_ylabel('Distance Travelled (mm)', fontsize = 18)
plt.ylim(0,3)
plt.yticks(fontsize=14)
ax.plot(df_jitter['Non Social'], df['Non Social'],'o',mec='lightsteelblue',mfc='lightsteelblue', ms=8)
ax.plot(df_jitter['Social'], df['Social'], 'o',mec='steelblue',mfc='steelblue', ms=8)

for idx in df.index:
    ax.plot(df_jitter.loc[idx,['Non Social','Social']], df.loc[idx,['Non Social','Social']], color = 'grey', linewidth = 0.5, linestyle = '--', alpha=0.5)
    
DistBout.savefig(FigureFolder + '/DistPerBout.png', dpi=300, bbox_inches='tight')



#Plot BPS
s_BPS,pvalue_BPS = stats.ttest_rel(BPS_NS_ALL, BPS_S_ALL) 

s1 = pd.Series(BPS_NS_ALL, name='Non Social')
s2 = pd.Series(BPS_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)

jitter = 0.05
df_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
df_jitter += np.arange(len(df.columns))

BoutsperSecond, ax = plt.subplots(figsize=(4,10), dpi=300)
sns.boxplot(data=df, color = '#BBBBBB', linewidth=1, showfliers=False)
sns.despine()  
ax.set_title('Bouts Per Second (n=' + format(numFiles) + ')'+ '\n' +'\n p-value:'+ format(pvalue_BPS),fontsize=24,fontweight='medium', y=-0.25)
ax.set_xticklabels(df.columns, fontsize=18)
ax.set_ylabel('Number of Bouts Per Second', fontsize=18)
plt.ylim(0,1.5)
plt.yticks(fontsize=14)
ax.plot(df_jitter['Non Social'], df['Non Social'],'o',mec='lightsteelblue',mfc='lightsteelblue', ms=8)
ax.plot(df_jitter['Social'], df['Social'], 'o',mec='steelblue',mfc='steelblue', ms=8)

for idx in df.index:
    ax.plot(df_jitter.loc[idx,['Non Social','Social']], df.loc[idx,['Non Social','Social']], color = 'grey', linewidth = 0.5, linestyle = '--', alpha=0.5)    

BoutsperSecond.savefig(FigureFolder + '/BPS.png', dpi=300, bbox_inches='tight')



#Calculate Number of Bouts
nBouts_NS = BPS_NS_ALL * (90000/100)
nBouts_S = BPS_S_ALL * (90000/100)

s_nBouts,pvalue_nBouts = stats.ttest_rel(nBouts_NS, nBouts_S) 
nBouts = plt.figure(figsize=(4,10), dpi=300)
plt.title(' Total number of Bouts (n=' + format(numFiles) + ')'+ '\n'+ '\n p-value: '+ format(pvalue_nBouts), pad=10, fontsize= 24, y=-0.25)
plt.ylabel('Number of Bouts', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=18)
plt.ylim(0,1000)
s1 = pd.Series(nBouts_NS, name='Non Social')
s2 = pd.Series(nBouts_S, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=8, jitter=True, edgecolor="gray")
sns.despine()
plt.show()

nBouts.tight_layout
nBouts.savefig(FigureFolder + '/nBouts.png', dpi=300, bbox_inches='tight')



#Plot distribution of Bouts // NS vs S

Bouts_map = plt.figure(figsize=(28,10), dpi=300)
plt.suptitle('Distribution of Bouts (n='+ format(numFiles) + ')', fontweight="medium", fontsize=24, x=0.26) 

ax = plt.subplot(221)
plt.hist2d(Bouts_NS_ALL[:,1], Bouts_NS_ALL[:,2],bins=10, cmap='Blues')
plt.title('Non Social',fontsize= 18, y=-0.15) 
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.clim(0,1500)

ax = plt.subplot(222)  
plt.hist2d(Bouts_S_ALL[:,1], Bouts_S_ALL[:,2],bins=10, cmap='Blues')
plt.title('Social', fontsize= 18, y=-0.15)
plt.xticks([])  
plt.yticks([]) 
plt.colorbar()
plt.clim(0,1500)

Bouts_map.tight_layout
Bouts_map.savefig(FigureFolder + '/BoutsMap.png', dpi=300, bbox_inches='tight')



# Plot Percent time Moving
s_Moving,pvalue_Moving = stats.ttest_rel(Percent_Moving_NS_ALL, Percent_Moving_S_ALL) 

Moving = plt.figure(figsize=(4,10), dpi=300)
plt.title('% Moving (n='+ format(numFiles) + ')'+ '\n'+'\n p-value: ' + format(pvalue_Moving), pad=10, fontsize= 24, y=-0.25)
plt.ylabel('% Time Moving', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=18)
plt.ylim(0,100)
s1 = pd.Series(Percent_Moving_NS_ALL, name='Non Social')
s2 = pd.Series(Percent_Moving_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=8, jitter=True, edgecolor="gray")
sns.despine()

Moving.savefig(FigureFolder + '/%Moving.png', dpi=300, bbox_inches='tight')



# Plot Percent time Moving
s_Pausing,pvalue_Pausing = stats.ttest_rel(Percent_Paused_NS_ALL, Percent_Paused_S_ALL) 

Pausing = plt.figure(figsize=(4,10), dpi=300)
plt.title('% Pausing (n='+ format(numFiles) + ')'+ '\n' +'\n p-value: ' + format(pvalue_Pausing), pad=10, fontsize= 24, y=-0.25)
plt.ylabel('% Time Pausing', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=18)
plt.ylim(0,100)
s1 = pd.Series(Percent_Paused_NS_ALL, name='Non Social')
s2 = pd.Series(Percent_Paused_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=8, jitter=True, edgecolor="gray")
sns.despine()

Pausing.savefig(FigureFolder + '/%Pausing.png', dpi=300, bbox_inches='tight')
 

#Make histogram and plot it with lines 
DistPauses = plt.figure(figsize=(10,12), dpi=300)

ax=plt.subplot(221)
sns.histplot((Pauses_NS_ALL[:,8]/100), binwidth=0.5)
plt.xlim(0,5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Dist Pausing(NS)', fontsize=24)
sns.despine()

ax=plt.subplot(222)
sns.histplot((Pauses_S_ALL[:,8]/100), binwidth=0.5)
plt.xlim(0,5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title('Dist Pausing(S)', fontsize=24)
sns.despine()

DistPauses.savefig(FigureFolder + '/Dist_Pausing.png', dpi=300, bbox_inches='tight')



# Plot Short Freezes
s_Freezes,pvalue_Freezes = stats.ttest_rel(numFreezes_NS_ALL, numFreezes_S_ALL) 

s1 = pd.Series(numFreezes_NS_ALL, name='Non Social')
s2 = pd.Series(numFreezes_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)

jitter = 0.05
df_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
df_jitter += np.arange(len(df.columns))

shortFreezes, ax = plt.subplots(figsize=(4,10), dpi=300)
sns.boxplot(data=df, color = '#BBBBBB', linewidth=1, showfliers=False)
sns.despine()  
ax.set_title('3s Freezes (n=' + format(numFiles) + ')'+ '\n' +'\n p-value:'+ format(pvalue_BPS),fontsize=24,fontweight= 'medium', y=-0.25)
ax.set_xticklabels(df.columns, fontsize=18)
ax.set_ylabel('Total Number of Freezes (>3s)', fontsize=18)
plt.yticks(fontsize=14)
ax.plot(df_jitter['Non Social'], df['Non Social'],'o',mec='lightsteelblue',mfc='lightsteelblue', ms=6)
ax.plot(df_jitter['Social'], df['Social'], 'o',mec='steelblue',mfc='steelblue', ms=6)

for idx in df.index:
    ax.plot(df_jitter.loc[idx,['Non Social','Social']], df.loc[idx,['Non Social','Social']], color = 'grey', linewidth = 0.5, linestyle = '--', alpha=0.5)    

shortFreezes.savefig(FigureFolder + '/3sFreezes.png', dpi=300, bbox_inches='tight')



# Scatterplot Position of Freezes
xFreezes = plt.figure(figsize=(8,2), dpi=300)
plt.title('Position of Freezes (n='+ format(numFiles)+ ')', fontsize=18, fontweight='medium', y=1.1)  
plt.scatter(Freezes_NS_ALL[:,1],Freezes_NS_ALL[:,2],color= 'lightsteelblue', label ="Non_Social")
plt.scatter(Freezes_S_ALL[:,1],Freezes_S_ALL[:,2],color= 'steelblue', label = "Social")
plt.xticks([])  
plt.yticks([])
plt.legend(bbox_to_anchor=(0.2, 1.6), fontsize=14)
plt.show()

xFreezes.savefig(FigureFolder + '/Pos_Freezes.png', dpi=300, bbox_inches='tight')



#Plot distribution of Freezes // NS vs S
Freezes_map = plt.figure(figsize=(28,10), dpi=300)
plt.suptitle("Distribution of Freezes"+ '(n='+ format(numFiles)+ ')', fontweight='medium', fontsize=18, x=0.26) 

ax = plt.subplot(221)
plt.hist2d(Freezes_NS_ALL[:,1],Freezes_NS_ALL[:,2], bins=10, cmap='Blues')
plt.title('Non Social',fontsize= 18, y=-0.15) 
plt.xticks([])  
plt.yticks([]) 
plt.colorbar()
plt.clim(0,20)

ax = plt.subplot(222)  
plt.hist2d(Freezes_S_ALL[:,1],Freezes_S_ALL[:,2],bins=10, cmap='Blues')
plt.title('Social',fontsize= 18, y=-0.15)
plt.xticks(fontsize=32)  
plt.yticks(fontsize=32) 
plt.xticks([])  
plt.yticks([]) 
plt.colorbar()
plt.clim(0,20)

Freezes_map.tight_layout
Freezes_map.savefig(FigureFolder + '/FreezeMap.png', dpi=300, bbox_inches='tight')



#Binned Freezes
mean_Freezes_NS = np.mean(Binned_Freezes_NS_ALL, axis=0)
sem_Freezes_NS = stats.sem(Binned_Freezes_NS_ALL)
std_Freezes_NS = np.std(Binned_Freezes_NS_ALL, axis=0)

mean_Freezes_S = np.mean(Binned_Freezes_S_ALL, axis=0)
sem_Freezes_S = stats.sem(Binned_Freezes_S_ALL)
std_Freezes_S = np.std(Binned_Freezes_S_ALL, axis=0)

Binned_Freezes = plt.figure(figsize=(14,10), dpi=300)
plt.suptitle('3s Freezes over time (n='+ format(numFiles)+ ')', fontsize=24) 

ax = plt.subplot(221)
plt.title('Non Social', fontsize= 18, y=-0.30)
plt.plot(mean_Freezes_NS, color = 'lightsteelblue', linewidth= 3)
plt.fill_between(np.arange(mean_Freezes_NS.shape[0]), mean_Freezes_NS + sem_Freezes_NS,
                 mean_Freezes_NS - sem_Freezes_NS,color= 'lightsteelblue',alpha=0.2)
plt.ylabel('mean nb Freezes', fontsize=18)
plt.ylim(0,4)
plt.yticks(fontsize=18)
plt.xticks(np.arange(0, 15, step= 1), ('1', '2','3','4','5','6','7', '8', '9', '10', '11','12','13','14','15'), fontsize=14)
plt.xlabel('minutes', fontsize =18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

ax = plt.subplot(222)
plt.title('Social',fontsize= 18, y=-0.30)
plt.plot(mean_Freezes_S, color = 'steelblue', linewidth= 3)
plt.fill_between(np.arange(mean_Freezes_S.shape[0]), mean_Freezes_S + sem_Freezes_S,
                 mean_Freezes_S - sem_Freezes_S,color= 'steelblue',alpha=0.2)
plt.ylim(0,4)
plt.xticks(np.arange(0, 15, step= 1), ('1', '2','3','4','5','6','7', '8', '9', '10', '11','12','13','14','15'), fontsize=14)
plt.xlabel('minutes', fontsize=18)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_yaxis().set_ticks([])

Binned_Freezes.savefig(FigureFolder + '/BinnedFreezes.png', dpi=300, bbox_inches='tight', transparent=True)



# Scatterplot Position 
s_Pos,pvalue_Pos = stats.ttest_rel(avgPosition_NS_ALL, avgPosition_S_ALL) 

s1 = pd.Series(avgPosition_NS_ALL, name='Non Social')
s2 = pd.Series(avgPosition_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)

jitter = 0.05
df_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=df.values.shape), columns=df.columns)
df_jitter += np.arange(len(df.columns))

Position, ax = plt.subplots(figsize=(4,10), dpi=300)
sns.boxplot(data=df, color = '#BBBBBB', linewidth=1, showfliers=False)
sns.despine()  
ax.set_title('Average Position (n=' + format(numFiles) + ')'+ '\n' + '\n p-value:'+ format(pvalue_Pos),fontsize=24, y=-0.25)
ax.set_xticklabels(df.columns, fontsize=18)
ax.set_ylabel('Average Position(mm)', fontsize=18)
plt.yticks(fontsize=14)
plt.ylim(20,120)
ax.plot(df_jitter['Non Social'], df['Non Social'],'o',mec='lightsteelblue',mfc='lightsteelblue', ms=6)
ax.plot(df_jitter['Social'], df['Social'], 'o',mec='steelblue',mfc='steelblue', ms=6)

for idx in df.index:
    ax.plot(df_jitter.loc[idx,['Non Social','Social']], df.loc[idx,['Non Social','Social']], color = 'grey', linewidth = 0.5, linestyle = '--', alpha=0.5)    

Position.savefig(FigureFolder + '/Avg_Pos.png', dpi=300, bbox_inches='tight')



#stacked_Histogram
normPosition_NS = pd.DataFrame(data = Position_NS_ALL*100/numFiles, columns = ["Cool","Hot","Noxious"])
normPosition_NS['condition']='Non Social'

normPosition_S = pd.DataFrame(data = Position_S_ALL*100/numFiles, columns = ["Cool","Hot","Noxious"])
normPosition_S['condition']='Social'

Position = normPosition_NS.append([normPosition_S])
total = Position.groupby('condition')['Cool','Hot','Noxious'].apply(sum).reset_index()
total['CoolHot'] = total['Cool'] + total['Hot']
total['Total'] = total['Cool']+ total['Hot'] + total['Noxious']

HistStack,ax=plt.subplots(figsize=(8,4), dpi=300)
sns.barplot(x= total['Total'], y='condition', data=total, color='darkorange')
sns.barplot(x=total["CoolHot"], y='condition', data=total, estimator=sum, ci=None,  color='purple')
sns.barplot(x=total["Cool"],y='condition', data=total, estimator=sum, ci=None,  color='midnightblue')
plt.xlabel('Proportion of Frames', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.ylabel(None)
sns.despine()
#setLabels
Cool = mpatches.Patch(color= 'midnightblue', label = 'cool')
Hot = mpatches.Patch(color= 'purple', label = 'hot')
Noxious = mpatches.Patch(color= 'darkorange', label = 'noxious')
plt.legend(handles=[Noxious, Hot, Cool], bbox_to_anchor=(1, 1), fontsize=14)

plt.savefig(FigureFolder + '/HistStack.eps', format='eps', dpi=300,bbox_inches= 'tight', transparent=True)
plt.savefig(FigureFolder + '/HistStack.png', dpi=300, bbox_inches= 'tight', transparent=True  )



#Plot Relative Position Shift
XM_values = np.column_stack((avgPosition_NS_ALL, avgPosition_S_ALL))
TTSs = XM_values[:,1] - XM_values[:,0]

# Stats: paired Ttest mean position of each fish in NS vs S
s, pvalue_rel = stats.ttest_rel(XM_values[:,1], XM_values[:,0])
s, pvalue_1samp = stats.ttest_1samp(TTSs, 0)

mean_TTS = np.mean(XM_values[:,1] - XM_values[:,0])
sem_TTS = np.std(XM_values[:,1] - XM_values[:,0])/np.sqrt(len(TTSs)-1)

#Make histogram and plot it with lines 
bins = np.arange(-40,80,10)
hist,edges=np.histogram(TTSs,bins)
freq = hist/float(hist.sum())

plt.figure(figsize=(6,4), dpi=300)
plt.bar(bins[:-1], freq, width=10, ec="k", color='white' )
plt.ylim(0,0.5)
plt.xlim(-70,70)
plt.title('Mean TTS +/- SEM: {0:0.2f} +/- {1:0.2f}\n(p-value: {2:0.4f})'.format(mean_TTS, sem_TTS, pvalue_rel) + '\n n={0}'.format(len(TTSs)))
plt.xlabel('Position Shift (mm)')
plt.ylabel('Relative Frequency')
sns.despine()
plt.savefig(FigureFolder + '/TTS.eps', format='eps', dpi=300, bbox_inches= 'tight', transparent=True)
plt.savefig(FigureFolder + '/TTS.png', dpi=300, bbox_inches= 'tight', transparent=True  )


# ORT_HIST Summary Plot

# Accumulate all histogram values and normalize
Accum_OrtHist_NS_Cool_ALL = np.sum(OrtHist_NS_Cool_ALL, axis=0)
Accum_OrtHist_NS_Hot_ALL = np.sum(OrtHist_NS_Hot_ALL, axis=0)
Accum_OrtHist_NS_Noxious_ALL = np.sum(OrtHist_NS_Noxious_ALL, axis=0)
Accum_OrtHist_S_Cool_ALL = np.sum(OrtHist_S_Cool_ALL, axis=0)
Accum_OrtHist_S_Hot_ALL = np.sum(OrtHist_S_Hot_ALL, axis=0)
Accum_OrtHist_S_Noxious_ALL = np.sum(OrtHist_S_Noxious_ALL, axis=0)

Norm_OrtHist_NS_Cool_ALL = Accum_OrtHist_NS_Cool_ALL/np.sum(Accum_OrtHist_NS_Cool_ALL)
Norm_OrtHist_NS_Hot_ALL = Accum_OrtHist_NS_Hot_ALL/np.sum(Accum_OrtHist_NS_Hot_ALL)
Norm_OrtHist_NS_Noxious_ALL = Accum_OrtHist_NS_Noxious_ALL/np.sum(Accum_OrtHist_NS_Noxious_ALL)
Norm_OrtHist_S_Cool_ALL = Accum_OrtHist_S_Cool_ALL/np.sum(Accum_OrtHist_S_Cool_ALL)
Norm_OrtHist_S_Hot_ALL = Accum_OrtHist_S_Hot_ALL/np.sum(Accum_OrtHist_S_Hot_ALL)
Norm_OrtHist_S_Noxious_ALL = Accum_OrtHist_S_Noxious_ALL/np.sum(Accum_OrtHist_S_Noxious_ALL)

# Plot Summary
xAxis = np.arange(-np.pi,np.pi+np.pi/18.0, np.pi/18.0)
Ort = plt.figure('Summary: Orientation Histograms', figsize=(25,20), dpi=300)

Ort = plt.subplot(131, polar=True)
plt.title('Cool', fontweight="bold",fontsize= 25, y=-0.2)
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_Cool_ALL, Norm_OrtHist_NS_Cool_ALL[0])), color= 'lightsteelblue',  linewidth = 3)
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_Cool_ALL, Norm_OrtHist_S_Cool_ALL[0])), color = 'steelblue', linewidth = 3)

plt.legend(labels=('Non Social', 'Social'), loc='upper right', bbox_to_anchor=(0.2, 1.2))

Ort = plt.subplot(132, polar=True)
plt.title('Hot', fontweight="bold", fontsize= 25, y=-0.2)
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_Hot_ALL, Norm_OrtHist_NS_Hot_ALL[0])),color='lightsteelblue', linewidth = 3)
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_Hot_ALL, Norm_OrtHist_S_Hot_ALL[0])), color= 'steelblue', linewidth = 3)

Ort = plt.subplot(133, polar=True)
plt.title('Noxious', fontweight="bold",fontsize= 25, y=-0.2)
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_Noxious_ALL, Norm_OrtHist_NS_Noxious_ALL[0])),color='lightsteelblue', linewidth = 3)
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_Noxious_ALL, Norm_OrtHist_S_Noxious_ALL[0])), color= 'steelblue', linewidth = 3)

Ort.figure.savefig(FigureFolder + '/Orientation.png', dpi=300, bbox_inches='tight')






xTurn_NS = (BoutType_NS_ALL[:,0])[BoutType_NS_ALL[:,5] == False]
yTurn_NS = (BoutType_NS_ALL[:,1])[BoutType_NS_ALL[:,5] == False]

xTurn_S = (BoutType_S_ALL[:,0])[BoutType_S_ALL[:,5] == False]
yTurn_S = (BoutType_S_ALL[:,1])[BoutType_S_ALL[:,5] == False]

Turn_map = plt.figure(figsize=(28,10), dpi=300)
plt.suptitle("Distribution of Turns ("+ 'n='+ format(numFiles)+ ')', fontweight='medium', fontsize=24, x=0.26) 

ax = plt.subplot(221)
plt.hist2d(xTurn_NS, yTurn_NS, bins=10, cmap='Blues')
plt.title('Non Social',fontsize= 18, y=-0.15) 
plt.xticks([])
plt.yticks([])
plt.clim(0,1500)
plt.colorbar()

ax = plt.subplot(222)  
plt.hist2d(xTurn_S, yTurn_S ,bins=10, cmap='Blues')
plt.title('Social',fontsize= 18, y=-0.15)
plt.xticks([])
plt.yticks([])
plt.clim(0,1500)
plt.colorbar()

Turn_map.tight_layout
Turn_map.savefig(FigureFolder + '/TurnMap.png', dpi=300, bbox_inches='tight')


B_labels = plt.figure(figsize =(10,12), dpi=300)
plt.suptitle('Frequency of Bout Type (n='+ format(numFiles) + ')', fontsize= 24, fontweight='medium')

s1 = pd.Series(LTurns_NS_ALL, name='Left')
s2 = pd.Series(RTurns_NS_ALL, name='Right')
s3 = pd.Series(FSwim_NS_ALL, name= 'Forward')
Type_NS = pd.concat([s1,s2, s3], axis=1)

s4 = pd.Series(LTurns_S_ALL, name='Left')
s5 = pd.Series(RTurns_S_ALL, name='Right')
s6 = pd.Series(FSwim_S_ALL, name= 'Forward')
Type_S = pd.concat([s4,s5, s6], axis=1)

ax = plt.subplot(221)
sns.barplot(data=Type_NS, ci='sd',  color='lightsteelblue')
sns.stripplot(data=Type_NS, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
plt.title('Non Social',fontsize= 18, y=-0.20) 
plt.ylim(0,1.1)
plt.yticks(fontsize=14)
plt.xticks(fontsize=18)
sns.despine()

ax = plt.subplot(222)
sns.barplot(data=Type_S, ci='sd',  color = 'steelblue')
sns.stripplot(data=Type_S, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
plt.title('Social',fontsize= 18, y=-0.20)
plt.ylim(0,1.1)
plt.yticks([])
plt.xticks(fontsize=18)
ax.spines['left'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

B_labels.tight_layout
B_labels.savefig(FigureFolder + '/BoutType.png', dpi=300, bbox_inches='tight')





        
        
        
    