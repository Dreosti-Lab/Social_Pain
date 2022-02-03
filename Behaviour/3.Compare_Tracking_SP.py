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
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from scipy import stats


# Specify Analysis folder
AnalysisFolder = base_path + '/Control_NewChamber/Analysis' 

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(AnalysisFolder+'/*.npz')

# Calculate how many files
numFiles = np.size(npzFiles, 0)

# Allocate space for summary data
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
numFreezes_NS_ALL = np.zeros(numFiles)
numFreezes_S_ALL = np.zeros(numFiles)
Binned_Freezes_NS_ALL = np.zeros((numFiles,9))
Binned_Freezes_S_ALL = np.zeros((numFiles,9))
Binned_DistanceT_NS_ALL = np.zeros((numFiles,10))
Binned_DistanceT_S_ALL = np.zeros((numFiles,10))
Percent_Moving_NS_ALL = np.zeros(numFiles)
Percent_Moving_S_ALL = np.zeros(numFiles)
Percent_Paused_NS_ALL = np.zeros(numFiles)
Percent_Paused_S_ALL = np.zeros(numFiles)
DistanceT_NS_ALL = np.zeros(numFiles)
DistanceT_S_ALL = np.zeros(numFiles)
Bouts_NS_ALL = np.zeros((0,9))
Bouts_S_ALL = np.zeros((0,9))
Pauses_NS_ALL = np.zeros((0,9))   
Pauses_S_ALL = np.zeros((0,9))
Freezes_S_ALL = np.zeros((0,4))
Freezes_NS_ALL = np.zeros((0,4))
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
    dataobject = np.load(filename)
    
    # Extract from the npz file 
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']
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
    Binned_DistanceT_NS = dataobject['Binned_DistanceT_NS']
    Binned_DistanceT_S = dataobject['Binned_DistanceT_S']
    OrtHist_NS_Cool = dataobject['OrtHist_NS_Cool']
    OrtHist_NS_Hot = dataobject['OrtHist_NS_Hot']
    OrtHist_NS_Noxious = dataobject['OrtHist_NS_Noxious']
    OrtHist_S_Cool = dataobject['OrtHist_S_Cool']
    OrtHist_S_Hot = dataobject['OrtHist_S_Hot']
    OrtHist_S_Noxious = dataobject['OrtHist_S_Noxious']
    Position_NS = dataobject['Position_NS']
    Position_S = dataobject['Position_S']
    
    # Make an array with all summary stats
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    numFreezes_NS_ALL[f] = numFreezes_NS
    numFreezes_S_ALL[f] = numFreezes_S
    Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
    Binned_Freezes_S_ALL[f,:] = Binned_Freezes_S
    Binned_DistanceT_NS_ALL[f,:] = Binned_DistanceT_NS
    Binned_DistanceT_S_ALL[f,:] = Binned_DistanceT_S
    Percent_Moving_NS_ALL[f] = Percent_Moving_NS
    Percent_Moving_S_ALL[f] = Percent_Moving_S
    Percent_Paused_NS_ALL[f] = Percent_Paused_NS
    Percent_Paused_S_ALL[f] = Percent_Paused_S
    Position_NS_ALL[f] = Position_NS
    Position_S_ALL[f] = Position_S
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
    Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
    Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S]) 
    Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
    Freezes_S_ALL = np.vstack([Freezes_S_ALL, Freezes_S])
    
     
s_DistanceT,pvalue_DistanceT = stats.ttest_rel(DistanceT_NS_ALL, DistanceT_S_ALL) 
# Plot Distance Travelled
plt.figure(figsize=(3,8), dpi=300)
plt.title('Distance Travelled n='+ format(numFiles) +'\n p-value:'+ format(pvalue_DistanceT),pad=10, fontsize= 20, y=-0.2)
plt.ylabel('Total Distance Travelled (mm)')
plt.ylim(0,12000)
#plt.ylim([0,50000])
s1 = pd.Series(DistanceT_NS_ALL, name='Non Social')
s2 = pd.Series(DistanceT_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue', 'steelblue'])
sns.stripplot(data=df,orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()  


s_BPS,pvalue_BPS = stats.ttest_rel(BPS_NS_ALL, BPS_S_ALL)    
#Plot BPS
plt.figure(figsize=(3,8), dpi=300)
plt.title('Bouts Per Second n=' + format(numFiles) + '\n p-value: '+ format(pvalue_BPS), pad=10, fontsize= 20, y=-0.2)
plt.ylabel('Number of Bouts Per Second (s)')
plt.ylim(0,2)
s1 = pd.Series(BPS_NS_ALL, name='Non Social')
s2 = pd.Series(BPS_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()
plt.show()

nBouts_NS = BPS_NS_ALL * (60000/120)
nBouts_S = BPS_S_ALL * (60000/120)

s_nBouts,pvalue_nBouts = stats.ttest_rel(nBouts_NS, nBouts_S) 
plt.figure(figsize=(3,8), dpi=300)
plt.title(' Total number of Bouts n=' + format(numFiles) + '\n p-value: '+ format(pvalue_nBouts), pad=10, fontsize= 20, y=-0.2)
plt.ylabel('Number of Bouts')
plt.ylim(0,650)
s1 = pd.Series(nBouts_NS, name='Non Social')
s2 = pd.Series(nBouts_S, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()
plt.show()

s_Freezes,pvalue_Freezes = stats.ttest_rel(numFreezes_NS_ALL, numFreezes_S_ALL) 
# Plot Short Freezes
plt.figure(figsize=(3,8), dpi=300)
plt.title('3s Freezes n='+ format(numFiles)+'\n p-value: ' + format(pvalue_Freezes), pad=10, fontsize= 20, y=-0.2)
plt.ylabel('Total Number of Freezes (>3s)')
plt.ylim(0,40)
s1 = pd.Series(numFreezes_NS_ALL, name='Non Social')
s2 = pd.Series(numFreezes_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()


# Plot Percent time Moving
s_Moving,pvalue_Moving = stats.ttest_rel(Percent_Moving_NS_ALL, Percent_Moving_S_ALL) 
plt.figure(figsize=(3,8), dpi=300)
plt.title('n='+ format(numFiles) +'\n p-value: ' + format(pvalue_Moving), pad=10, fontsize= 20, y=-0.2)
plt.ylabel('% Time Moving')
plt.ylim(0,100)
s1 = pd.Series(Percent_Moving_NS_ALL, name='Non Social')
s2 = pd.Series(Percent_Moving_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()

# Plot Percent time Moving
s_Pausing,pvalue_Pausing = stats.ttest_rel(Percent_Paused_NS_ALL, Percent_Paused_S_ALL) 
plt.figure(figsize=(3,8), dpi=300)
plt.title('n='+ format(numFiles) +'\n p-value: ' + format(pvalue_Pausing), pad=10, fontsize= 20, y=-0.2)
plt.ylabel('% Time Pausing')
plt.ylim(0,100)
s1 = pd.Series(Percent_Paused_NS_ALL, name='Non Social')
s2 = pd.Series(Percent_Paused_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()
  
# Scatterplot Position of Freezes
plt.figure(figsize=(8,2), dpi=300)
plt.title("Position of Freezes"+ '\n n='+ format(numFiles))  
plt.scatter(Freezes_NS_ALL[:,1],Freezes_NS_ALL[:,2],color= 'lightsteelblue', label ="Non_Social")
plt.scatter(Freezes_S_ALL[:,1],Freezes_S_ALL[:,2],color= 'steelblue', label = "Social")
plt.legend(bbox_to_anchor=(0.3, 1.6))
plt.show()

#Plot distribution of Bouts // NS vs S

fig = plt.figure(figsize=(48,18), dpi=300)
plt.suptitle("Distribution of Bouts"+ '\n n='+ format(numFiles), fontweight="bold", fontsize=64, y=1) 

ax = plt.subplot(221)
plt.hist2d(Bouts_NS_ALL[:,1], Bouts_NS_ALL[:,2], bins=10, cmap='Blues')
plt.title('Bout Starts', fontweight="bold", fontsize= 32, y=-0.25)
plt.xticks(fontsize=32)  
plt.yticks(fontsize=32) 
plt.colorbar()

ax = plt.subplot(223)
plt.hist2d(Bouts_NS_ALL[:,5], Bouts_NS_ALL[:,6],bins=10, cmap='Blues' )
plt.title('Bout Stops', fontweight="bold", fontsize= 32, y=-0.25)
plt.xticks(fontsize=32)  
plt.yticks(fontsize=32) 
plt.colorbar()

ax = plt.subplot(222)  
plt.hist2d(Bouts_S_ALL[:,1], Bouts_S_ALL[:,2],bins=10, cmap='Blues')
plt.title('Bout Starts', fontweight="bold", fontsize= 32, y=-0.25)
plt.xticks(fontsize=32)  
plt.yticks(fontsize=32) 
plt.colorbar()

ax = plt.subplot(224) 
plt.hist2d(Bouts_S_ALL[:,5], Bouts_S_ALL[:,6],bins=10, cmap='Blues' )
plt.title('Bout Stops', fontweight="bold", fontsize= 32, y=-0.25)
plt.xticks(fontsize=32)  
plt.yticks(fontsize=32)  
plt.colorbar()

fig.tight_layout(pad=2)



#Binned Freezes
mean_Freezes_NS = np.mean(Binned_Freezes_NS_ALL, axis=0)
sem_Freezes_NS = stats.sem(Binned_Freezes_NS_ALL)
std_Freezes_NS = np.std(Binned_Freezes_NS_ALL, axis=0)

mean_Freezes_S = np.mean(Binned_Freezes_S_ALL, axis=0)
sem_Freezes_S = stats.sem(Binned_Freezes_S_ALL)
std_Freezes_S = np.std(Binned_Freezes_S_ALL, axis=0)


# Plot Freezes_Time
plt.figure(figsize=(8,6), dpi=300)
plt.plot(mean_Freezes_NS, color = 'lightsteelblue', linewidth= 3)
plt.fill_between(np.arange(mean_Freezes_NS.shape[0]), mean_Freezes_NS + sem_Freezes_NS,
                 mean_Freezes_NS - sem_Freezes_NS,color= 'lightsteelblue',alpha=0.2)
#plt.errorbar(np.arange(mean_Freezes_NS.shape[0]), mean_Freezes_NS,std_Freezes_NS, color ='lightsteelblue', linewidth=1)
plt.plot(mean_Freezes_S, color = 'steelblue', linewidth= 3)
plt.fill_between(np.arange(mean_Freezes_S.shape[0]), mean_Freezes_S + sem_Freezes_S,
                 mean_Freezes_S - sem_Freezes_S,color= 'steelblue',alpha=0.2)
#plt.errorbar(np.arange(mean_Freezes_S.shape[0]), mean_Freezes_S,std_Freezes_S, color ='steelblue', linewidth=1)

plt.title('3s Freezes over time', fontweight="bold",fontsize= 18)
plt.legend(labels=('Non Social', 'Social'), loc='upper right', bbox_to_anchor=(0.2, 1.2))
plt.xticks(np.arange(0, 10, step= 1), ('1', '2','3','4','5','6','7', '8', '9', '10'))
plt.xlabel('minutes')
plt.ylabel('mean nb Freezes')
plt.ylim(0,2)

# Plot Distance Travelled Time
mean_DistanceT_NS = np.mean(Binned_DistanceT_NS_ALL, axis=0)
sem_DistanceT_NS = stats.sem(Binned_DistanceT_NS_ALL)
std_DistanceT_NS = np.std(Binned_DistanceT_NS_ALL, axis=0)

mean_DistanceT_S = np.mean(Binned_DistanceT_S_ALL, axis=0)
sem_DistanceT_S = stats.sem(Binned_DistanceT_S_ALL)
std_DistanceT_S = np.std(Binned_DistanceT_S_ALL, axis=0)

plt.figure(figsize=(8,6), dpi=300)
plt.plot(mean_DistanceT_NS, color = 'lightsteelblue', linewidth= 3)
plt.fill_between(np.arange(mean_DistanceT_NS.shape[0]), mean_DistanceT_NS + sem_DistanceT_NS,
                 mean_DistanceT_NS - sem_DistanceT_NS,color= 'lightsteelblue',alpha=0.2)
#plt.errorbar(np.arange(mean_DistanceT_NS.shape[0]), mean_DistanceT_NS,std_DistanceT_NS, color ='lightsteelblue', linewidth=1)
plt.plot(mean_DistanceT_S, color = 'steelblue', linewidth= 3)
plt.fill_between(np.arange(mean_DistanceT_S.shape[0]), mean_DistanceT_S + sem_DistanceT_S,
                 mean_DistanceT_S - sem_DistanceT_S,color= 'steelblue',alpha=0.2)
plt.title('Distance Travelled over time', fontweight="bold",fontsize= 12)

plt.legend(labels=('Non Social', 'Social'), loc='upper right', bbox_to_anchor=(0.2, 1.2))
plt.xticks(np.arange(0, 10, step= 1), ('1', '2','3','4','5','6','7', '8', '9', '10'))
plt.xlabel('minutes')
plt.ylabel('Distance Travelled (mm)')
plt.ylim(200,700)

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
plt.figure('Summary: Orientation Histograms')
plt.figure()

ax = plt.subplot(131, polar=True)
plt.title('Cool', fontweight="bold",fontsize= 25, y=-0.2)
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_Cool_ALL, Norm_OrtHist_NS_Cool_ALL[0])), color= 'lightsteelblue',  linewidth = 3)
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_Cool_ALL, Norm_OrtHist_S_Cool_ALL[0])), color = 'steelblue', linewidth = 3)

plt.legend(labels=('Non Social', 'Social'), loc='upper right', bbox_to_anchor=(0.2, 1.2))

ax = plt.subplot(132, polar=True)
plt.title('Hot', fontweight="bold", fontsize= 25, y=-0.2)
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_Hot_ALL, Norm_OrtHist_NS_Hot_ALL[0])),color='lightsteelblue', linewidth = 3)
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_Hot_ALL, Norm_OrtHist_S_Hot_ALL[0])), color= 'steelblue', linewidth = 3)

ax = plt.subplot(133, polar=True)
plt.title('Noxious', fontweight="bold",fontsize= 25, y=-0.2)
plt.plot(xAxis, np.hstack((Norm_OrtHist_NS_Noxious_ALL, Norm_OrtHist_NS_Noxious_ALL[0])),color='lightsteelblue', linewidth = 3)
plt.plot(xAxis, np.hstack((Norm_OrtHist_S_Noxious_ALL, Norm_OrtHist_S_Noxious_ALL[0])), color= 'steelblue', linewidth = 3)


normPosition_NS = pd.DataFrame(data = Position_NS_ALL*100/numFiles, columns = ["Cool","Hot","Noxious"])
normPosition_NS['condition']='Non Social'

normPosition_S = pd.DataFrame(data = Position_S_ALL*100/numFiles, columns = ["Cool","Hot","Noxious"])
normPosition_S['condition']='Social'

#stacked_Histogram
Position = normPosition_NS.append([normPosition_S])
total = Position.groupby('condition')['Cool','Hot','Noxious'].sum().reset_index()
total['CoolHot'] = total['Cool'] + total['Hot']
total['Total'] = total['Cool']+ total['Hot'] + total['Noxious']

plt.figure(figsize=(6,3), dpi=300)
sns.set(style="white", font_scale=1.5)
sns.barplot(x='Total', y="condition" , data=total, color='darkorange')
sns.barplot(x="CoolHot", y="condition", data=total, estimator=sum, ci=None,  color='purple')
sns.barplot(x="Cool",y= "condition", data=total, estimator=sum, ci=None,  color='midnightblue')
plt.xlabel('Proportion of Frames')
plt.ylabel(None)
sns.despine()
#setLabels
Cool = mpatches.Patch(color= 'midnightblue', label = 'cool')
Hot = mpatches.Patch(color= 'purple', label = 'hot')
Noxious = mpatches.Patch(color= 'darkorange', label = 'noxious')
plt.legend(handles=[Noxious, Hot, Cool], bbox_to_anchor=(1, 1))
plt.show()


        
        
        
        
    