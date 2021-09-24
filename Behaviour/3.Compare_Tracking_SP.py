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
import pandas as pd
import seaborn as sns
from scipy import stats


# Specify Analysis folder
AnalysisFolder = base_path + '/Analysis_Control' 

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(AnalysisFolder+'/*.npz')

# Calculate how many files
numFiles = np.size(npzFiles, 0)

# Allocate space for summary data
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
Long_Freezes_NS_ALL = np.zeros(numFiles)
Short_Freezes_NS_ALL = np.zeros(numFiles)
Long_Freezes_S_ALL = np.zeros(numFiles)
Short_Freezes_S_ALL = np.zeros(numFiles)
Short_Freezes_X_NS_ALL = np.zeros(numFiles)
Short_Freezes_Y_NS_ALL = np.zeros(numFiles)
Short_Freezes_X_S_ALL = np.zeros(numFiles)
Short_Freezes_Y_S_ALL = np.zeros(numFiles)
DistanceT_NS_ALL = np.zeros(numFiles)
DistanceT_S_ALL = np.zeros(numFiles)
Bouts_X_NS_ALL = np.zeros(numFiles)
Bouts_Y_NS_ALL = np.zeros(numFiles)
Bouts_NS_ALL = np.zeros((0,9))
Bouts_S_ALL = np.zeros((0,9))
Pauses_NS_ALL = np.zeros((0,9))   
Pauses_S_ALL = np.zeros((0,9))
OrtHist_NS_Cool_ALL = np.zeros((numFiles,36))
OrtHist_NS_Hot_ALL = np.zeros((numFiles,36))
OrtHist_NS_Noxious_ALL = np.zeros((numFiles,36))
OrtHist_S_Cool_ALL = np.zeros((numFiles,36))
OrtHist_S_Hot_ALL = np.zeros((numFiles,36))
OrtHist_S_Noxious_ALL = np.zeros((numFiles,36))
# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file 
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']
    Bouts_X_NS = dataobject['Bouts_X_NS']
    Bouts_Y_NS = dataobject['Bouts_Y_NS']
    Pauses_NS = dataobject['Pauses_NS']   
    Pauses_S = dataobject['Pauses_S']
    Long_Freezes_NS = dataobject['Long_Freezes_NS']
    Short_Freezes_NS = dataobject['Short_Freezes_NS']
    Long_Freezes_S = dataobject['Long_Freezes_S']
    Short_Freezes_S = dataobject['Short_Freezes_S']
    Short_Freezes_X_NS = dataobject['Short_Freezes_X_NS']
    Short_Freezes_X_S = dataobject['Short_Freezes_X_S']
    Short_Freezes_Y_S = dataobject['Short_Freezes_Y_S']
    Short_Freezes_Y_NS = dataobject['Short_Freezes_Y_NS']
    DistanceT_NS = dataobject['DistanceT_NS']
    DistanceT_S = dataobject['DistanceT_S']
    OrtHist_NS_Cool = dataobject['OrtHist_NS_Cool']
    OrtHist_NS_Hot = dataobject['OrtHist_NS_Hot']
    OrtHist_NS_Noxious = dataobject['OrtHist_NS_Noxious']
    OrtHist_S_Cool = dataobject['OrtHist_S_Cool']
    OrtHist_S_Hot = dataobject['OrtHist_S_Hot']
    OrtHist_S_Noxious = dataobject['OrtHist_S_Noxious']
    
    # Make an array with all summary stats
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    Long_Freezes_NS_ALL[f] = Long_Freezes_NS
    Short_Freezes_NS_ALL[f] = Short_Freezes_NS
    Long_Freezes_S_ALL[f] = Long_Freezes_S
    Short_Freezes_S_ALL[f] = Short_Freezes_S
    Short_Freezes_X_NS_ALL[f] = Short_Freezes_X_NS
    Short_Freezes_X_S_ALL[f] = Short_Freezes_X_S
    Short_Freezes_Y_NS_ALL[f] = Short_Freezes_Y_NS
    Short_Freezes_Y_S_ALL[f] = Short_Freezes_Y_S
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
plt.ylim(0,7)
s1 = pd.Series(BPS_NS_ALL, name='Non Social')
s2 = pd.Series(BPS_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()
plt.show()

s_Freezes,pvalue_Freezes = stats.ttest_rel(Long_Freezes_NS_ALL, Long_Freezes_S_ALL) 
# Plot Long Freezes
plt.figure(figsize=(3,8), dpi=300)
plt.title('10s Freezes n=' + format(numFiles)+'\n p-value: ' + format(pvalue_Freezes),pad=10, fontsize= 20, y=-0.2)
plt.ylabel('Total Number of Freezes (>10s)')
plt.ylim(0,50)
s1 = pd.Series(Long_Freezes_NS_ALL, name='Non Social')
s2 = pd.Series(Long_Freezes_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()


s_Freezes,pvalue_Freezes = stats.ttest_rel(Short_Freezes_NS_ALL, Short_Freezes_S_ALL) 
# Plot Short Freezes
plt.figure(figsize=(3,8), dpi=300)
plt.title('3s Freezes n='+ format(numFiles)+'\n p-value: ' + format(pvalue_Freezes), pad=10, fontsize= 20, y=-0.2)
plt.ylabel('Total Number of Freezes (>3s)')
plt.ylim(0,50)
s1 = pd.Series(Short_Freezes_NS_ALL, name='Non Social')
s2 = pd.Series(Short_Freezes_S_ALL, name='Social')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['lightsteelblue','steelblue'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=6, jitter=True, edgecolor="gray")
sns.despine()
 

plt.figure()
plt.subplot
plt.scatter(Short_Freezes_X_NS_ALL,Short_Freezes_Y_NS_ALL, color= 'lightsteelblue')
plt.scatter(Short_Freezes_X_S_ALL,Short_Freezes_Y_S_ALL, color= 'steelblue')

# plt.figure() 
# plt.subplot           
# plt.hist2d(Bouts_X_NS_ALL, Bouts_Y_NS_ALL, bins=14, cmap='Blues')
# plt.colorbar()

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









        
        
        
        
        
        
    