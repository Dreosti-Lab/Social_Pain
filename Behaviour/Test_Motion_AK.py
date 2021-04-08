#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 19:00:40 2021

@author: alizeekastler
Bouts
"""                        
# Set Library Path - Social_Pain Repos
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
base_path = r'/Users/alizeekastler/Desktop'

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



# Specify Analysis folder
AnalysisFolder = base_path + '/Experiment_8/Analysis'

# Find all the npz files saved for each group and fish with all the information
npzFiles = AnalysisFolder+'/*.npz'

# Calculate how many files
numFiles = np.size(npzFiles, 0)

# Allocate space for summary data
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
DistanceT_NS_ALL = np.zeros(numFiles)
DistanceT_S_ALL = np.zeros(numFiles)
Freezes_NS_ALL = np.zeros(numFiles)
Freezes_S_ALL = np.zeros(numFiles)
Bouts_NS_ALL = np.zeros((0,9))
Bouts_S_ALL = np.zeros((0,9))
Pauses_NS_ALL = np.zeros((0,9))   
Pauses_S_ALL = np.zeros((0,9))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file 
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    DistanceT_NS = dataobject['Distance_NS']   
    DistanceT_S = dataobject['Distance_S']   
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']
    Pauses_NS = dataobject['Pauses_NS']   
    Pauses_S = dataobject['Pauses_S']
    Freezes_NS = dataobject['Freezes_NS']
    Freezes_S = dataobject['Freezes_S']
    
    # Make an array with all summary stats
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    DistanceT_NS_ALL[f] = DistanceT_NS
    DistanceT_S_ALL[f] = DistanceT_S
    Freezes_NS_ALL[f] = Freezes_NS
    Freezes_S_ALL[f] = Freezes_S

    # Concat all Pauses/Bouts
    Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
    Bouts_S_ALL = np.vstack([Bouts_S_ALL, Bouts_S])
    Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
    Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S])    
     
 
#Plot BPS
plt.subplot()
plt.title('Bouts Per Second')
s1 = pd.Series(BPS_NS, name='NS')
s2 = pd.Series(BPS_S, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['Steelblue', 'plum'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray")
sns.despine()

# Plot Distance Traveled
plt.subplot()
plt.title('Distance Traveled')
s1 = pd.Series(DistanceT_NS, name='NS')
s2 = pd.Series(DistanceT_S, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['Steelblue', 'plum'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray")
sns.despine()  
        
# Plot Freezes
plt.subplot()
plt.title('Freezes')
s1 = pd.Series(Freezes_NS, name='NS')
s2 = pd.Series(Freezes_S, name='S')
df = pd.concat([s1,s2], axis=1)
sns.barplot(data=df, ci='sd',  palette=['Steelblue', 'plum'])
sns.stripplot(data=df, orient="v", color= 'dimgrey',size=4, jitter=False, edgecolor="gray")
sns.despine()
    


#FIN




        
        
        
        
        
        
    