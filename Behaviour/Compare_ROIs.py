# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 13:37:22 2021

@author: Alizee Kastler
"""
lib_path = r'C:/Repos/Social_Pain/Libs'
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# -----------------------------------------------------------------------------

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
from scipy.stats import mannwhitneyu
import seaborn as sns
import pandas as pd

# Import local modules
import glob
import pylab as pl

# Set analysis folder and label for experiment/condition A
analysisFolder_A = base_path + r'/Analysis_Control_New'
conditionName_A = "Controls"

# Set analysis folder and label for experiment/condition B
analysisFolder_B = base_path + r'/Analysis_Heat_New'
conditionName_B = "Heat"

# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B]
conditionNames = [conditionName_A, conditionName_B]


# Summary Containers
NS_Cool_summary = []
NS_Hot_summary = []
NS_Noxious_summary = []
S_Cool_summary = []
S_Hot_summary = []
S_Noxious_summary = []


# Go through each condition (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):

    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder +'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data
    NS_Cool_ALL = np.zeros(numFiles)
    NS_Hot_ALL = np.zeros(numFiles)        
    NS_Noxious_ALL = np.zeros(numFiles)
    S_Cool_ALL = np.zeros(numFiles)
    S_Hot_ALL = np.zeros(numFiles)
    S_Noxious_ALL = np.zeros(numFiles)    
    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        NS_Cool = dataobject['NS_Cool']    
        NS_Hot = dataobject['NS_Hot']   
        NS_Noxious = dataobject['NS_Noxious']   
        S_Cool = dataobject['S_Cool']
        S_Hot = dataobject['S_Hot']   
        S_Noxious = dataobject['S_Noxious']   
       
        # Make an array with all summary stats
        NS_Cool_ALL[f] = NS_Cool
        NS_Hot_ALL[f] = NS_Hot
        NS_Noxious_ALL[f] = NS_Noxious
        S_Cool_ALL[f] = S_Cool
        S_Hot_ALL[f] = S_Hot
        S_Noxious_ALL[f] = S_Noxious
        
    Cool = mannwhitneyu(NS_Cool_ALL, S_Cool_ALL, use_continuity=True, alternative='two-sided')
    Hot = mannwhitneyu(NS_Hot_ALL, S_Hot_ALL, use_continuity=True, alternative='two-sided')
    Noxious = mannwhitneyu(NS_Noxious_ALL, S_Noxious_ALL, use_continuity=True, alternative='two-sided')
    print(Cool, Hot, Noxious)
    
    # Add to summary lists
    NS_Cool_summary.append(NS_Cool_ALL)
    NS_Hot_summary.append(NS_Hot_ALL)
    NS_Noxious_summary.append(NS_Noxious_ALL)
    
    S_Cool_summary.append(S_Cool_ALL)
    S_Hot_summary.append(S_Hot_ALL)
    S_Noxious_summary.append(S_Noxious_ALL)


#df= NS_Cool_summary.append([NS_Hot_summary, NS_Noxious_summary])     

# Summary plots
plt.figure()

# VPI
plt.title('Time Spent in each ROI')
series_list = []
for i, name in enumerate(conditionNames):
        Cool = pd.Series(NS_Cool_summary[i], name="NS_Cool" + '\n'+ name)
        Hot = pd.Series(NS_Hot_summary[i], name="NS_Hot" + '\n' + name)
        Noxious = pd.Series(NS_Noxious_summary[i], name="NS_Noxious" + '\n'+ name)
        s = pd.concat([Cool, Hot, Noxious], axis=1)
        series_list.append(s)

    
for i, name in enumerate(conditionNames):
        Cool = pd.Series(S_Cool_summary[i], name="S_Cool" + '\n'+ name)
        Hot = pd.Series(S_Hot_summary[i], name="S_Hot" + '\n' + name)
        Noxious = pd.Series(S_Noxious_summary[i], name="S_Noxious" '\n' + name)
        s = pd.concat([Cool, Hot, Noxious], axis=1)
        series_list.append(s)

df = pd.concat(series_list, axis=1)
sns.barplot(data=df, orient="v", saturation=0.1, color=[0.75,0.75,0.75,1], ci=95, capsize=0.05, errwidth=2)
sns.stripplot(data=df, orient="v", size=4, jitter=True, edgecolor="gray")


#FIN