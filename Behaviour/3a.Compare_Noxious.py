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


FigureFolder = base_path + '/Figure_Nox/test'

# Set analysis folder and label for experiment/condition A
analysisFolder_A = base_path + r'/Baseline/Analysis' 
conditionName_A = "Bas"

# Set analysis folder and label for experiment/condition A
analysisFolder_B = base_path + r'/Heat_36°C/Analysis' 
conditionName_B = "NoxHeat"

# Set analysis folder and label for experiment/condition B
analysisFolder_C = base_path + r'/AITC_50uM/Analysis' 
conditionName_C = "50uM"

# Set analysis folder and label for experiment/condition B
analysisFolder_D = base_path + r'/AITC_100uM/Analysis' 
conditionName_D = "100uM"

# Set analysis folder and label for experiment/condition B
analysisFolder_E = base_path + r'/AITc_200uM/Analysis' 
conditionName_E = "200uM"

# Assemble lists
analysisFolders = [analysisFolder_A, analysisFolder_B, analysisFolder_C, analysisFolder_D, analysisFolder_E]
conditionNames = [conditionName_A, conditionName_B, conditionName_C, conditionName_D, conditionName_E]

# Summary Containers
BPS_NS_summary = []
DistanceT_NS_summary = []
Freezes_NS_summary = []
numFreezes_NS_summary = []
Percent_Moving_NS_summary = []
Percent_Paused_NS_summary = []
avgdistPerBout_NS_summary = []
Binned_Bouts_NS_summary = []
Binned_Freezes_NS_summary = []
Binned_PTF_summary = []
Binned_PTM_summary = []
Bouts_NS_summary = []
Turns_NS_summary = []
FSwim_NS_summary = []

# Go through each condition (analysis folder)
for i, analysisFolder in enumerate(analysisFolders):
    
    # Freeze time threshold
    freeze_threshold = 200 # more than 3 seconds

    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(analysisFolder+'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data       
    BPS_NS_ALL = np.zeros(numFiles)
    DistanceT_NS_ALL = np.zeros(numFiles)    
    Freezes_NS_ALL = np.zeros((0,4))
    numFreezes_NS_ALL = np.zeros(numFiles)
    Percent_Moving_NS_ALL = np.zeros(numFiles)
    Percent_Paused_NS_ALL = np.zeros(numFiles)
    avgdistPerBout_NS_ALL = np.zeros(numFiles)
    Binned_Bouts_NS_ALL = np.zeros((numFiles,15))
    Binned_Freezes_NS_ALL = np.zeros((numFiles,15))
    #Binned_PTF_ALL = np.zeros((numFiles,15))
    #Binned_PTM_ALL = np.zeros((numFiles,15))
    Bouts_NS_ALL = np.zeros((0,11))
    Turns_NS_ALL = np.zeros(numFiles)
    FSwim_NS_ALL = np.zeros(numFiles)
    
    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):
    
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file   
        BPS_NS = dataobject['BPS_NS'] 
        DistanceT_NS = dataobject['DistanceT_NS']   
        Freezes_NS = dataobject['Freezes_NS'] 
        numFreezes_NS = dataobject['numFreezes_NS']
        Percent_Moving_NS = dataobject['Percent_Moving_NS'] 
        Percent_Paused_NS = dataobject['Percent_Paused_NS']
        avgdistPerBout_NS = dataobject['avgdistPerBout_NS']
        Binned_Bouts_NS = dataobject['Binned_Bouts_NS']
        Binned_Freezes_NS = dataobject['Binned_Freezes_NS']
        #Binned_PTF = dataobject['Binned_PTF']
        #Binned_PTM = dataobject['Binned_PTM']
        Bouts_NS = dataobject['Bouts_NS']  
        Turns_NS = dataobject['Turns_NS']
        FSwim_NS= dataobject['FSwim_NS']
        
        
        # Make an array with all summary stats
        BPS_NS_ALL[f] = BPS_NS
        DistanceT_NS_ALL[f] = DistanceT_NS
        Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
        numFreezes_NS_ALL[f] = numFreezes_NS
        Percent_Moving_NS_ALL[f] = Percent_Moving_NS
        Percent_Paused_NS_ALL[f] = Percent_Paused_NS
        avgdistPerBout_NS_ALL[f] = avgdistPerBout_NS
        Binned_Bouts_NS_ALL[f,:] = Binned_Bouts_NS
        Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
        #Binned_PTF_ALL[f,:] = Binned_PTF
        #Binned_PTM_ALL[f,:] = Binned_PTM
        Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
        Turns_NS_ALL[f] = Turns_NS
        FSwim_NS_ALL[f] = FSwim_NS
    
    # Add to summary lists
    BPS_NS_summary.append(BPS_NS_ALL)
    DistanceT_NS_summary.append(DistanceT_NS_ALL)
    Freezes_NS_summary.append(Freezes_NS_ALL)
    numFreezes_NS_summary.append(numFreezes_NS_ALL)
    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Paused_NS_summary.append(Percent_Moving_NS_ALL)
    avgdistPerBout_NS_summary.append(avgdistPerBout_NS_ALL)
    Binned_Bouts_NS_summary.append(Binned_Bouts_NS_ALL) 
    Binned_Freezes_NS_summary.append(Binned_Freezes_NS_ALL) 
    #Binned_PTF_summary.append(Binned_PTF_ALL) 
    #Binned_PTM_summary.append(Binned_PTM_ALL)
    Bouts_NS_summary.append(Bouts_NS_ALL)
    Turns_NS_summary.append(Turns_NS_ALL)
    FSwim_NS_summary.append(FSwim_NS_ALL)
# Summary plots


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


m = np.nanmean(Binned_Bouts_NS_summary[0], 0)
std = np.nanstd(Binned_Bouts_NS_summary[0], 0)
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



# Scatterplot dist Angles 
distAngles = plt.figure(figsize =(12,14), dpi=300)

series_list = []
for i, name in enumerate(conditionNames):
    s = pd.Series(Bouts_NS_summary[i][:,9],Bouts_NS_summary[i][:,10],name= name)
    series_list.append(s)

# Create an empty DataFrame to store the final data
Angles_data = pd.DataFrame()

# Iterate through each series in the list
for series in series_list:
    # Extract the condition name from the series name
    condition_name = series.name
    
    # Create a DataFrame with the data from the series
    df = pd.DataFrame({f'{condition_name}_Pos': series.index, f'{condition_name}_Ort': series.values})
    
    # Add the DataFrame to the final DataFrame
    Angles_data = pd.concat([Angles_data, df], axis=1)

# Get a list of unique condition names (assuming they are in column names)    
condition_names = set(column.split('_')[0] for column in Angles_data.columns)

# Iterate through the unique condition names and create scatter plots
for condition_name in condition_names:
    # Select the columns for the current condition
    columns_for_condition = [column for column in Angles_data.columns if condition_name in column]
    
    
    # Create a scatter plot
    angle= plt.figure(figsize=(6, 8))
    plt.scatter(Angles_data[columns_for_condition[1]], Angles_data[columns_for_condition[0]], alpha=0.08)
    plt.title(f'distAngles{condition_name}')
    plt.xlabel('∆ Orientation (deg)', fontsize=14)
    plt.ylabel('∆ Position(mm)')
    plt.xlim(-80, 80)
    plt.ylim(0,4)

    angle.savefig(FigureFolder + f'/{condition_name}.png', dpi=300, bbox_inches='tight')



B_labels = plt.figure(figsize =(6,8), dpi=300)

series_list = []
for i, name in enumerate(conditionNames):
    s1 = pd.Series(Turns_NS_summary[i],name = name + '_Turn')
    s2 = pd.Series(FSwim_NS_summary[i],name = name + '_Forward')
    s = pd.concat([s1,s2], axis=1)
    series_list.append(s)

B_labels_data = pd.concat(series_list, axis=1)    


# Get a list of unique condition names (assuming they are in column names)    
condition_names = set(column.split('_')[0] for column in B_labels_data.columns)

# Iterate through the unique condition names and create scatter plots
for condition_name in condition_names:
    # Select the columns for the current condition
    columns_for_condition = [column for column in B_labels_data.columns if condition_name in column]
    
    labels= plt.figure(figsize=(6, 8))
    sns.swarmplot(data=B_labels_data[columns_for_condition],color='lightsteelblue',zorder=1)
    sns.pointplot(data=B_labels_data[columns_for_condition],estimator=np.median,capsize=0.1, join=False, zorder=100, color='dimgrey')
    plt.ylim(0,1.1)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    sns.despine()

    labels.savefig(FigureFolder + f'/B_lables_{condition_name}.png', dpi=300, bbox_inches='tight')

