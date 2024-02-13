#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 12:10:53 2023

@author: alizeekastler
"""

# -----------------------------------------------------------------------------
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'
#base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'


import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import  ttest_ind, stats
import statsmodels.stats.multicomp as mc



FigureFolder = base_path + '/MustardOil/Figures'


# Define condition data in a structured manner
conditions = [
    {
        'folder': base_path + r'/Baseline/Analysis',
        'name': "Baseline",
        'color': '#aaa4c8',
    },
    # {
    #     'folder': base_path + r'/Heat_36°C/Analysis',
    #     'name': "Heat",
    #     'color': '#452775',
    # },
    {
          'folder': base_path + r'/DMSO/Analysis',
          'name': "DMSO",
          'color': 'plum',
    },
    {
          'folder': base_path + r'/AITC_100uM/Analysis',
          'name': "100uM",
          'color': 'lightcoral',
    },
    {
          'folder': base_path + r'/AITC_200uM/Analysis',
          'name': "200uM",
          'color': '#d85c1a',
    }
]


# Summary Containers
BPS_NS_summary = []
DistanceT_NS_summary = []
avgdistPerBout_NS_summary = []
avgSpeedPerBout_NS_summary = []
avgBout_interval_NS_summary = []
Freezes_NS_summary = []
numFreezes_NS_summary = []
Percent_Moving_NS_summary = []
Percent_Paused_NS_summary = []
Binned_Bouts_NS_summary = []
Binned_Freezes_NS_summary = []
Binned_PTF_NS_summary = []
Binned_PTM_NS_summary = []
Bouts_NS_summary = []
Turns_NS_summary = []
FSwim_NS_summary = []

# Go through each condition (analysis folder)
for i  in range(0,len(conditions)):

    # Find all the npz files saved for each group and fish with all the information
    npzFiles = glob.glob(conditions[i]['folder']+'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data       
    BPS_NS_ALL = np.zeros(numFiles)
    DistanceT_NS_ALL = np.zeros(numFiles) 
    avgdistPerBout_NS_ALL = np.zeros(numFiles)
    avgSpeedPerBout_NS_ALL = np.zeros(numFiles)
    avgBout_interval_NS_ALL = np.zeros(numFiles)
    Freezes_NS_ALL = np.zeros((0,4))
    numFreezes_NS_ALL = np.zeros(numFiles)
    Percent_Moving_NS_ALL = np.zeros(numFiles)
    Percent_Paused_NS_ALL = np.zeros(numFiles)
    Binned_Bouts_NS_ALL = np.zeros((numFiles,15))
    Binned_Freezes_NS_ALL = np.zeros((numFiles,15))
    Binned_PTF_NS_ALL = np.zeros((numFiles,15))
    Binned_PTM_NS_ALL = np.zeros((numFiles,15))
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
        avgdistPerBout_NS = dataobject['avgdistPerBout_NS']
        avgSpeedPerBout_NS = dataobject['avgSpeedPerBout_NS']
        avgBout_interval_NS = dataobject['avgBout_interval_NS']
        Freezes_NS = dataobject['Freezes_NS'] 
        numFreezes_NS = dataobject['numFreezes_NS']
        Percent_Moving_NS = dataobject['Percent_Moving_NS'] 
        Percent_Paused_NS = dataobject['Percent_Paused_NS']
        Binned_Bouts_NS = dataobject['Binned_Bouts_NS']
        Binned_Freezes_NS = dataobject['Binned_Freezes_NS']
        Binned_PTF_NS = dataobject['Binned_PTF_NS']
        Binned_PTM_NS = dataobject['Binned_PTM_NS']
        Bouts_NS = dataobject['Bouts_NS']  
        Turns_NS = dataobject['Turns_NS']
        FSwim_NS= dataobject['FSwim_NS']
        
        
        # Make an array with all summary stats
        BPS_NS_ALL[f] = BPS_NS
        DistanceT_NS_ALL[f] = DistanceT_NS
        avgdistPerBout_NS_ALL[f] = avgdistPerBout_NS
        avgSpeedPerBout_NS_ALL[f]= avgSpeedPerBout_NS
        avgBout_interval_NS_ALL[f]= avgBout_interval_NS
        Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
        numFreezes_NS_ALL[f] = numFreezes_NS
        Percent_Moving_NS_ALL[f] = Percent_Moving_NS
        Percent_Paused_NS_ALL[f] = Percent_Paused_NS
        Binned_Bouts_NS_ALL[f,:] = Binned_Bouts_NS
        Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
        Binned_PTF_NS_ALL[f,:] = Binned_PTF_NS
        Binned_PTM_NS_ALL[f,:] = Binned_PTM_NS
        Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
        Turns_NS_ALL[f] = Turns_NS
        FSwim_NS_ALL[f] = FSwim_NS
    
    # Add to summary lists
    BPS_NS_summary.append(BPS_NS_ALL)
    DistanceT_NS_summary.append(DistanceT_NS_ALL)
    avgdistPerBout_NS_summary.append(avgdistPerBout_NS_ALL)
    avgSpeedPerBout_NS_summary.append(avgSpeedPerBout_NS_ALL)
    avgBout_interval_NS_summary.append(avgBout_interval_NS_ALL)
    Freezes_NS_summary.append(Freezes_NS_ALL)
    numFreezes_NS_summary.append(numFreezes_NS_ALL)
    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Paused_NS_summary.append(Percent_Moving_NS_ALL)
    Binned_Bouts_NS_summary.append(Binned_Bouts_NS_ALL) 
    Binned_Freezes_NS_summary.append(Binned_Freezes_NS_ALL)
    Binned_PTF_NS_summary.append(Binned_PTF_NS_ALL)
    Binned_PTM_NS_summary.append(Binned_PTM_NS_ALL)
    Bouts_NS_summary.append(Bouts_NS_ALL)
    Turns_NS_summary.append(Turns_NS_ALL)
    FSwim_NS_summary.append(FSwim_NS_ALL)


def box_strip_plot(conditions, variable, title, ylabel,ylim,FigureFolder):
    
    x = plt.figure(figsize =(6,10), dpi=300)
    plt.title(title, fontsize=36, y=1.1, fontname="Arial")
    series_list = []
    
    # Data processing and plotting
    for i in range (0,len(conditions)):
        s = pd.Series(variable[i], name= conditions[i]['name'])
        series_list.append(s)
       
    x_data = pd.concat(series_list, axis=1) 
    
    plt.ylabel(ylabel, fontsize = 30, fontname='Arial')
    plt.ylim(ylim)
    plt.yticks(fontsize=26, fontname="Arial")
    colors = [condition['color'] for condition in conditions]
    
    ax=sns.boxplot(data=x_data,color = 'whitesmoke', linewidth=3, showfliers=False)
    ax=sns.stripplot(data= x_data, orient="v", palette = colors,size=8, jitter=True, edgecolor="gray")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize = 30, fontname='Arial')
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    sns.despine() 
    
    # Perform the significance test (e.g., Mann-Whitney U Test or Independent t-test)
    statistic, p_value = ttest_ind(variable[0], variable[1])  # Change this line as needed
    
    # # Perform ANOVA
    # # Impute missing values with the mean of each column
    # x_data = x_data.fillna(x_data.mean())
    
    # f_statistic, p_value = stats.f_oneway(*[x_data[col] for col in x_data.columns])
    # # Perform post hoc test (Tukey's HSD)
    # posthoc = mc.MultiComparison(pd.melt(x_data)['value'], pd.melt(x_data)['variable'])
    # posthoc_result = posthoc.tukeyhsd()
    # print(posthoc_result)
    
    # #Define the significance level (e.g., 0.05 for a 5% significance level)
    # significance_level = 0.05

    # # Add asterisks to indicate significance
    # if p_value < significance_level:
    #     significance_label = "*\n" + "_" * 10
    #     ax.text(0.5, 1.03, significance_label, ha='center', va='center', transform=ax.transAxes, fontsize=24, fontweight='bold', color='black')
   
    x.savefig(FigureFolder + '/'+ title+'.png', dpi=300, bbox_inches='tight')    
        

#Plot number of Bouts per second
box_strip_plot(conditions, BPS_NS_summary, 'Bouts per Second', 'nb of BPS',(0,3), FigureFolder)
#Plot distance Travelled
box_strip_plot(conditions, DistanceT_NS_summary, 'Distance Travelled', 'total distance (mm)',(0,20000), FigureFolder)
#Plot distance Travelled per Bout
box_strip_plot(conditions, avgdistPerBout_NS_summary, 'Distance Travelled per Bout', 'total distance (mm)',(0,15) , FigureFolder)
#Plot Time Speed
box_strip_plot(conditions, avgSpeedPerBout_NS_summary, 'Bout Speed', 'average speed (mm/s)',(0,15), FigureFolder)
#Plot Bout Interval
box_strip_plot(conditions, avgBout_interval_NS_summary, 'Inter-Bout Interval', 'interval (s)',(0,4), FigureFolder)
#Plot number of Freezes
box_strip_plot(conditions, numFreezes_NS_summary, 'Freezes', 'total nb of freezes',(0,100), FigureFolder)
#Plot Time Moving
box_strip_plot(conditions, Percent_Moving_NS_summary, 'Moving', '% Time Moving',(0,100), FigureFolder)
#Plot Time Pausing
box_strip_plot(conditions, Percent_Moving_NS_summary, 'Pausing', '% Time Pausing',(0,100), FigureFolder)



# Scatterplot dist Angles 
series_list = [
    pd.Series(Bouts_NS_summary[i][:, 9], Bouts_NS_summary[i][:, 10], name=condition['name'])
    for i, condition in enumerate(conditions)
]

# Create a dictionary to store DataFrames for each condition
condition_data = {}

# Iterate through each series and create DataFrames
for series in series_list:
    condition_name = series.name
    df = pd.DataFrame({f'{condition_name}_Pos': series.index, f'{condition_name}_Ort': series.values})
    condition_data[condition_name] = df

# Iterate through unique condition names and create scatter plots
for condition_name, df in condition_data.items():
    condition_color = [condition['color'] for condition in conditions if condition['name'] == condition_name][0]

    fig, ax = plt.subplots(figsize=(6, 8))
    #ax=plt.figure(figsize=(6, 8))
    #plt.scatter(x, y, alpha=0.08)
    ax.scatter(df[f'{condition_name}_Ort'], df[f'{condition_name}_Pos'], alpha=0.08, c=condition_color)
    ax.set_title(f'{condition_name}', fontsize=38, y=1.05, fontname='Arial')
    ax.set_xlabel('∆ Orientation (°)', fontsize=38, fontname='Arial')
    ax.set_ylabel('∆ Position (mm)', fontsize = 38, fontname='Arial')
    ax.set_xlim(-80, 80)
    ax.set_ylim(0, 8)
    ax.tick_params(axis='both', which='major', labelsize=32)
    ax.set_xticks(np.arange(-80, 81, step=40))
    ax.set_yticks(np.arange(0, 9, step=2))
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    sns.despine()
    
    # Save the plot
    plt.savefig(FigureFolder + f'/BoutAngles_{condition_name}.png', dpi=300, bbox_inches='tight')


# Scatterplot dist Angles 

series_list = [
    pd.Series(Turns_NS_summary[i], FSwim_NS_summary[i], name=condition['name'])
    for i, condition in enumerate(conditions)
]

# Create a dictionary to store DataFrames for each condition
condition_data = {}

# Iterate through each series and create DataFrames
for series in series_list:
    condition_name = series.name
    df = pd.DataFrame({f'Forward': series.index,f'Turn': series.values})
    condition_data[condition_name] = df

# Iterate through unique condition names and create scatter plots
for condition_name, df in condition_data.items():
    condition_color = [condition['color'] for condition in conditions if condition['name'] == condition_name][0]
    
    plt.figure(figsize=(4, 10))
    
    ax=sns.boxplot(data = df,color = 'whitesmoke', linewidth=3, showfliers=False)
    ax=sns.stripplot(data= df, orient="v", color = condition_color,size=8, jitter=True, edgecolor="gray")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, fontsize = 32)
    plt.title(f'{condition_name}', fontsize=38, y=1.05)
    plt.ylabel('Proportion of Bouts', fontsize = 38)
    plt.ylim(0,1)
    plt.yticks(fontsize=32)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    sns.despine()    


    # Save the plot
    plt.savefig(FigureFolder + f'/BoutTypes_{condition_name}.png', dpi=300, bbox_inches='tight')


# PTM vs PTF

# Iterate through conditions and plot
for i, condition in enumerate (conditions):
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    # Plot PTM
    m = np.nanmean(Binned_PTM_NS_summary[i], 0)
    std = np.nanstd(Binned_PTM_NS_summary[i], 0)
    valid = ~np.isnan(Binned_PTM_NS_summary[i])
    n = np.sum(valid, 0)
    se = std / np.sqrt(n - 1)

    ax.plot(m, color=condition['color'], linewidth=4, marker='o', markersize=10, label=f'% Moving ({condition["name"]})')
    ax.plot(m + se, color=condition['color'], linewidth=1, alpha = 0.5)
    ax.plot(m - se, color=condition['color'], linewidth=1, alpha=0.5)
    ax.fill_between(np.arange(15), m - se, m + se, color=condition['color'], alpha=0.2)
    
    # Plot PTF
    m = np.nanmean(Binned_PTF_NS_summary[i], 0)
    std = np.nanstd(Binned_PTF_NS_summary[i], 0)
    valid = ~np.isnan(Binned_PTF_NS_summary[i])
    n = np.sum(valid, 0)
    se = std / np.sqrt(n - 1)

    ax.plot(m, color=condition['color'], linewidth=4, marker='^', markersize=10, label=f'% Freezing ({condition["name"]})')
    ax.plot(m + se, color=condition['color'], linewidth=1, alpha=0.5)
    ax.plot(m - se, color=condition['color'], linewidth=1, alpha=0.5)
    ax.fill_between(np.arange(15), m - se, m + se, color=condition['color'], alpha=0.2)
    
    # Set labels and legend
    ax.legend(loc="upper right", fontsize=24)
    ax.set_xlabel('Minutes', fontsize=24, fontname= 'Arial')
    ax.set_ylabel('Percentage', fontsize=24, fontname = 'Arial')
    ax.set_ylim(-1, 90)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xticks(np.arange(0, 15, step=2))
    ax.set_xticklabels(['0','2', '4', '6', '8', '10', '12', '14'], fontsize=24)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    
    # Save the plot
    plt.savefig(FigureFolder + f'/A to P_'+condition['name']+'.png', dpi=300, bbox_inches='tight')



