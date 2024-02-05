#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 23:02:31 2023

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
from sklearn.preprocessing import StandardScaler
from scipy.stats import mannwhitneyu



AnalysisFolder = base_path + '/Gradient_Social/Analysis'
FigureFolder = base_path + '/Gradient_Social/Figures'

df = pd.read_csv(base_path + '/Gradient_Social/fish_clusters.csv')

# Calculate how many files
npzFiles = glob.glob(AnalysisFolder+'/*.npz')
numFiles = np.size(npzFiles, 0)

# Allocate space for summary data       
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
DistanceT_NS_ALL = np.zeros(numFiles) 
DistanceT_S_ALL = np.zeros(numFiles) 
avgPosition_NS_ALL = np.zeros(numFiles)
avgPosition_S_ALL = np.zeros(numFiles)
avgdistPerBout_NS_ALL = np.zeros(numFiles)
avgdistPerBout_S_ALL = np.zeros(numFiles)
avgSpeedPerBout_NS_ALL = np.zeros(numFiles)
avgSpeedPerBout_S_ALL = np.zeros(numFiles)
avgBout_interval_NS_ALL = np.zeros(numFiles)
avgBout_interval_S_ALL = np.zeros(numFiles)
numFreezes_NS_ALL = np.zeros(numFiles)
numFreezes_S_ALL = np.zeros(numFiles)
Percent_Moving_NS_ALL = np.zeros(numFiles)
Percent_Moving_S_ALL = np.zeros(numFiles)
Percent_Paused_NS_ALL = np.zeros(numFiles)
Percent_Paused_S_ALL = np.zeros(numFiles)
Binned_PTF_NS_ALL = np.zeros((numFiles,15))
Binned_PTF_S_ALL = np.zeros((numFiles,15))
Binned_PTM_NS_ALL = np.zeros((numFiles,15))
Binned_PTM_S_ALL = np.zeros((numFiles,15))
Bouts_NS_ALL = np.zeros((0,11))
Bouts_S_ALL = np.zeros((0,11))
Turns_NS_ALL = np.zeros(numFiles)
Turns_S_ALL = np.zeros(numFiles)
FSwim_NS_ALL = np.zeros(numFiles)
OrtHist_S_Noxious_ALL = np.zeros((numFiles,36))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file   
    BPS_NS = dataobject['BPS_NS'] 
    BPS_S = dataobject['BPS_S']
    DistanceT_NS = dataobject['DistanceT_NS']
    DistanceT_S = dataobject['DistanceT_S']
    avgdistPerBout_NS = dataobject['avgdistPerBout_NS']
    avgdistPerBout_S = dataobject['avgdistPerBout_S']
    avgPosition_NS = dataobject['avgPosition_NS']
    avgPosition_S = dataobject['avgPosition_S']
    avgSpeedPerBout_NS = dataobject['avgSpeedPerBout_NS']
    avgSpeedPerBout_S = dataobject['avgSpeedPerBout_S']
    avgBout_interval_NS = dataobject['avgBout_interval_NS']
    avgBout_interval_S = dataobject['avgBout_interval_S']
    numFreezes_NS = dataobject['numFreezes_NS']
    numFreezes_S = dataobject['numFreezes_S']
    Percent_Moving_NS = dataobject['Percent_Moving_NS'] 
    Percent_Moving_S = dataobject['Percent_Moving_S'] 
    Percent_Paused_NS = dataobject['Percent_Paused_NS']
    Percent_Paused_S = dataobject['Percent_Paused_S']
    Binned_PTF_NS = dataobject['Binned_PTF_NS']
    Binned_PTF_S = dataobject['Binned_PTF_S']
    Binned_PTM_NS = dataobject['Binned_PTM_NS']
    Binned_PTM_S = dataobject['Binned_PTM_S']
    Bouts_NS = dataobject['Bouts_NS']  
    Bouts_S = dataobject['Bouts_S'] 
    Turns_NS = dataobject['Turns_NS']
    Turns_S = dataobject['Turns_S']
    FSwim_NS= dataobject['FSwim_NS']
    OrtHist_S_Noxious = dataobject['OrtHist_S_Noxious']   
        
    # Make an array with all summary stats
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    DistanceT_NS_ALL[f] = DistanceT_NS
    DistanceT_S_ALL[f] = DistanceT_S
    avgdistPerBout_NS_ALL[f] = avgdistPerBout_NS
    avgdistPerBout_S_ALL[f] = avgdistPerBout_S
    avgPosition_NS_ALL[f] = avgPosition_NS
    avgPosition_S_ALL[f] = avgPosition_S
    avgSpeedPerBout_NS_ALL[f]= avgSpeedPerBout_NS
    avgSpeedPerBout_S_ALL[f]= avgSpeedPerBout_S
    avgBout_interval_NS_ALL[f]= avgBout_interval_NS
    avgBout_interval_S_ALL[f]= avgBout_interval_S
    numFreezes_NS_ALL[f] = numFreezes_NS
    numFreezes_S_ALL[f] = numFreezes_S
    Percent_Moving_NS_ALL[f] = Percent_Moving_NS
    Percent_Moving_S_ALL[f] = Percent_Moving_S
    Percent_Paused_NS_ALL[f] = Percent_Paused_NS
    Percent_Paused_S_ALL[f] = Percent_Paused_S
    Binned_PTF_NS_ALL[f,:] = Binned_PTF_NS
    Binned_PTF_S_ALL[f,:] = Binned_PTF_S
    Binned_PTM_NS_ALL[f,:] = Binned_PTM_NS
    Binned_PTM_S_ALL[f,:] = Binned_PTM_S
    Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
    Turns_NS_ALL[f] = Turns_NS
    Turns_S_ALL[f] = Turns_S
    FSwim_NS_ALL[f] = FSwim_NS
    OrtHist_S_Noxious_ALL[f,:] = OrtHist_S_Noxious
    
# Add to summary lists
BPS_summary = pd.DataFrame({'NS': BPS_NS_ALL,'S': BPS_S_ALL, 'Cluster': df['ClusterLabel']})
DistanceT_summary = pd.DataFrame({'NS': DistanceT_NS_ALL, 'S': DistanceT_S_ALL, 'Cluster': df['ClusterLabel']})
avgdistPerBout_summary = pd.DataFrame({'NS': avgdistPerBout_NS_ALL,'S': avgdistPerBout_S_ALL,'Cluster': df['ClusterLabel']})
avgPosition_summary = pd.DataFrame({'NS': avgPosition_NS_ALL,'S': avgPosition_S_ALL,'Cluster': df['ClusterLabel']})
avgSpeedPerBout_summary = pd.DataFrame({'NS': avgSpeedPerBout_NS_ALL,'S': avgSpeedPerBout_S_ALL,'Cluster': df['ClusterLabel']})
avgBout_interval_summary = pd.DataFrame({'NS': avgBout_interval_NS_ALL,'S': avgBout_interval_S_ALL,'Cluster': df['ClusterLabel']})
Turns_summary = pd.DataFrame({'NS': Turns_NS_ALL,'S':Turns_S_ALL, 'Cluster': df['ClusterLabel']})
numFreezes_summary = pd.DataFrame({'NS': numFreezes_NS_ALL,'S': numFreezes_S_ALL, 'Cluster': df['ClusterLabel']})
Percent_Moving_summary = pd.DataFrame({'NS': Percent_Moving_NS_ALL,'S': Percent_Moving_S_ALL, 'Cluster': df['ClusterLabel']})
Percent_Paused_summary= pd.DataFrame({'NS': Percent_Paused_NS_ALL,'S': Percent_Paused_S_ALL, 'Cluster': df['ClusterLabel']})

Binned_PTM_summary = pd.DataFrame(Binned_PTM_S_ALL)
Binned_PTM_summary['Cluster'] = df['ClusterLabel']

Binned_PTF_summary = pd.DataFrame(Binned_PTF_S_ALL)
Binned_PTF_summary['Cluster'] = df['ClusterLabel']


Ort_Noxious_summary = pd.DataFrame(OrtHist_S_Noxious_ALL)
Ort_Noxious_summary['Cluster']= df['ClusterLabel']


def cat_box_plot(data, title, ylabel,ylim,FigureFolder):

    # Convert 'Cluster' to string for categorical plot
    data['Cluster'] = data['Cluster'].astype(str)

    # Combine 'NS' and 'S' into a single DataFrame for plotting
    combined_df = pd.melt(data, id_vars='Cluster', var_name='Variable', value_name='Values')
    cluster_order = ['Low-Low', 'High-Low', 'High-High', 'Low-High']
    
    # Creating grouped boxplots for 'NS' and 'S with 'Clusters' as hue
    plt.figure(figsize=(8, 10))
    plt.title(title, fontsize=36, y=1.1, fontname="Arial")
    
    ax = sns.boxplot(x='Variable', y='Values', hue='Cluster', hue_order=cluster_order,data=combined_df, color='whitesmoke', showfliers=False)
    ax = sns.stripplot(x='Variable', y='Values', hue='Cluster',hue_order=cluster_order, data=combined_df, dodge=True, size=6,jitter=True, palette = ['yellow','steelblue', 'cadetblue' ,'lightgreen'])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=2)
    
    # Customize legend

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[len(data['Cluster'].unique()):], labels[len(data['Cluster'].unique()):], fontsize=16)
    
    plt.xticks([0, 1], ['No SC', 'SC'], fontsize=30, fontname="Arial")
    plt.yticks(fontsize=26)
    ax.set_xlabel(None)
    plt.ylabel(ylabel, fontsize=26, fontname="Arial")
    plt.ylim(ylim)
    ax.spines['left'].set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(2.5)
    sns.despine()
    
    
    # Save the figure
    plt.savefig(FigureFolder + '/' + title + '.eps', dpi=300, bbox_inches='tight')
    plt.show()
        
#Plot position
cat_box_plot(avgPosition_summary, 'Position', 'Average Position (mm)',(0,100), FigureFolder)
#Plot number of Bouts per second
cat_box_plot(BPS_summary, 'BPS', 'Number of Bouts per Second (s)',(0,4), FigureFolder)
#Plot distance Travelled
cat_box_plot(DistanceT_summary, 'Distance Travelled', 'Total Distance Travelled (mm)',(0,15000), FigureFolder)
#Plot distance Travelled per Bout
cat_box_plot(avgdistPerBout_summary, 'Distance Travelled per Bout', 'Total Distance Travelled (mm)',(0,10) , FigureFolder)
#Plot average Speed
cat_box_plot(avgSpeedPerBout_summary, 'Bout Speed', 'average speed (mm/s)',(0,10) , FigureFolder)
#Plot inter bout interval
cat_box_plot(avgBout_interval_summary, 'Inter-Bout Interval', 'interval (s)',(0,1.5) , FigureFolder)
#Plot number of Freezes
cat_box_plot(numFreezes_summary, 'Freezes', 'Number of Freezes',(0,120), FigureFolder)
#Plot proportion of turns
cat_box_plot(Turns_summary, 'Turn', 'proportion of turns',(0,1), FigureFolder)
#Plot Time Moving
cat_box_plot(Percent_Moving_summary, 'Moving', '% Time Moving',(0,100), FigureFolder)
#Plot Time Pausing
cat_box_plot(Percent_Moving_summary, 'Pausing', '% Time Pausing',(0,100), FigureFolder)

#Fingerprint

# Create a DataFrame to store z-scores
zscore_df = pd.DataFrame({'Cluster': df['ClusterLabel']})
stats_df = pd.DataFrame()

# List of summaries to compare along with their names
summaries_to_compare = [
    ('BPS', BPS_summary),
    ('Distance', avgdistPerBout_summary),
    ('Speed', avgSpeedPerBout_summary),
    ('Turn', Turns_summary),
    ('Interval', avgBout_interval_summary),
    ('Freezes', numFreezes_summary),
]

# Iterate over each summary and calculate z-scores
for summary_name,summary_df in summaries_to_compare:
    # Extract the 'NS' and 'S' columns from the original DataFrame
    NS_column = summary_df['NS']
    S_column = summary_df['S']
     
    NS_Mean = NS_column.mean()
    NS_Std = NS_column.std()

    zscore =(S_column-NS_Mean)/NS_Std
     
    # Add the z-scores to the DataFrame with new columns for each summary
    zscore_df[f'{summary_name}'] = zscore
    
    # Perform Mann-Whitney U test for each cluster
    cluster_p_values = []
    for cluster_name, group_data in summary_df.groupby('Cluster'):
        NS_cluster = group_data['NS']
        S_cluster = group_data['S']
        _, p_value = mannwhitneyu(NS_cluster, S_cluster, alternative='two-sided')
        cluster_p_values.append((cluster_name, p_value))

    for cluster_name, p_value in cluster_p_values:
        stats_df.loc[cluster_name, f'{summary_name}_p_value'] = p_value
        
        
#plot all behaviours as a heatmap
cluster_order = ['Low-Low', 'High-Low', 'High-High', 'Low-High']
zscore_df['Cluster'] = pd.Categorical(zscore_df['Cluster'], categories=cluster_order, ordered=True)
df_MeanZ = zscore_df.groupby('Cluster').mean()

fingerprint =plt.figure(figsize= (10,6))

ax=sns.heatmap(df_MeanZ,vmin=-0.7,vmax=0.7 ,cmap= 'coolwarm', linewidth=2, square = True)#, cbar_kws={'shrink':0.8})
ax.set(xlabel='', ylabel='')
ax.xaxis.tick_top()
plt.xticks(rotation=20,fontsize =16, fontname='Arial')
plt.yticks(rotation= 360, fontsize=16, fontname='Arial')

# Add p-values on top of the heatmap
for i, cluster_name in enumerate(cluster_order):
    for j, (summary_name, _) in enumerate(summaries_to_compare):
        value = stats_df.loc[cluster_name, f'{summary_name}_p_value']
        if value < 0.001:
            ax.text(j+0.5, i+0.5, "p<0.001", ha="center", va="center", color='black', fontsize = 10)
        elif value < 0.01:
            ax.text(j+0.5, i+0.5, "p<0.01", ha="center", va="center", color='black', fontsize=10)
        elif value < 0.05:
            ax.text(j+0.5, i+0.5, "p<0.05", ha="center", va="center", color='black', fontsize=10)

fingerprint.savefig(FigureFolder + '/fingerprint.eps', dpi=300, bbox_inches='tight')            




# PTM vs PTF

cluster_colors = {'Low-Low': '#FCE205', 'High-High': 'cadetblue', 'High-Low': 'steelblue', 'Low-High': 'lightgreen'}


for cluster_label, cluster_data in Binned_PTM_summary.groupby('Cluster'):

    # Get the color for the current cluster label
    cluster_color = cluster_colors[cluster_label]
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Plot PTM
    m_ptm = np.nanmean(cluster_data.iloc[:, :-1], axis=0)
    std_ptm = np.nanstd(cluster_data.iloc[:, :-1], axis=0)
    valid_ptm = ~np.isnan(cluster_data.iloc[:, :-1])
    n_ptm = np.sum(valid_ptm, axis=0)
    se_ptm = std_ptm / np.sqrt(n_ptm - 1)

    ax.plot(m_ptm,linewidth=4, marker='o', markersize=10,label=f'% Moving',color=cluster_color)
    ax.plot(m_ptm + se_ptm, linewidth=1, alpha=0.5,color=cluster_color)
    ax.plot(m_ptm - se_ptm,linewidth=1, alpha=0.5, color=cluster_color)
    ax.fill_between(np.arange(15), m_ptm - se_ptm, m_ptm + se_ptm, alpha=0.2, color=cluster_color)

    # Plot PTF
    cluster_data_ptf = Binned_PTF_summary[Binned_PTF_summary['Cluster'] == cluster_label]
    m_ptf = np.nanmean(cluster_data_ptf.iloc[:, :-1], axis=0)
    std_ptf = np.nanstd(cluster_data_ptf.iloc[:, :-1], axis=0)
    valid_ptf = ~np.isnan(cluster_data_ptf.iloc[:, :-1])
    n_ptf = np.sum(valid_ptf, axis=0)
    se_ptf = std_ptf / np.sqrt(n_ptf - 1)

    ax.plot(m_ptf, linewidth=4, marker='^', markersize=10, label=f'% Freezing', color=cluster_color)
    ax.plot(m_ptf + se_ptf, linewidth=1, alpha=0.5, color=cluster_color)
    ax.plot(m_ptf - se_ptf, linewidth=1, alpha=0.5, color=cluster_color)
    ax.fill_between(np.arange(15), m_ptf - se_ptf, m_ptf + se_ptf, alpha=0.2, color=cluster_color)

    # Set labels and legend
    ax.legend(loc="upper right", fontsize=24)
    ax.set_xlabel('Minutes', fontsize=24, fontname='Arial')
    ax.set_ylabel('Percentage', fontsize=24, fontname='Arial')
    ax.set_ylim(-1, 90)
    ax.tick_params(axis='both', which='both', labelsize=20)
    ax.set_xticks(np.arange(0, 15, step=2))
    ax.set_xticklabels(['0', '2', '4', '6', '8', '10', '12', '14'], fontsize=24)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Add cluster label as title
    ax.set_title(f'Cluster: {cluster_label}', fontsize=24, fontname='Arial')

    # Save the plot
    plt.savefig(FigureFolder + f'/PTM_PTF_{cluster_label}.png', dpi=300, bbox_inches='tight')
    plt.savefig(FigureFolder + f'/PTM_PTF_{cluster_label}.eps', dpi=300, bbox_inches='tight')



df = Ort_Noxious_summary

cluster_labels = df['Cluster'].unique()
order = ['Low-Low','High-Low','Low-High','High-High']

# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10), subplot_kw=dict(polar=True))

axes = axes.flatten()

# Iterate over cluster labels and create polar plots
for i, cluster_label in enumerate(order):
    ax = axes[i]
    
    cluster_data = df[df['Cluster'] == cluster_label].iloc[:, :-1].mean(axis=0)
    cluster_color = cluster_colors[cluster_label]
    
    # Create theta values (angle bins)
    theta = np.linspace(0, 2*np.pi, len(cluster_data), endpoint=False)
    
    # Plot the polar plot
    ax.plot(theta, cluster_data, label=cluster_label, color=cluster_color, linewidth =4)
    ax.fill(theta, cluster_data, alpha=0.4, color=cluster_color)
    
    ax.set_title(cluster_label, fontsize=24, fontname='Arial')
    
    ax.set_theta_zero_location("W")  
    ax.set_theta_direction(-1)
    ax.set_yticklabels([])

    ax.tick_params(axis='x',labelsize=18,pad=15, direction='out') 

plt.tight_layout()
plt.savefig(FigureFolder + f'/PolarPlot.eps', dpi=300, bbox_inches='tight')



