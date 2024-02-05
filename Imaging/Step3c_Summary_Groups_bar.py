# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:04:52 2023

This code loads the npz file with cfos values for the list of masks selected
- for each mask the cfos values are compared between the specified condition and the value at baseline. 
- mann whitney u test is used for comparison
- plot a bar chart with two different conditons showing the cfos values for the specified masks
- save bar plots as png

@author: Alizee Kastler
"""
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:/Repos/Social_Pain/libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
import os
sys.path.append(lib_path)

import numpy as np
import SP_cfos as SPCFOS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu
import random
from matplotlib.lines import Line2D

ROI_folder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'
FigureFolder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/figures'

# List of mask folders
mask_folders = [
   ROI_folder + 'TS',
   ROI_folder + 'Ppp',
   ROI_folder + 'Dm'
]

mask_names = [
   'vHb',
   'PGZ',
   'Dm'
    ]

# Initialize empty lists to store data for each condition
baseline = []
index_values = []

# Initialize lists to store U-test results
u_test_results_heat = []
u_test_results_AITC = []


# Iterate through mask folders
for i in range(len(mask_folders)):
    mask_folder = mask_folders[i]

#for mask_folder in mask_folders:
    # Construct file paths for the NPZ files in each mask folder
    cfos_baseline_file = os.path.join(mask_folder, 'Baseline_cfos.npz')
    cfos_heat_file = os.path.join(mask_folder, 'Heat_cfos.npz')
    cfos_DMSO_file = os.path.join(mask_folder, 'DMSO_cfos.npz')
    cfos_AITC_file = os.path.join(mask_folder, 'AITC_cfos.npz')
    cfos_social_file = os.path.join(mask_folder, 'Social_cfos.npz')
    
    # Load data from the NPZ files
    cfos_baseline_data = np.load(cfos_baseline_file)
    cfos_heat_data = np.load(cfos_heat_file)
    cfos_DMSO_data = np.load(cfos_DMSO_file)
    cfos_AITC_data = np.load(cfos_AITC_file)
    cfos_social_data = np.load(cfos_social_file)
    
    # Extract and append data for each condition
    cfos_values_baseline = cfos_baseline_data['cfos_values']
    cfos_values_heat = cfos_heat_data['cfos_values']
    cfos_values_DMSO = cfos_DMSO_data['cfos_values']
    cfos_values_AITC = cfos_AITC_data['cfos_values']
    
    # Perform Mann-Whitney U-test
    u_statistic_heat, p_value_heat = mannwhitneyu(cfos_values_heat, cfos_values_baseline,  use_continuity=False,alternative='two-sided')
    u_statistic_AITC, p_value_AITC = mannwhitneyu(cfos_values_AITC, cfos_values_DMSO, use_continuity=False, alternative='two-sided')

    # Append U-test results to the respective lists
    u_test_results_heat.append((u_statistic_heat, p_value_heat))
    u_test_results_AITC.append((u_statistic_AITC, p_value_AITC))
    
    # Calculate the index for Heat condition for the current mask
    mean_baseline = np.mean(cfos_values_baseline)
    mean_DMSO = np.mean(cfos_values_DMSO)
    index_heat = (cfos_values_heat - mean_baseline) / mean_baseline
    index_AITC = (cfos_values_AITC - mean_DMSO) / mean_DMSO
   
    baseline.append (mean_baseline)
    index_values.append((index_heat, index_AITC))
 
# Create a figure with a fixed size
fig, ax = plt.subplots(figsize=(4, 3))
bar_width = 1

# Calculate the center positions for bars
num_masks = len(mask_names)
center_positions = np.arange(7)
colors = ['#452775', '#d85c1a']

# Iterate through the masks and plot the bars for Heat and AITC conditions
# for i in range(num_masks):

    # Calculate standard errors
    # std_error_heat = np.std(index_values[i][0]) / np.sqrt(len(index_values[i][0]))
    # std_error_AITC = np.std(index_values[i][1]) / np.sqrt(len(index_values[i][1]))

    # # Plot the bars for Heat and AITC conditions for the current mask with error bars
    # ax.bar(center_positions[i] - bar_width / 2, np.mean(index_values[i][0]), bar_width, yerr=std_error_heat, color='whitesmoke', edgecolor='gray', error_kw={'ecolor':'gray'},zorder=1)
    # ax.bar(center_positions[i] + bar_width / 2, np.mean(index_values[i][1]), bar_width, yerr=std_error_AITC, color='whitesmoke', edgecolor='gray', error_kw={'ecolor':'gray'},zorder=1)
     
    # # Plot individual values as a scatterplot above the bars
    # ax.scatter(np.repeat(center_positions[i] - bar_width / 2, len(index_values[i][0]))+[random.uniform(-0.05, 0.05) for _ in index_values[i][0]], index_values[i][0], color=colors[0], s=10,zorder=2)
    # ax.scatter(np.repeat(center_positions[i] + bar_width / 2, len(index_values[i][1]))+[random.uniform(-0.05, 0.05) for _ in index_values[i][1]], index_values[i][1], color=colors[1], s=10,zorder=2)

    # Add significance asterisks if p-value is below a threshold (e.g., 0.05)
    # if u_test_results_heat[i][1] <= 0.05:
    #     ax.text(center_positions[i], 1.5, '*', horizontalalignment='center', fontsize=16)

    # if u_test_results_AITC[i][1] <= 0.05:
    #     ax.text(1+center_positions[num_masks+i], 1.5, '*', horizontalalignment='center', fontsize=16)

# Plot Heat conditon first
for i in range(num_masks):    

    std_error_heat = np.std(index_values[i][0]) / np.sqrt(len(index_values[i][0]))
    
    ax.bar(center_positions[i], np.mean(index_values[i][0]), bar_width, yerr=std_error_heat, color='whitesmoke', edgecolor='gray', error_kw={'ecolor':'gray'}, zorder=1)
    ax.scatter(np.repeat(center_positions[i], len(index_values[i][0]))+[random.uniform(-0.05, 0.05) for _ in index_values[i][0]], index_values[i][0], color=colors[0], s=10, zorder=2)

    # Add significance asterisks if p-value is below a threshold (e.g., 0.05)
    if u_test_results_heat[i][1] <= 0.055:
        ax.text(center_positions[i], 1.5, '*', horizontalalignment='center', fontsize=20)

# Add the required number of empty bars after the actual mask bars for Heat condition
empty_bars_count = max(0, 3 - num_masks)

for i in range(empty_bars_count):
    ax.bar(num_masks + i, 1, bar_width, alpha=0)

# Plot AITC condition
AITC_start = max(4, num_masks)

for i in range(num_masks):
    # Calculate standard errors
    std_error_AITC = np.std(index_values[i][1]) / np.sqrt(len(index_values[i][1]))

    # Plot AITC condition
    ax.bar(AITC_start + i, np.mean(index_values[i][1]), bar_width, yerr=std_error_AITC, color='whitesmoke', edgecolor='gray', error_kw={'ecolor':'gray'}, zorder=1)
    ax.scatter(np.repeat(AITC_start + i, len(index_values[i][1]))+[random.uniform(-0.05, 0.05) for _ in index_values[i][1]], index_values[i][1], color=colors[1], s=10, zorder=2)

    # Add significance asterisks if p-value is below a threshold (e.g., 0.05)
    if u_test_results_AITC[i][1] <= 0.055:
        ax.text(AITC_start + i, 1.5, '*', horizontalalignment='center', fontsize=20)

for i in range(empty_bars_count):
    ax.bar(num_masks*2 + 2 + i, 1, bar_width, alpha=0)


# Add labels and legend
# Define the legend manually using Line2D objects with the correct colors
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[0], markersize=8, label='Heat'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[1], markersize=8, label='AITC')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
ax.set_ylabel("Δcfos/cfos Baseline", fontsize=18)
ax.set_ylim(-0.5, 1.5)
ax.axvline(x=3, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0, color='black', linestyle='solid', linewidth=1)

# Set the positions for the ticks
xtick_positions = np.arange(2 * num_masks + 1 + 2*empty_bars_count)
xtick_labels = [''] * len(xtick_positions)  # Create an empty label list
# Assign the mask names to corresponding positions in the label list
for i in range(num_masks):
    xtick_labels[i] = mask_names[i]
    xtick_labels[AITC_start + i] = mask_names[i]
# Set the ticks
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels, fontweight='bold', fontsize=14)

ax.tick_params(axis='y', labelsize=16)
ax.grid(linestyle='--', alpha=0.5)


# # Add empty bars for missing masks (if needed)
# if num_masks < 4:
#     empty_bar_positions = np.arange(num_masks, 4)
#     for position in empty_bar_positions:
#         ax.bar(position, 0, bar_width, color='white')  # Adding empty white bars


plt.savefig(FigureFolder+ 'Heat_AITC_230.png', dpi=300, bbox_inches='tight')


