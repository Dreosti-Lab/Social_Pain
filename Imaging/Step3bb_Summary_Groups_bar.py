# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:04:52 2023

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

ROI_folder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'
FigureFolder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/figures'

# List of mask folders
mask_folders = [
   ROI_folder + 'Dp',
   ROI_folder + 'GC',
   ROI_folder + 'TS',
   ROI_folder + 'vHb'
]

mask_names = [
   'Dp',
   'GC',
   'TS',
   'vHb'
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
    cfos_AITC_file = os.path.join(mask_folder, 'AITC_cfos.npz')
    cfos_social_file = os.path.join(mask_folder, 'Social_cfos.npz')
    
    # Load data from the NPZ files
    cfos_baseline_data = np.load(cfos_baseline_file)
    cfos_heat_data = np.load(cfos_heat_file)
    cfos_AITC_data = np.load(cfos_AITC_file)
    cfos_social_data = np.load(cfos_social_file)
    
    # Extract and append data for each condition
    cfos_values_baseline = cfos_baseline_data['cfos_values']
    cfos_values_heat = cfos_heat_data['cfos_values']
    cfos_values_AITC = cfos_AITC_data['cfos_values']
    
    # Perform Mann-Whitney U-test
    u_statistic_heat, p_value_heat = mannwhitneyu(cfos_values_heat, cfos_values_baseline, alternative='two-sided')
    u_statistic_AITC, p_value_AITC = mannwhitneyu(cfos_values_AITC, cfos_values_baseline, alternative='two-sided')

    # Append U-test results to the respective lists
    u_test_results_heat.append((u_statistic_heat, p_value_heat))
    u_test_results_AITC.append((u_statistic_AITC, p_value_AITC))
    
    # Calculate the index for Heat condition for the current mask
    mean_baseline = np.mean(cfos_values_baseline)
    index_heat = (cfos_values_heat - mean_baseline) / mean_baseline
    index_AITC = (cfos_values_AITC - mean_baseline) / mean_baseline
   
    baseline.append (mean_baseline)
    index_values.append((index_heat, index_AITC))
 
# Create a figure with a fixed size
fig, ax = plt.subplots(figsize=(4, 3))
bar_width = 0.35

# Calculate the center positions for bars
num_masks = len(mask_names)
center_positions = np.arange(num_masks)
colors = ['#452775', '#d85c1a']

# Iterate through the masks and plot the bars for Heat and AITC conditions
for i in range(num_masks):
    # Calculate standard errors
    std_error_heat = np.std(index_values[i][0]) / np.sqrt(len(index_values[i][0]))
    std_error_AITC = np.std(index_values[i][1]) / np.sqrt(len(index_values[i][1]))

    # Plot the bars for Heat and AITC conditions for the current mask with error bars
    ax.bar(center_positions[i] - bar_width / 2, np.mean(index_values[i][0]), bar_width, yerr=std_error_heat, color=colors[0], linewidth=0)
    ax.bar(center_positions[i] + bar_width / 2, np.mean(index_values[i][1]), bar_width, yerr=std_error_AITC, color=colors[1], linewidth=0)

    # Add significance asterisks if p-value is below a threshold (e.g., 0.05)
    if u_test_results_heat[i][1] <= 0.055:
        ax.text(center_positions[i] - bar_width / 2, 1.5, '*', horizontalalignment='center', fontsize=14)

    if u_test_results_AITC[i][1] <= 0.055:
        ax.text(center_positions[i] + bar_width / 2, 1.5, '*', horizontalalignment='center', fontsize=14)

# Add labels and legend
ax.legend(['Heat', 'AITC'], loc='upper right')
ax.set_ylabel("Δcfos/cfos Baseline", fontsize=16)
ax.set_ylim(-0.5, 1.5)
ax.axhline(y=0, color='black', linestyle='solid', linewidth=2)
ax.set_xticks(center_positions)
ax.set_xticklabels(mask_names, fontweight='bold', fontsize=16)


ax.grid(linestyle='--', alpha=0.7)

# Add empty bars for missing masks (if needed)
if num_masks < 4:
    empty_bar_positions = np.arange(num_masks, 4)
    for position in empty_bar_positions:
        ax.bar(position, 0, bar_width, color='white')  # Adding empty white bars


plt.savefig(FigureFolder+ 'Heat_AITC_130.png', dpi=300, bbox_inches='tight')


