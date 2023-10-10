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
from scipy.stats import mode

ROI_folder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'
FigureFolder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/figures'

# List of mask folders
mask_folders = [
   ROI_folder + 'cH',
   ROI_folder + 'PTN'
]

mask_names = [
   'cH',
   'PTN'
    ]

# Initialize empty lists to store data for each condition
cfos_values_baseline = []
cfos_values_heat = []
cfos_values_AITC = []
cfos_values_social = []


# Iterate through mask folders
for mask_folder in mask_folders:
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
    cfos_values_baseline.append(cfos_baseline_data['cfos_values'])
    cfos_values_heat.append(cfos_heat_data['cfos_values'])
    cfos_values_AITC.append(cfos_AITC_data['cfos_values'])
    cfos_values_social.append(cfos_social_data['cfos_values'])


# Convert lists to NumPy arrays for easier calculation
cfos_values_baseline = np.array(cfos_values_baseline)
cfos_values_heat = np.array(cfos_values_heat)
cfos_values_AITC = np.array(cfos_values_AITC)
cfos_values_social = np.array(cfos_values_social)

# Calculate mean and standard error for each condition in each mask
mean_baseline = np.mean(cfos_values_baseline, axis=1)

std_error_baseline = np.std(cfos_values_baseline, axis=1) 
std_error_heat = np.std(cfos_values_heat, axis=1)
std_error_AITC = np.std(cfos_values_AITC, axis=1)


# Calculate indices for Heat and AITC conditions
index_heat = np.mean(cfos_values_heat, axis=1)  - mean_baseline
index_AITC = np.mean(cfos_values_AITC, axis=1) - mean_baseline


# Create a figure with a fixed size
fig, ax = plt.subplots(figsize=(4, 3))
bar_width = 0.35

# Calculate the center positions for bars
num_masks = len(mask_names)
center_positions = np.arange(num_masks)

# Plot the bars for Heat and AITC conditions
ax.bar(center_positions - bar_width / 2, index_heat, bar_width, yerr=std_error_heat, label='Heat', color='#452775')
ax.bar(center_positions + bar_width / 2, index_AITC, bar_width, yerr=std_error_AITC, label='AITC', color='#d85c1a')

# Add labels and legend
ax.set_ylabel("Δcfos/cfos Baseline", fontsize=16)
ax.set_ylim(-1, 2)
ax.axhline(y=0, color='black', linestyle='solid', linewidth=2)
ax.set_xticks(center_positions)
ax.set_xticklabels(mask_names, fontweight='bold', fontsize=16)
ax.legend()

ax.grid(linestyle='--', alpha=0.7)

# Add empty bars for missing masks (if needed)
if num_masks < 4:
    empty_bar_positions = np.arange(num_masks, 4)
    for position in empty_bar_positions:
        ax.bar(position, 0, bar_width, color='white')  # Adding empty white bars


plt.savefig(FigureFolder+ 'Heat_AITC_270.png', dpi=300, bbox_inches='tight')