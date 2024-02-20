# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:57:58 2023
This script performs cross-correlation analysis comparing a set of cfos values in ROIs for different conditions.

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

ROI_folder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'
FigureFolder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/figures'

# List of mask folders
mask_folders = [
   ROI_folder + 'cH',
   ROI_folder + 'Dm',
   ROI_folder + 'GC',
   ROI_folder + 'IN',
   ROI_folder + 'P',
   ROI_folder + 'PG',
   ROI_folder + 'PGZ',
   ROI_folder + 'Ppa',
   ROI_folder + 'Ppp',
   ROI_folder + 'PVO',
   ROI_folder + 'TPd',
   ROI_folder + 'TPp',
   ROI_folder + 'TS',
   ROI_folder + 'Vd',
   ROI_folder + 'vHb',
   ROI_folder + 'Vp'
   
]

# Initialize empty lists to store data for each condition
mask_names = []
index_values = []

# Iterate through mask folders
for i in range(len(mask_folders)):
    mask_folder = mask_folders[i]
    mask_names.append(os.path.basename(mask_folder))
    
    
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
    cfos_values_social = cfos_social_data['cfos_values']
    
    # Calculate the index for Heat condition for the current mask
    mean_baseline = np.mean(cfos_values_baseline)
    mean_DMSO = np.mean(cfos_values_DMSO)
    index_heat = (np.mean(cfos_values_heat) - mean_baseline) / mean_baseline
    index_AITC = (np.mean(cfos_values_AITC) - mean_DMSO) / mean_DMSO
    index_social = (np.mean(cfos_values_social) - mean_baseline) / mean_baseline

    index_values.append((index_heat, index_AITC, index_social))

# Create a scatter plot of Heat vs. AITC indices for each mask
plt.figure(figsize=(10, 6))
for i in range(len(mask_folders)):
    index_heat, index_AITC, _ = index_values[i]  # Extract the indices from the list
    plt.scatter(index_heat, index_AITC, label=mask_names[i])

plt.xlabel('Heat Index')
plt.ylabel('AITC Index')
plt.title('Heat vs. AITC Index for Different Masks')
plt.legend()


# Calculate the cross-correlation
heat = np.array([column[0] for column in index_values])
AITC = np.array([column[1] for column in index_values])
social = np.array([column[2] for column in index_values])

heat = heat - np.mean(heat)
AITC = AITC- np.mean(AITC)

corr_nox = np.sum (heat * AITC)
corr_soc = np.sum (heat * social)

#Normalisation 
auto_correlation = (np.sum (heat * heat) + np.sum(AITC*AITC))/2
Norm_corr_nox = corr_nox/auto_correlation

#Permutations

permuted_corrs = []

for _ in range(10000):
    
    # Permute the 'AITC' values
    permuted_AITC = np.random.permutation(AITC)
    permuted_heat = np.random.permutation(heat)
    
    # Calculate the cross-correlation for the permuted data
    permuted_corr = np.sum(AITC * permuted_heat)
    permuted_corrs.append(permuted_corr)

# Calculate ratio of permutted correlations higher than initial
ratio = np.sum(np.array(permuted_corrs) > corr_nox)/10000


# Plot the distribution of permuted correlations
plt.figure(figsize=(8, 6))
sns.distplot(permuted_corrs, hist=False, kde=True,color='silver', kde_kws = {'shade':True, 'linewidth':2},label='Permuted Correlation')
#plt.hist(permuted_corrs, bins=50,alpha=0.5, color='whitesmoke',ec="k", label='Permuted Correlations')
plt.axvline(corr_nox, color='red', linewidth=2,linestyle='dashed', label='Initial Correlation')
plt.axvline([0], color='silver', linewidth=1)
plt.axvline([1], color='silver', linewidth=1)

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Cross-Correlation', fontsize =16)
plt.ylabel('Frequency', fontsize = 16)
plt.legend(fontsize=16)
plt.grid(b=None)
sns.despine()
#plt.title('Distribution of Permuted Correlations')

