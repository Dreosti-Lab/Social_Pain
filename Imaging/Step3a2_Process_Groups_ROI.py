# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:54 2023
This code reads a summary list file and loads a specific ROI 
- calculates the cfos_value of a specific ROI according to a defined behaviour group
- save values as npz 
@author: Alizee Kastler
"""


# -----------------------------------------------------------------------------

# Specify Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
sys.path.append(lib_path)

# Import necessary libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SPCFOS
import pandas as pd
import seaborn as sns


# -----------------------------------------------------------------------------

# Set Summary List
summaryListFile = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Cfos_Summary/Cfos_Summary_512_2.xlsx'

# Set ROI Path
roi_path = 'F:/ATLAS/MASK_CFOS/cH.tif'
roi_name = r'cH'

#Set analysis folder and filename
analysis_folder ='S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/'+ roi_name

# Create output folder (if doesn't exist)
if not os.path.exists(analysis_folder):
   os.makedirs(analysis_folder)


def calculate_and_save_cfos_value (group_name, cfos_paths, roi_path, analysis_folder):
    #generate analysis path
    analysis_path = os.path.join(analysis_folder, f'{group_name}_cfos.npz')
    group = (group_names==group_name)
    cfos_values = SPCFOS.cfos_value_ROI(group, cfos_paths,roi_path)
    np.savez(analysis_path, cfos_values=cfos_values, group_name = group, roi_name = roi_name)


# Read summary list
cfos_paths, group_names = SPCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
group_names = np.array(group_names)

groups = ['Baseline', 'Heat','AITC', 'Social','DMSO']
for group_name in groups:
    calculate_and_save_cfos_values(group_name,cfos_paths,roi_path,analysis_folder)

 

# Plot
bar_colours = [ "#aaa4c8","#452775",'plum', '#d75d1b', 'lightgrey']

cfos_val = plt.figure(figsize=(4,4),dpi=300)
plt.title(roi_name,fontsize= 14)  

dfs = []

for group_name in groups:
    analysis_path = os.path.join(analysis_folder, f'{group_name}_cfos.npz')
    npzfile = np.load(analysis_path)
    cfos_values = npzfile['cfos_values']
    dfs.append(pd.Series(cfos_values,name=group_name))

df=pd.concat(dfs,axis=1)

ax=sns.swarmplot(data=df, orient="v", size=6, palette=bar_colours, zorder=1) 
with plt.rc_context({'lines.linewidth': 0.8}):
    sns.pointplot(data=df, orient="v", linewidth=0, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)
plt.ylabel("cFos expression")
ax.set(ylim=(0, 4))
sns.despine()
plt.show ()

cfos_val.savefig(analysis_folder + '/'+ roi_name +'.png', dpi=300, bbox_inches='tight')

#FIN