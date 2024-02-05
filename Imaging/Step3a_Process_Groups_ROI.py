# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 15:56:54 2023
This code reads a summary list file and loads a specific ROI 
- calculates the cfos_value of a specific ROI according to a defined behaviour group
- save values as npz 
@author: Alizee Kastler
"""


# -----------------------------------------------------------------------------

# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
sys.path.append(lib_path)

# Import useful libraries
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


analysis_path_A = analysis_folder + '/Baseline_cfos.npz'
analysis_path_B = analysis_folder + '/Heat_cfos.npz'
analysis_path_C = analysis_folder + '/AITC_cfos.npz'
analysis_path_D = analysis_folder + '/Social_cfos.npz'
analysis_path_E = analysis_folder + '/DMSO_cfos.npz'


# -----------------------------------------------------------------------------

# Read summary list
cfos_paths, group_names = SPCFOS.read_summarylist(summaryListFile, normalized=True)
cfos_paths = np.array(cfos_paths)
group_names = np.array(group_names)


group_A = (group_names == 'Baseline')
cfos_values_A = SPCFOS.cfos_value_ROI(group_A, cfos_paths, roi_path)
np.savez(analysis_path_A, cfos_values=cfos_values_A, group_name=group_A, roi_name=roi_name)

group_B = (group_names == 'Heat')
cfos_values_B = SPCFOS.cfos_value_ROI(group_B, cfos_paths, roi_path)
np.savez(analysis_path_B, cfos_values=cfos_values_B, group_name=group_B, roi_name=roi_name)

group_C = (group_names == 'AITC')
cfos_values_C = SPCFOS.cfos_value_ROI(group_C, cfos_paths, roi_path)
np.savez(analysis_path_C, cfos_values=cfos_values_C, group_name=group_C, roi_name=roi_name)

group_D = (group_names == 'Social')
cfos_values_D = SPCFOS.cfos_value_ROI(group_D, cfos_paths, roi_path)
np.savez(analysis_path_D, cfos_values=cfos_values_D, group_name=group_D, roi_name=roi_name)

group_E = (group_names == 'DMSO')
cfos_values_E = SPCFOS.cfos_value_ROI(group_E, cfos_paths, roi_path)
np.savez(analysis_path_E, cfos_values=cfos_values_E, group_name=group_E, roi_name=roi_name)



# Plot
bar_colours = [ "#aaa4c8","#452775",'plum', '#d75d1b', 'lightgrey']


cfos_val = plt.figure(figsize=(4,4),dpi=300)
plt.title(roi_name,fontsize= 14)  

s1 = pd.Series(cfos_values_A, name='Baseline')
s2 = pd.Series(cfos_values_B, name='Noxious')
s3 = pd.Series(cfos_values_E, name='DMSO')
s4 = pd.Series(cfos_values_C, name='AITC')
s5 = pd.Series(cfos_values_D, name='Social')
df = pd.concat([s1,s2,s3,s4,s5], axis=1)
ax=sns.swarmplot(data=df, orient="v", size=6, palette=bar_colours, zorder=1) 
with plt.rc_context({'lines.linewidth': 0.8}):
    sns.pointplot(data=df, orient="v", linewidth=0, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)
plt.ylabel("cFos expression")
ax.set(ylim=(0, 4))
sns.despine()
plt.show ()

cfos_val.savefig(analysis_folder + '/'+ roi_name +'.png', dpi=300, bbox_inches='tight')





#FIN