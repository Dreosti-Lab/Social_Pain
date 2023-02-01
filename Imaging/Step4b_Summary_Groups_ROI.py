"""
This script plots summary stats for ROI cFOS activity across groups
@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:\Repos\Dreosti-Lab\Social_Zebrafish\libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
import os
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import SP_cfos as SZCFOS
import seaborn as sns
import pandas as pd



#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set cFos file (group A and B)
cFos_file_A = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/SERT_hyp_3/group_A_cFos.npz'
cFos_file_B = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/SERT_hyp_3/group_B_cFos.npz'
cFos_file_C = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/SERT_hyp_3/group_C_cFos.npz'
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/SERT_hyp_3'

# Load data
npzfile = np.load(cFos_file_A)
cFos_A = npzfile['cFos_values']
group_name_A = npzfile['group_name']
roi_name_A = npzfile['roi_name']

npzfile = np.load(cFos_file_B)
cFos_B = npzfile['cFos_values']
group_name_B = npzfile['group_name']
roi_name_B = npzfile['roi_name']

npzfile = np.load(cFos_file_C)
cFos_C = npzfile['cFos_values']
group_name_C = npzfile['group_name']
roi_name_C = npzfile['roi_name']

# Analyze
mean_A = np.mean(cFos_A)
std_A = np.std(cFos_A)

mean_B = np.mean(cFos_B)
std_B = np.std(cFos_B)

mean_C = np.mean(cFos_C)
std_C = np.std(cFos_C)

# Plot
bar_colours = [ "#c0c0c0","#ff0000",'#0033ff']

cfos_val = plt.figure(dpi=300)
plt.title(roi_name_A,fontsize= 14)  

s1 = pd.Series(cFos_A, name='Baseline')
s2 = pd.Series(cFos_B, name='Social')
s3 = pd.Series(cFos_C, name='Noxious')
df = pd.concat([s1,s2,s3], axis=1)
sns.swarmplot(data=df, orient="v", size=6, color="0.25",palette=sns.color_palette(bar_colours), zorder=1) 
with plt.rc_context({'lines.linewidth': 0.8}):
    sns.pointplot(data=df, orient="v", linewidth=.1, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)


cfos_val.savefig(analysis_folder + '/cfos_value.png', dpi=300, bbox_inches='tight')




# FIN
