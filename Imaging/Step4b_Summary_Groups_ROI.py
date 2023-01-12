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
cFos_file_A = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/GRADIENT/Cfos_Values/Resilient_PoA_cFos.npz'
cFos_file_B = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/GRADIENT/Cfos_Values/Susceptible_PoA_cFos.npz'

# Load data
npzfile = np.load(cFos_file_A)
cFos_A = npzfile['cFos_values']
group_name_A = npzfile['group_name']
roi_name_A = npzfile['roi_name']

npzfile = np.load(cFos_file_B)
cFos_B = npzfile['cFos_values']
group_name_B = npzfile['group_name']
roi_name_B = npzfile['roi_name']

# Analyze
mean_A = np.mean(cFos_A)
std_A = np.std(cFos_A)

mean_B = np.mean(cFos_B)
std_B = np.std(cFos_B)

# Plot
bar_colours = [ "#c0c0c0","#ff0000"]
plt.figure()
s1 = pd.Series(cFos_A, name='meanA')
s2 = pd.Series(cFos_B, name='meanB')
df = pd.concat([s1,s2], axis=1)
sns.swarmplot(data=df, orient="v", size=6, color="0.25",palette=sns.color_palette(bar_colours), zorder=1) 
with plt.rc_context({'lines.linewidth': 0.8}):
    sns.pointplot(data=df, orient="v", linewidth=.1, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)





# FIN
