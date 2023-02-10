"""
This script plots summary stats for ROI cFOS activity across groups
@author: Dreosti Lab
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:/Repos/Social_Pain/libs'
# -----------------------------------------------------------------------------

# Set Library Paths
import sys
import os
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd



#---------------------------------------------------------------------------
#---------------------------------------------------------------------------

# Set cFos file
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/HABITUATION/Cfos_Values/PAG'
cFos_file = analysis_folder + '/PAG_cFos_2.npz'

# Load data
npzfile = np.load(cFos_file)
cFos = npzfile['cFos_values']
behaviour_metrics = npzfile['behaviour_metrics']
roi_name = npzfile['roi_name']

TTS= behaviour_metrics[:,4]

# Analyze
mean = np.mean(cFos)
std = np.std(cFos)


cfos_val = plt.figure(dpi=300)
plt.title(roi_name,fontsize= 14)  

s1 = pd.Series(cFos, name='cFos')
s2 = pd.Series(TTS, name='Relative Position Shift')

ax=sns.stripplot(x=s2,y=s1, color='steelblue')
ax.set_xticklabels(['[-10;-5]', '[-5;5]','[5;10]'])
ax.set(ylim=(0, 3))
plt.ylabel('cFos expression')


cfos_val.savefig(analysis_folder + '/cfos_value_TTS_cat.png', dpi=300, bbox_inches='tight')

# FIN
