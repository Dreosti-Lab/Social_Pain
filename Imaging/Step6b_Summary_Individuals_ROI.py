"""
This script plots cfos_values within the selected ROI for specified behaviour_metric
- reads .npz file from Step4a

"""

# -----------------------------------------------------------------------------

# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'

import sys
import os
sys.path.append(lib_path)

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
import pandas as pd

# -----------------------------------------------------------------------------

# Set cFos file
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/HABITUATION/Cfos_Values/Habenula'
cFos_file = analysis_folder + '/Habenula_cFos_2.npz'

# Load cFos data
npzfile = np.load(cFos_file)
cFos = npzfile['cFos_values']
behaviour_metrics = npzfile['behaviour_metrics']
roi_name = npzfile['roi_name']


# Set Behaviour_metric to plot
TTS= behaviour_metrics[:,2]


# Plot cfos_value within ROI according to specifies behaviour metric 
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

# Save Plot
cfos_val.savefig(analysis_folder + '/cfos_value_PositionS.png', dpi=300, bbox_inches='tight')

# FIN
