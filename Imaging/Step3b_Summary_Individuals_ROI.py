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
analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/HABITUATION/Cfos_Values/SERT_hyp_3'
cFos_file = analysis_folder + '/SERT_hyp_3_cFos.npz'

# Load data
npzfile = np.load(cFos_file)
cFos = npzfile['cFos_values']
behaviour_metrics = npzfile['behaviour_metrics']
roi_name = npzfile['roi_name']

TTS= behaviour_metrics[:,3]

# Analyze
mean = np.mean(cFos)
std = np.std(cFos)


cfos_val = plt.figure(dpi=300)
plt.title(roi_name,fontsize= 14)  

s1 = pd.Series(cFos, name='cFos')
s2 = pd.Series(TTS, name='Relative Position Shift')

sns.stripplot(x=s2,y=s1, color='steelblue')

cfos_val.savefig(analysis_folder + '/cfos_value.png', dpi=300, bbox_inches='tight')

# FIN
