"""
This script plots summary cfos_values for a slected ROI  across groups
-loads .npz files from step 3ba
@author: Alizee Kastler
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
import SP_cfos as SPCFOS
import seaborn as sns
import pandas as pd


#---------------------------------------------------------------------------

# Set cFos file (group A and B)
cfos_Baseline= r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/vHb/Baseline_cfos.npz'
cfos_Social = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/vHb/Social_cfos.npz'
cfos_Noxious= r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI/vHb/Noxious_cfos.npz'

analysis_folder = r'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI'
roi_name = r'vHabenula'

# Create output folders (if they do not exist)
if not os.path.exists(analysis_folder):
   os.makedirs(analysis_folder)

cfos_value_Baseline, mean_Baseline, std_Baseline= SPCFOS.load_cfos_ROI(cfos_Baseline)
cfos_value_Social, mean_Social, std_Social= SPCFOS.load_cfos_ROI(cfos_Social)
cfos_value_Noxious, mean_Noxious, std_Noxious= SPCFOS.load_cfos_ROI(cfos_Noxious)



# Plot
bar_colours = [ "#c0c0c0","#ff0000",'#0033ff']


cfos_val = plt.figure(dpi=300)
plt.title(roi_name,fontsize= 14)  

s1 = pd.Series(cfos_value_Baseline, name='Baseline')
s2 = pd.Series(cfos_value_Social, name='Social')
s3 = pd.Series(cfos_value_Noxious, name='Noxious')
df = pd.concat([s1,s2,s3], axis=1)
ax=sns.swarmplot(data=df, orient="v", size=6, palette=bar_colours, zorder=1) 
with plt.rc_context({'lines.linewidth': 0.8}):
    sns.pointplot(data=df, orient="v", linewidth=.1, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)
plt.ylabel("cFos expression")
ax.set(ylim=(0.5, 2))


cfos_val.savefig(analysis_folder + '/vHabenula.png', dpi=300, bbox_inches='tight')
cfos_val.savefig(analysis_folder + '/vHabenula.eps', dpi=300, bbox_inches='tight')



# FIN
