# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:05:33 2023

@author: Alizee Kastler
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 08:31:26 2023

@author: Alizee Kastler
"""

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


cfos_data_folder = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Analysis/ROI'



B_PT = cfos_data_folder + r'/SERT_PT/Baseline_cfos.npz'
N_PT = cfos_data_folder + r'/SERT_PT/Noxious_cfos.npz'

B_aPVO = cfos_data_folder + r'/SERT_hyp_3/Baseline_cfos.npz'
N_aPVO = cfos_data_folder + r'/SERT_hyp_3/Noxious_cfos.npz'

B_pPVO = cfos_data_folder + r'/SERT_hyp_2/Baseline_cfos.npz'
N_pPVO = cfos_data_folder + r'/SERT_hyp_2/Noxious_cfos.npz'

B_cH = cfos_data_folder + r'/SERT_hyp_1/Baseline_cfos.npz'
N_cH = cfos_data_folder + r'/SERT_hyp_1/Noxious_cfos.npz'



cfos_value_B_PT, mean_B_PT, std_B_PT= SPCFOS.load_cfos_ROI(B_PT)
cfos_value_N_PT, mean_N_PT, std_N_PT= SPCFOS.load_cfos_ROI(N_PT)
cfos_value_B_aPVO, mean_B_aPVO, std_B_aPVO= SPCFOS.load_cfos_ROI(B_aPVO)
cfos_value_N_aPVO, mean_N_aPVO, std_N_aPVO= SPCFOS.load_cfos_ROI(N_aPVO)
cfos_value_B_pPVO, mean_B_pPVO, std_B_pPVO= SPCFOS.load_cfos_ROI(B_pPVO)
cfos_value_N_pPVO, mean_N_pPVO, std_N_pPVO= SPCFOS.load_cfos_ROI(N_pPVO)
cfos_value_B_cH, mean_cH, std_B_cH= SPCFOS.load_cfos_ROI(B_cH)
cfos_value_N_cH, mean_cH, std_N_cH= SPCFOS.load_cfos_ROI(N_cH)

# Plot
bar_colours = [ "#c0c0c0",'#0033ff']

cfos_val = plt.figure(dpi=300)
plt.title('Serotonin',fontsize= 14)  

s1 = pd.Series(cfos_value_B_PT, name='PT_')
s2 = pd.Series(cfos_value_N_PT, name='PT')
s3 = pd.Series(cfos_value_B_aPVO, name='aPVO_')
s4 = pd.Series(cfos_value_N_aPVO, name='aPVO')
s5 = pd.Series(cfos_value_B_pPVO, name='pPVO_')
s6 = pd.Series(cfos_value_N_pPVO, name='pPVO')
s7 = pd.Series(cfos_value_B_cH, name='cH_')
s8 = pd.Series(cfos_value_N_cH, name='cH')
df = pd.concat([s1,s2,s3,s4,s5,s6,s7,s8], axis=1)
# ax=sns.swarmplot(data=df, orient="v", size=6, palette = bar_colours, zorder=1) 
# with plt.rc_context({'lines.linewidth': 0.8}):
#     sns.pointplot(data=df, orient="v", linewidth=.1, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)

ax= sns.barplot(data=df, palette = bar_colours, zorder=1)
plt.ylabel("cFos expression")

cfos_val.savefig(cfos_data_folder + '/cfos_Serotonin.png', dpi=300, bbox_inches='tight')