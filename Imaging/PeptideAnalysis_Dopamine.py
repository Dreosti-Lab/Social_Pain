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

B_TH1_SP = cfos_data_folder + r'/TH1_SP/Baseline_cfos.npz'
N_TH1_SP = cfos_data_folder + r'/TH1_SP/Noxious_cfos.npz'

B_TH1_PvPp = cfos_data_folder + r'/TH1_PvPp/Baseline_cfos.npz'
N_TH1_PvPp = cfos_data_folder + r'/TH1_PvPp/Noxious_cfos.npz'

B_TH1_PVO = cfos_data_folder + r'/TH1_PVO/Baseline_cfos.npz'
N_TH1_PVO = cfos_data_folder + r'/TH1_PVO/Noxious_cfos.npz'

B_TH2_PT = cfos_data_folder + r'/TH2_PT/Baseline_cfos.npz'
N_TH2_PT = cfos_data_folder + r'/TH2_PT/Noxious_cfos.npz'

B_TH2_LR = cfos_data_folder + r'/TH2_LR/Baseline_cfos.npz'
N_TH2_LR = cfos_data_folder + r'/TH2_LR/Noxious_cfos.npz'

B_TH1_cH = cfos_data_folder + r'/TH1_cH/Baseline_cfos.npz'
N_TH1_cH = cfos_data_folder + r'/TH1_cH/Noxious_cfos.npz'



cfos_value_TH1_SP_B, mean_TH1_SP_B, std_TH1_SP_B= SPCFOS.load_cfos_ROI(B_TH1_SP)
cfos_value_TH1_SP_N, mean_TH1_SP_N, std_TH1_SP_N= SPCFOS.load_cfos_ROI(N_TH1_SP)
cfos_value_TH1_PvPp_B, mean_TH1_PvPp_B, std_TH1_PvPp_B= SPCFOS.load_cfos_ROI(B_TH1_PvPp)
cfos_value_TH1_PvPp_N, mean_TH1_PvPp_N, std_TH1_PvPp_N= SPCFOS.load_cfos_ROI(N_TH1_PvPp)
cfos_value_TH1_PVO_B, mean_TH1_PVO_B, std_TH1_PVO_B= SPCFOS.load_cfos_ROI(B_TH1_PVO)
cfos_value_TH1_PVO_N, mean_TH1_PVO_N, std_TH1_PVO_N= SPCFOS.load_cfos_ROI(N_TH1_PVO)
cfos_value_TH2_PT_B, mean_TH2_PT_B, std_TH2_PT_B= SPCFOS.load_cfos_ROI(B_TH2_PT)
cfos_value_TH2_PT_N, mean_TH2_PT_N, std_TH2_PT_N= SPCFOS.load_cfos_ROI(N_TH2_PT)
cfos_value_TH2_LR_B, mean_TH2_LR_B, std_TH2_LR_B= SPCFOS.load_cfos_ROI(B_TH2_LR)
cfos_value_TH2_LR_N, mean_TH2_LR_N, std_TH2_LR_N= SPCFOS.load_cfos_ROI(N_TH2_LR)
cfos_value_TH1_cH_B, mean_TH1_cH_B, std_TH1_cH_B= SPCFOS.load_cfos_ROI(B_TH1_cH)
cfos_value_TH1_cH_N, mean_TH1_cH_N, std_TH1_cH_N= SPCFOS.load_cfos_ROI(N_TH1_cH)



# Plot
bar_colours = [ "#c0c0c0",'#0033ff']

cfos_val = plt.figure(dpi=300)
plt.title('Dopamine',fontsize= 14)  

s1 = pd.Series(cfos_value_TH1_SP_B, name='SP_')
s2 = pd.Series(cfos_value_TH1_SP_N, name='SP')
s3 = pd.Series(cfos_value_TH1_PvPp_B, name='PvPp_')
s4 = pd.Series(cfos_value_TH1_PvPp_N, name='PvPp')
s5 = pd.Series(cfos_value_TH1_PVO_B, name='PVO_')
s6 = pd.Series(cfos_value_TH1_PVO_N, name='PVO')
s7 = pd.Series(cfos_value_TH2_PT_B, name='PT_')
s8 = pd.Series(cfos_value_TH2_PT_N, name='PT')
s9 = pd.Series(cfos_value_TH2_LR_B, name='LR_')
s10 = pd.Series(cfos_value_TH2_LR_N, name='LR')
s11 = pd.Series(cfos_value_TH1_cH_B, name='cH_')
s12 = pd.Series(cfos_value_TH1_cH_N, name='cH')
df = pd.concat([s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12], axis=1)
# ax=sns.swarmplot(data=df, orient="v", size=6, palette = bar_colours, zorder=1) 
# with plt.rc_context({'lines.linewidth': 0.8}):
#     sns.pointplot(data=df, orient="v", linewidth=.1, ci=68, capsize=0.4, join=False, color="#444444", zorder=100)
# plt.ylabel("cFos expression")


ax= sns.barplot(data=df, palette = bar_colours, zorder=1)
plt.ylabel("cFos expression")


cfos_val.savefig(cfos_data_folder + '/cfos_Dopamine.png', dpi=300, bbox_inches='tight')