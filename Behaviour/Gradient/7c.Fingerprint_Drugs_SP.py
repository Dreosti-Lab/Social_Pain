# -*- coding: utf-8 -*-
"""
Compare summaries of analyzed social preference experiments

@author: Tom Ryan, UCL (Dreosti-Group) 
"""
# -----------------------------------------------------------------------------
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'
#base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'



# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import fdrcorrection

import glob

conditionNames=[]
conditionNames.append(r'Old_Social')
conditionNames.append(r'Old_Gradient')
conditionNames.append(r'L368_100uM_Social')
conditionNames.append(r'L368_100uM_Gradient')
# conditionNames.append(r'DMSO')


analysisFolders=[]
# Set analysis folder and label for experiment/condition A
for condition in conditionNames:
    analysisFolders.append(base_path + r'/' + condition + '/Analysis')

BPS_NS_summary = []
avgdistPerBout_NS_summary = []
avgSpeedPerBout_NS_summary = []
DistanceT_NS_summary = []
avgBout_interval_NS_summary= []
numFreezes_NS_summary = []
Percent_Moving_NS_summary = []
Percent_Pausing_NS_summary= []


AllFish = []

for i, analysisFolder in enumerate(analysisFolders):
    
   
    npzFiles = glob.glob(analysisFolder+'/*.npz')
    
    # Calculate how many files
    numFiles = np.size(npzFiles, 0)
    
    BPS_NS_ALL = []
    avgdistPerBout_NS_ALL = []
    avgSpeedPerBout_NS_ALL= []
    DistanceT_NS_ALL = []
    FSwim_NS_ALL = []
    Turn_NS_ALL = []
    avgBout_interval_NS_ALL=[]
    numFreezes_NS_ALL = [] 
    Percent_Moving_NS_ALL = []
    Percent_Pausing_NS_ALL = []
    
    
    for f, filename in enumerate(npzFiles):
        
        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        BPS_NS = dataobject['BPS_S']  
        avgdistPerBout_NS = dataobject['avgdistPerBout_S']
        avgSpeedPerBout_NS = dataobject['avgSpeedPerBout_S']
        avgBout_interval_NS = dataobject['avgBout_interval_S']
        #DistanceT_NS = dataobject['DistanceT_NS'] 
        #FSwim_NS = dataobject['FSwim_NS']
        Turn_NS = dataobject['Turns_S']
        numFreezes_NS = dataobject['numFreezes_S']
        #Percent_Moving_NS = dataobject['Percent_Moving_NS']
        #Percent_Pausing_NS = dataobject['Percent_Paused_NS']
        
            
        BPS_NS_ALL.append(BPS_NS)
        avgdistPerBout_NS_ALL.append(avgdistPerBout_NS)
        avgSpeedPerBout_NS_ALL.append(avgSpeedPerBout_NS)
        avgBout_interval_NS_ALL.append(avgBout_interval_NS)
        #DistanceT_NS_ALL.append(DistanceT_NS)
        #FSwim_NS_ALL.append(FSwim_NS)
        Turn_NS_ALL.append(Turn_NS)
        numFreezes_NS_ALL.append(numFreezes_NS)
        #Percent_Moving_NS_ALL.append(Percent_Moving_NS)
        #Percent_Pausing_NS_ALL.append(Percent_Pausing_NS)
            
      
        AllFish.append([conditionNames[i],float(BPS_NS), float(avgdistPerBout_NS),float(avgSpeedPerBout_NS), float(Turn_NS),float(avgBout_interval_NS), float(numFreezes_NS)])
        
    
    BPS_NS_summary.append(BPS_NS_ALL)
    avgdistPerBout_NS_summary.append(avgdistPerBout_NS_ALL)
    avgSpeedPerBout_NS_summary.append(avgSpeedPerBout_NS_ALL)
    avgBout_interval_NS_summary.append(avgBout_interval_NS_ALL)
    DistanceT_NS_summary.append(DistanceT_NS_ALL)
    numFreezes_NS_summary.append(numFreezes_NS_ALL)
    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Pausing_NS_summary.append(Percent_Pausing_NS_ALL)


    # Calculate z-score
    df_All = pd.DataFrame(AllFish, columns=['Condition', 'BPS', 'Distance', 'Speed', 'Turn', 'Interval', 'Freezes'])


# Extract data for relevant conditions
social_condition = df_All[df_All['Condition'] == 'Old_Social']
L368_social_condition = df_All[df_All['Condition'] == 'L368_100uM_Social']
gradient_condition = df_All[df_All['Condition'] == 'Old_Gradient']
L368_gradient_condition = df_All[df_All['Condition'] == 'L368_100uM_Gradient']

# Select relevant swim kinematics columns
kinematics_columns = ['BPS', 'Distance', 'Speed', 'Turn', 'Interval', 'Freezes']

# Calculate z-scores for L368_100uM_Social vs Social
z_scores_l368_vs_social = (L368_social_condition[kinematics_columns] - social_condition[kinematics_columns].mean()) / social_condition[kinematics_columns].std()

# Calculate z-scores for L368_100uM_Gradient vs Gradient
z_scores_l368_vs_gradient = (L368_gradient_condition[kinematics_columns] - gradient_condition[kinematics_columns].mean()) / gradient_condition[kinematics_columns].std()

# Perform Mann-Whitney U tests and store p-values
p_values_l368_vs_social = [mannwhitneyu(L368_social_condition[col], social_condition[col]).pvalue for col in kinematics_columns]
p_values_l368_vs_gradient = [mannwhitneyu(L368_gradient_condition[col], gradient_condition[col]).pvalue for col in kinematics_columns]

# Combine the z-scores and p-values into a DataFrame
df_zscore = pd.DataFrame({'L368,899_Social': z_scores_l368_vs_social.mean(),
                            'L368,899_Gradient': z_scores_l368_vs_gradient.mean()})
df_Pvalues= pd.DataFrame({'p-value (L368,899_Social)': p_values_l368_vs_social,
                            'p-value (L368,899_Gradient)': p_values_l368_vs_gradient})



# Plot heatmap
fingerprint =plt.figure(figsize=(10, 6))
ax=sns.heatmap(df_zscore.T, annot=False, cmap="coolwarm", fmt=".2f", vmin = -2, vmax=2,linewidth=2, square=True )
ax.set(xlabel='', ylabel='')
ax.xaxis.tick_top()
plt.xticks(rotation=20,fontsize =16, fontname='Arial')
plt.yticks(rotation= 360, fontsize=16, fontname='Arial')
plt.xticks(rotation=20,fontsize =16, fontname='Arial')


for i in range(df_zscore.shape[1]):
    for j in range(df_zscore.shape[0]):
        value = df_Pvalues.T.iloc[i, j]
        if value < 0.001:
            ax.text(j+0.5, i+0.5, "p<0.001", ha="center", va="center", color='black', fontsize = 10)
        elif value < 0.01:
            ax.text(j+0.5, i+0.5, "p<0.01", ha="center", va="center", color='black', fontsize=10)
        elif value < 0.05:
            ax.text(j+0.5, i+0.5, "p<0.05", ha="center", va="center", color='black', fontsize=10)
            

fingerprint.savefig(base_path + '/Gradient_Social/Figures/fingerprint_2.eps', format='eps', dpi=300,bbox_inches= 'tight')           
                
