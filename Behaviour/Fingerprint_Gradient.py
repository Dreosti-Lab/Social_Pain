#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 18:08:47 2023

@author: alizeekastler
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

conditions=[]
conditions.append(r'L368_100uM_Gradient')
conditions.append(r'L368_100uM_Gradient_Social')
#conditions.append(r'Gradient')
#conditions.append(r'Re-exposure')
#conditions.append(r'Heat_Gradient')
#conditionNames.append(r'Resilient')
#conditionNames.append(r'Susceptible')
#conditions.append(r'RES+')
#conditions.append(r'RES-')
#conditions.append(r'Gradient_Social')
# conditions.append(r'Gradient_Social_Resilient')
# conditions.append(r'Gradient_Social_Susceptible')
#conditionNames.append(r'Gradient_Isolated')

analysisFolders=[]

for condition in conditions:
    analysisFolders.append(base_path + r'/' + condition + '/Analysis')

BPS_NS_summary = []
avgdistPerBout_NS_summary = []
avgSpeedPerBout_NS_summary = []
DistanceT_NS_summary = []
avgBout_interval_NS_summary= []
numFreezes_NS_summary = []
Percent_Moving_NS_summary = []
Percent_Pausing_NS_summary= []


SummaryAll = []
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
        #FSwim_NS = dataobject['FSwim_S']
        Turn_NS = dataobject['Turns_S']
        numFreezes_NS = dataobject['numFreezes_S']
        #Percent_Moving_NS = dataobject['Percent_Moving_S']
        #Percent_Pausing_NS = dataobject['Percent_Paused_S']
        
            
        BPS_NS_ALL.append(BPS_NS)
        avgdistPerBout_NS_ALL.append(avgdistPerBout_NS)
        avgSpeedPerBout_NS_ALL.append(avgSpeedPerBout_NS)
        avgBout_interval_NS_ALL.append(avgBout_interval_NS)
        #FSwim_NS_ALL.append(FSwim_NS)
        Turn_NS_ALL.append(Turn_NS)
        numFreezes_NS_ALL.append(numFreezes_NS)
        #Percent_Moving_NS_ALL.append(Percent_Moving_NS)
        #Percent_Pausing_NS_ALL.append(Percent_Pausing_NS)
            
      
        SummaryAll.append([conditions[i],float(BPS_NS), float(avgdistPerBout_NS),float(avgSpeedPerBout_NS), float(Turn_NS),float(avgBout_interval_NS), float(numFreezes_NS)])
    
    BPS_NS_summary.append(BPS_NS_ALL)
    avgdistPerBout_NS_summary.append(avgdistPerBout_NS_ALL)
    avgSpeedPerBout_NS_summary.append(avgSpeedPerBout_NS_ALL)
    avgBout_interval_NS_summary.append(avgBout_interval_NS_ALL)
    numFreezes_NS_summary.append(numFreezes_NS_ALL)
    Percent_Moving_NS_summary.append(Percent_Moving_NS_ALL)
    Percent_Pausing_NS_summary.append(Percent_Pausing_NS_ALL)


dfAll=pd.DataFrame(SummaryAll,columns=['Condition','BPS','Distance','Speed', 'Turn','Interval', 'Freezes'])



#Calculate zscore

df_zscore=dfAll.copy()

# extract Baseline ( no stimuli)
numeric_cols = dfAll.select_dtypes(include=[np.number]).columns 
Baseline = dfAll.loc[(dfAll['Condition']=='Baseline')]
# compute mean and SD
BMean=Baseline.mean()
Bstd=Baseline.std()
# normalise dataframe
for col in numeric_cols:
    df_zscore[col]=(dfAll[col]-BMean[col])/Bstd[col]
    

# perform mannwhitney with bonferroni correction
def mann_whitney_all_vs_Bas(df):
    """
    Perform Mann-Whitney U tests comparing each condition to the Baseline
    
    Parameters:
        df (pandas.DataFrame): Input dataframe containing the behavioral data.

    Returns:
        pandas.DataFrame: Dataframe containing the p-values and FDR-corrected p-values for each condition and column.
    """
    # Get list of conditions
    Conditions = df["Condition"].unique().tolist()

    # Remove "Baseline" from list 
    Conditions.remove("Baseline")

    # Create empty dataframe to store results
    stats_df = pd.DataFrame()

    # Perform Mann-Whitney U test for each genotype compared to "Scrambled" for all columns
    for col_name in df.columns:
        if col_name == "Condition":
            continue
        for Condition in Conditions:
            group1 = df.loc[df["Condition"] == "Baseline", col_name]
            group2 = df.loc[df["Condition"] == Condition, col_name]
            _, pvalue = mannwhitneyu(group1, group2, alternative="two-sided")
            stats_df = stats_df.append({"Condition": Condition, "Column": col_name, "pvalue": pvalue}, ignore_index=True)

    # Correct p-values for multiple comparisons using FDR
    stats_df["pvalue_corr"] = fdrcorrection(stats_df["pvalue"])[1]

    return stats_df

MannW_results_all = mann_whitney_all_vs_Bas(dfAll)


#-----------------------------------------------------------------------------


#plot all behaviours as a heatmap
df_MeanZ=df_zscore.groupby(['Condition'], sort=False).mean()


fingerprint =plt.figure(figsize= (14,5))

df_MeanZ_exBas=df_MeanZ.drop('Baseline')

ax=sns.heatmap(df_MeanZ_exBas,vmin=-1,vmax=1,cmap='PuOr', linewidth=0.5, square = True)#, cbar_kws={'shrink':0.8})
ax.set(xlabel='', ylabel='')
ax.xaxis.tick_top()
plt.xticks(rotation=20, fontsize = 18)
plt.yticks(rotation= 360, fontsize=18)


df_pivot = pd.pivot_table(MannW_results_all, values='pvalue', index=['Condition'], columns=['Column'], sort=False)
df_pivot = df_pivot[['BoutsPerSecond', 'distPerBout', 'distTravelled','Turn', 'Freezes']]

for i in range(df_MeanZ_exBas.shape[0]):
    for j in range(df_MeanZ_exBas.shape[1]):
        value = df_pivot.iloc[i, j]
        if value < 0.001:
            ax.text(j+0.5, i+0.5, "p<0.001", ha="center", va="center", color='black', fontsize =14)
        elif value < 0.01:
            ax.text(j+0.5, i+0.5, "p<0.01", ha="center", va="center", color='black', fontsize=14)
        elif value < 0.05:
            ax.text(j+0.5, i+0.5, "p<0.05", ha="center", va="center", color='black', fontsize=14)
  

fingerprint.savefig(base_path + '/test_l368.eps', format='eps', dpi=300,bbox_inches= 'tight')
#fingerprint.savefig(base_path + '/RE-Exposed.png', dpi=300, bbox_inches='tight')
              
