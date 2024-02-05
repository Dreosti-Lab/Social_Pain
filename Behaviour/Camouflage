#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:24:30 2023

@author: alizeekastler
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Define your pre and post-processing average gray values for the baseline condition
baseline_pre = [65.6, 80.9, 60.2, 58.5,60]
baseline_post = [66.4, 79.9, 61.2, 61.6,62]

# Define your pre and post-processing average gray values for the heat condition
heat_pre = [66.1, 87.5, 55.9, 71.0]
heat_post = [58.5, 74.0, 41.9, 66.4]

# Define your pre and post-processing average gray values for the AITC condition
AITC_pre = [73.3,79,49.7,57.2,51.5,50.9,71.8]
AITC_post = [42.8,46.4,37.5,30.9,43.8,45.3,53.2]

# Calculate the diff in mean gray value
heat_diff = np.array(heat_pre) - np.array(heat_post)
baseline_diff = np.array(baseline_pre) - np.array(baseline_post)
AITC_diff = np.array(AITC_pre) - np.array(AITC_post)
data = [baseline_diff, heat_diff, AITC_diff]


plt.figure(figsize=(4, 10))
ax=sns.boxplot(data = data,color ='whitesmoke', linewidth =4, showfliers=False)
ax=sns.stripplot(data = data, palette = ['#aaa4c8', '#452775','#d85c1a'], size=12, jitter=True, edgecolor="gray")
ax.set_xticklabels(['Baseline', 'Heat', 'AITC'],rotation=32, fontsize = 28, fontname = 'Arial')
ax.spines['bottom'].set_linewidth(2.5)  
ax.spines['left'].set_linewidth(2.5) 
# Add labels and a title
plt.ylabel('Δ Mean Gray Value (post-pre)', fontsize = 32, fontname='Arial')
plt.xticks(fontsize=32, fontname = 'Arial')
plt.yticks(fontsize=32, fontname='Arial')
plt.ylim(-2,35)

# Remove the top and right spines (outlines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Show the plot
plt.show()




