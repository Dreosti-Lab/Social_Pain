#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 16:03:40 2023

@author: alizeekastler
This code use the average position of the fish in the Non Social vs Social condition to perform k-means clustering and identify groups of more or less resilient fish 
The groups are plotted as a scatterplot and proportion of fish in each group is calculated. 
A csv file will also be generated with the folder path to the individual fish and their belonging to the different clusters
"""

#Set Library path 
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

#Set Base path
base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'
#base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib.patches import Ellipse, Circle

# Import local modules
import SP_utilities as SPU

# Specify Analysis folder with summary npz files 
AnalysisFolder = base_path + '/Gradient_Social/Analysis'

# Read folder list
FolderlistFile = base_path + '/Gradient_Social/Folderlist.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(AnalysisFolder+'/*.npz')

# Calculate how many files
numFiles = np.size(npzFiles, 0)
avgPosition_NS_ALL = np.zeros(numFiles)
avgPosition_S_ALL = np.zeros(numFiles)

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):
    
    # Load each npz file
    dataobject = np.load(filename)
    avgPosition_NS = dataobject['avgPosition_NS']
    avgPosition_S = dataobject['avgPosition_S']
    
    avgPosition_NS_ALL[f] = avgPosition_NS
    avgPosition_S_ALL[f] = avgPosition_S

#Plot position NS vs S for each fish 
XM_values = np.column_stack((avgPosition_NS_ALL, avgPosition_S_ALL))
fig = plt.figure(figsize=(10, 6))
x = XM_values[:, 0]
y = XM_values[:, 1]

plt.scatter(x, y, marker=".")
plt.xlabel('Non Social')
plt.ylabel('Social')
plt.title('AvgPosition')
plt.show()


# Perform KMeans clustering 
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10).fit(XM_values)#number of clusters can be manually defined
cluster_labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Map numeric labels to cluster names
cluster_names = {0: 'Low-Low', 1: 'High-High', 2: 'High-Low', 3:'Low-High'}
cluster_colors = {0: 'yellow', 1: 'cadetblue', 2: 'steelblue', 3: 'lightgreen'}
clusters = [cluster_names[label] for label in cluster_labels]


# Plot output clusters on top of scatterplot
fig, ax = plt.subplots(figsize=(10, 8))

# Loop through each cluster
for i, center in enumerate(centers):
    # Draw circles around cluster centers
    circle = Circle((center[1], center[0]), radius=20, color='whitesmoke', fill=True, zorder=1)
    ax.add_patch(circle)
    # Select data points belonging to the current cluster
    cluster_members = XM_values[cluster_labels == i]
    # Plot data points for the current cluster with cluster name as label
    ax.scatter(
        cluster_members[:, 1], cluster_members[:, 0],
        label=cluster_names[i], s=90, color=cluster_colors[i], zorder=2
    )

# Plot cluster centers
ax.scatter(centers[:, 1], centers[:, 0], marker='o', s=200, color='black', alpha = 0.7, zorder=3)
ax.set_xlabel('SC', fontsize=30, fontname='Arial')
ax.set_ylabel('No SC', fontsize=30, fontname='Arial')
ax.set_title('KMeans Clustering', fontsize=34, fontname='Arial')
ax.legend(prop={'size': 20, 'family': 'Arial'})

# Increase marker size and set tick/label sizes
ax.tick_params(axis='both', which='major', labelsize=26)
ax.tick_params(axis='both', which='minor', labelsize=26)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(2.5)
ax.spines['left'].set_linewidth(2.5)

plt.savefig(base_path + '/Gradient_Social/figures/clusters.eps', dpi=300, bbox_inches='tight')


# Plot pie chart showing the proportion of fish in each cluster
fig_pie, ax_pie = plt.subplots(figsize=(4, 4))

cluster_counts = np.bincount(cluster_labels)
ax_pie.pie(cluster_counts, labels=cluster_names.values(), colors=cluster_colors.values(), autopct='%1.1f%%', startangle=90)
ax_pie.set_title('Cluster Proportions', fontsize=30, fontname='Arial')

plt.savefig(base_path + '/Gradient_Social/figures/cluster_pie_chart.eps', dpi=300, bbox_inches='tight')


# Assign folders to clusters based on fishStatus and cluster labels
fishStatus = fishStatus * [1,2,3,4,5,6]    
datapoints = []

for idx, (status, path) in enumerate(zip(fishStatus, folderNames)):
    for status in status:
        if status > 0:
            datapoints.append((path, status))

df = pd.DataFrame(datapoints, columns=['FolderPath', 'FishStatus'])
df = df.assign(ClusterLabel=clusters)
# Save DataFrame to a excel file
df.to_excel(base_path + '/Gradient_Social/fish_clusters.xlsx', index=False)
df.to_csv(base_path + '/Gradient_Social/fish_clusters.csv', index=False)

#FIN