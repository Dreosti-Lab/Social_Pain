# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 12:58:43 2021

@author: Alizee Kastler
"""
# Set Library Path - Social_Pain Repos
#lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)

# Set Base Path
#base_path = r'/Users/alizeekastler/Desktop'
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'

# Import useful libraries
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Import local modules

import SP_utilities as SPU
import SP_Analysis as SPA
import BONSAI_ARK


FolderlistFile = base_path + '/Folderlist_New.txt' 
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Analyze temporal bouts
def analyze_temporal_bouts(bouts, binning):

    # Determine total bout counts
    num_bouts = bouts.shape[0]

    # Determine largest frame number in all bouts recordings (make multiple of 100)
    max_frame = np.int(np.max(bouts[:, 4]))
    max_frame = max_frame + (binning - (max_frame % binning))
    max_frame = 100 * 60 * 15 # 15 minutes

    # Temporal bouts
    visible_bout_hist = np.zeros(max_frame)
    non_visible_bout_hist = np.zeros(max_frame)
    frames_moving = 0
    visible_frames_moving = 0
    non_visible_frames_moving = 0
    for i in range(0, num_bouts):
        # Extract bout params
        start = np.int(bouts[i][0])
        stop = np.int(bouts[i][4])
        duration = np.int(bouts[i][8])
        visible = np.int(bouts[i][9])

        # Ignore bouts beyond 15 minutes
        if stop >= max_frame:
            continue

        # Accumulate bouts in histogram
        if visible == 1:
            visible_bout_hist[start:stop] = visible_bout_hist[start:stop] + 1
            visible_frames_moving += duration
        else:
            non_visible_bout_hist[start:stop] = non_visible_bout_hist[start:stop] + 1
            non_visible_frames_moving += duration
        frames_moving += duration

    #plt.figure()
    #plt.plot(visible_bout_hist, 'b')
    #plt.plot(non_visible_bout_hist, 'r')
    #plt.show()

    # Bin bout histograms
    visible_bout_hist_binned = np.sum(np.reshape(visible_bout_hist.T, (binning, -1), order='F'), 0)
    non_visible_bout_hist_binned = np.sum(np.reshape(non_visible_bout_hist.T, (binning, -1), order='F'), 0)

    #plt.figure()
    #plt.plot(visible_bout_hist_binned, 'b')
    #plt.plot(non_visible_bout_hist_binned, 'r')
    #plt.show()

    # Compute Ratio
    total_bout_hist_binned = visible_bout_hist_binned + non_visible_bout_hist_binned
    vis_vs_non = (visible_bout_hist_binned - non_visible_bout_hist_binned) / total_bout_hist_binned

    # Normalize bout histograms
    #visible_bout_hist_binned = visible_bout_hist_binned / frames_moving
    #non_visible_bout_hist_binned = non_visible_bout_hist_binned / frames_moving
    #vis_v_non = visible_bout_hist_binned / non_visible_bout_hist_binned

    # ----------------
    # Temporal Bouts Summary Plot
    #plt.figure()
    #plt.plot(vis_vs_non, 'k')
    #plt.ylabel('VPI')
    #plt.xlabel('minutes')
    #plt.show()

    return vis_vs_non