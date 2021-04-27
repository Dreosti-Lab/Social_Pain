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
    max_frame = 100 * 60 * 10 # 10 minutes

    # Temporal bouts
    Temporal_bout_hist = np.zeros(max_frame)
    frames_moving = 0

    for i in range(0, num_bouts):
        # Extract bout params
        start = np.int(bouts[i][0])
        stop = np.int(bouts[i][4])
        duration = np.int(bouts[i][8])

        # Ignore bouts beyond 15 minutes
        if stop >= max_frame:
            continue

        # Accumulate bouts in histogram
        if start == 1:
            Temporal_bout_hist[start:stop] = Temporal_bout_hist[start:stop] + 1
            frames_moving += duration  
    
    plt.figure()
    plt.plot(Temporal_bout_hist, 'b')        
    
    # Bin bout histograms
    bout_hist_binned = np.sum(np.reshape(Temporal_bout_hist.T, (binning, -1), order='F'), 0)
    
    plt.figure()
    plt.plot(bout_hist_binned, 'b')
    plt.ylabel('bouts')
    plt.xlabel('minutes')
    plt.show()
    
    return bout_hist_binned

def computeVPI(bouts,FPS):
     
    bin_size = 60 * FPS
    max_frame = bin_size * 10

    binned_bouts = np.sum(np.reshape(bouts[:max_frame].T, (bin_size, -1), order='F'), 0)
    Temporal_bouts = (binned_bouts)/bin_size
   

    plt.figure()
    plt.plot(Temporal_bouts, 'b')
    plt.ylabel('bouts')
    plt.xlabel('minutes')
    plt.show()
    

    return Temporal_bouts

def bouts(bouts, FPS):
    
    moving_frames = np.zeros(60000)
    for bout in bouts:
        start  = np.int(bout[0])
        stop = np.int(bout[4])
        moving_frames[start:stop] = 1
        moving_frames = moving_frames[:60000]
        
        bin_size = 60 * FPS
        max_frame = bin_size * 10
        binned_frames = np.reshape(moving_frames, (bin_size, -1), order='F')
        bins = np.sum(binned_frames, 0)/bin_size
    
    return moving_frames, bins



# Allocate space for summary data
PTM_NS_BINS = np.zeros((numFiles,15))
PTM_S_BINS = np.zeros((numFiles,15))

# Go through all the files contained in the analysis folder
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file
    Bouts_NS = dataobject['Bouts_NS']    
    Bouts_S = dataobject['Bouts_S']   
    Pauses_NS = dataobject['Pauses_NS']    
    Pauses_S = dataobject['Pauses_S']

    # Guess FPS
    last_frame_NS = Pauses_NS[-1,4]
    last_frame_S = Pauses_S[-1,4]
    last_frame = (last_frame_NS + last_frame_S) / 2
    if(last_frame < 107000):
        FPS = 100
    else:
        FPS = 120

    # Compute percent time freezing in one minute bins
    moving_frames_NS = fill_bouts(Bouts_NS, FPS)
    moving_frames_S = fill_bouts(Bouts_S, FPS)
    PTM_NS_BINS[f] = bin_frames(moving_frames_NS, FPS)
    PTM_S_BINS[f] = bin_frames(moving_frames_S, FPS)

return PTM_NS_BINS, PTM_S_BINS