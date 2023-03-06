"""
Create summary (figures and report) for all analyzed fish in a social preference experiment
@author: kampff
"""
# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'\
    
import sys
sys.path.append(lib_path)

base_path = r'/Volumes/T7 Touch/Behaviour_Heat_Gradient'    


# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats

import glob


## Local Functions
## ------------------------------------------------------------------------
def fill_bouts(bouts, FPS):
    moving_frames = np.zeros(90000)
    for bout in bouts:
        start  = np.int(bout[0])
        stop = np.int(bout[4])
        moving_frames[start:stop] = 1
        moving_frames = moving_frames[:90000]
    
    return moving_frames

def bin_frames(frames, FPS):
    bin_size = 60 * FPS
    reshaped_frames = np.reshape(frames, (bin_size, -1), order='F')
    bins = np.sum(reshaped_frames, 0) / bin_size
    return bins*100

def load_and_process_npz(npzFiles):
    # Analysis Settings

    # Calculate how many files
    numFiles = np.size(npzFiles, 0)

    # Allocate space for summary data
    PTM_NS_BINS = np.zeros((numFiles,15))

    # Go through all the files contained in the analysis folder
    for f, filename in enumerate(npzFiles):

        # Load each npz file
        dataobject = np.load(filename)
        
        # Extract from the npz file
        Bouts_NS = dataobject['Bouts_NS']     
        
        FPS = 100
        
        
        # Compute percent time freezing in one minute bins
        moving_frames_NS = fill_bouts(Bouts_NS, FPS)
        PTM_NS_BINS[f] = bin_frames(moving_frames_NS, FPS)

    return PTM_NS_BINS


## ------------------------------------------------------------------------

# Specify save folder
figureFolder = base_path + r'/Figure_Nox'

# Specify Analysis folder
analysisFolder_Heat = base_path + r'/Noxious++/Analysis'
analysisFolder_50 = base_path + r'/MustardOil/50uM/Analysis'
analysisFolder_500 = base_path + r'/MustardOil/500uM/Analysis'

# Find all the npz files saved for each group and fish with all the information
npzFiles_Heat = glob.glob(analysisFolder_Heat + '/*.npz')
npzFiles_50 = glob.glob(analysisFolder_50 + '/*.npz')
npzFiles_500 = glob.glob(analysisFolder_500 + '/*.npz')

# LOAD CONTROLS
PTM_NS_BINS_Heat = load_and_process_npz(npzFiles_Heat)

# LOAD ISO
PTM_NS_BINS_50 = load_and_process_npz(npzFiles_50)

# LOAD DRUGGED
PTM_NS_BINS_500 = load_and_process_npz(npzFiles_500)

# ----------------------------------
# Plots
# ----------------------------------

# PTM binned 
plt.figure(figsize=(10.24,7.68))
plt.title("Percent Time Moving (one minute bins)")

m = np.nanmean(PTM_NS_BINS_Heat, 0)
std = np.nanstd(PTM_NS_BINS_Heat, 0)
valid = (np.logical_not(np.isnan(PTM_NS_BINS_Heat)))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'purple', LineWidth=4)
plt.plot(m, 'purple',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'purple', LineWidth=1)
plt.plot(m-se, 'purple', LineWidth=1)

m = np.nanmean(PTM_NS_BINS_50, 0)
std = np.nanstd(PTM_NS_BINS_50, 0)
valid = (np.logical_not(np.isnan(PTM_NS_BINS_50)))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'lightcoral', LineWidth=4)
plt.plot(m, 'lightcoral', Marker='o', MarkerSize=7)
plt.plot(m+se, 'lightcoral', LineWidth=1)
plt.plot(m-se, 'lightcoral', LineWidth=1)

m = np.nanmean(PTM_NS_BINS_500, 0)
std = np.nanstd(PTM_NS_BINS_500, 0)
valid = (np.logical_not(np.isnan(PTM_NS_BINS_500)))
n = np.sum(valid, 0)
se = std/np.sqrt(n-1)
plt.plot(m, 'tomato', LineWidth=4)
plt.plot(m, 'tomato',Marker = 'o', MarkerSize=7)
plt.plot(m+se, 'tomato', LineWidth=1)
plt.plot(m-se, 'tomato', LineWidth=1)

#plt.axis([0, 14, 0.0, 0.02])
plt.xlabel('minutes')
plt.ylabel('% Moving')

plt.tight_layout() 
filename = figureFolder +'/Figure_4B_ptm.png'
plt.savefig(filename, dpi=600)


# FIN
