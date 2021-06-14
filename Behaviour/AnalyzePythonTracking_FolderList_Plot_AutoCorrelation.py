# -*- coding: utf-8 -*-
"""
Created on Sun May 11 14:01:46 2014

@author: kampff
"""

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Functions to find Dropbox Folder on each computer
def _get_appdata_path():
    import ctypes
    from ctypes import wintypes, windll
    CSIDL_APPDATA = 26
    _SHGetFolderPath = windll.shell32.SHGetFolderPathW
    _SHGetFolderPath.argtypes = [wintypes.HWND,
                                 ctypes.c_int,
                                 wintypes.HANDLE,
                                 wintypes.DWORD,
                                 wintypes.LPCWSTR]
    path_buf = wintypes.create_unicode_buffer(wintypes.MAX_PATH)
    result = _SHGetFolderPath(0, CSIDL_APPDATA, 0, 0, path_buf)
    return path_buf.value

def dropbox_home():
    from platform import system
    import base64
    import os.path
    _system = system()
    if _system in ('Windows', 'cli'):
        host_db_path = os.path.join(_get_appdata_path(),
                                    'Dropbox',
                                    'host.db')
    elif _system in ('Linux', 'Darwin'):
        host_db_path = os.path.expanduser('~'
                                          '/.dropbox'
                                          '/host.db')
    else:
        raise RuntimeError('Unknown system={}'
                           .format(_system))
    if not os.path.exists(host_db_path):
        raise RuntimeError("Config path={} doesn't exists"
                           .format(host_db_path))
    with open(host_db_path, 'r') as f:
        data = f.read().split()

    return base64.b64decode(data[1])

# -----------------------------------------------------------------------------
# Set Base Path (Shared Dropbox Folder)
base_path = dropbox_home()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
#Function to load all the SPI_NS and SPI_S and to make summary figure
# -----------------------------------------------------------------------------


#Import libraries
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import glob
import pylab as pl

#All_auto_corr_test_NS=[]
#All_auto_corr_test=[]
#All_auto_corr_stim=[]
#All_cross_corr=[]
#All_cross_corr_rev=[]

#auto_corr_test_NS_All=[]


#-------------------------
# Set Analysis Folder Path
analysisFolder = base_path + r'\Adam_Ele\Shared Programming\Python\Social Zebrafish\Analyzed Data'

analysisFolder = analysisFolder + r'\Single Fish\Correlations20_22'
#analysisFolder = analysisFolder + r'\Mk_801\Mk801vsMK801andWt\Final'
#analysisFolder = analysisFolder + r'\EtOH_Low\Correlations'
#analysisFolder = analysisFolder + r'\EtOH_High\Correlations'
#analysisFolder = analysisFolder + r'\Isolation\Isolation\Correlation'
#analysisFolder = analysisFolder + r'\Isolation\Isolation_Control\Correlation'

# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(analysisFolder+'\*FIG3.npz')

#CAlculate how many files
numFiles = np.size(npzFiles, 0)
corrLength=250

# SPI Structure
SPI_All = np.zeros(numFiles) 

# TonS neabs Test motion triggered on Stim bouts
BTA_TonS_All = np.zeros((numFiles,2*corrLength))
BTA_SonT_All  = np.zeros((numFiles,2*corrLength))
BTA_TonT_All = np.zeros((numFiles,2*corrLength))
BTA_SonS_All  = np.zeros((numFiles,2*corrLength))
numBouts_test_All = np.zeros(numFiles) 
numBouts_stim_All = np.zeros(numFiles) 

# Speeds
speeds_test_All = np.zeros((numFiles, 3))
speeds_stim_All = np.zeros((numFiles, 2))

# Orientation Histogrames
ortHistograms_test_All = np.zeros((3, 36, numFiles))
ortHistograms_stim_All = np.zeros((2, 36, numFiles))

# Num Subset Fileds
numSubsetFields = 13
test_SubsetBTAs_All  = np.zeros((numSubsetFields,2*corrLength, numFiles))
stim_SubsetBTAs_All  = np.zeros((numSubsetFields,2*corrLength, numFiles))

# Bouts Structure
bouts_test_All = np.zeros((0,8))
bouts_stim_All = np.zeros((0,8))

# BTA Subset Fields
# 0 - Avg Bout (Test motion on Test Bout)
# 1 - BTA Visible (Test motion on Stim Bouts when Visibile)
# 2 - BTA NonVisible (Test motion on Stim Bouts when Not Visibile)

i=0
#Go through al the files contained in the analysis folder
for filename in npzFiles:

    #Load each npz file
    dataobject = np.load(filename)
    print filename
    
    #Extract from the nz file the different correlations
    SPI = dataobject['SPI'] 
    BTA_TonS = dataobject['AVG_aligned_testOnStim'] 
    BTA_SonT = dataobject['AVG_aligned_stimOnTest'] 
    BTA_TonT = dataobject['AVG_aligned_testOntest'] 
    BTA_SonS = dataobject['AVG_aligned_stimOnstim'] 
    bouts_test = dataobject['bouts_test']
    bouts_test_ns = dataobject['bouts_test_ns']
    bouts_stim = dataobject['bouts_stim']
    test_SubsetBTAs = dataobject['test_SubsetBTAs']
    stim_SubsetBTAs = dataobject['stim_SubsetBTAs']
    test_SubsetNum = dataobject['test_SubsetNum']
    stim_SubsetNum = dataobject['stim_SubsetNum']
    speeds_test = dataobject['speeds_test']
    speeds_stim = dataobject['speeds_stim']
    ortHistograms_test = dataobject['ortHistograms_test']
    ortHistograms_stim = dataobject['ortHistograms_stim']
     
    # Load SPIs
    SPI_All[i] = SPI

    # Measure and remove baseline motion offset
    baseline_test_motion = np.nanmean(np.concatenate((BTA_TonT[:10], BTA_TonT[(corrLength*2-10):])))
    baseline_stim_motion = np.nanmean(np.concatenate((BTA_SonS[:10], BTA_SonS[(corrLength*2-10):])))
    BTA_TonS=BTA_TonS-baseline_test_motion
    BTA_SonT=BTA_SonT-baseline_stim_motion
    BTA_TonT=BTA_TonT-baseline_test_motion
    BTA_SonS=BTA_SonS-baseline_stim_motion
    
    # Normalize by Avg. Bout Peak
    max_test_motion = np.amax(BTA_TonT)    
    max_stim_motion = np.amax(BTA_SonS)    
    BTA_TonS_All[i,:]=BTA_TonS/max_test_motion
    BTA_SonT_All[i,:]=BTA_SonT/max_stim_motion
    BTA_TonT_All[i,:]=BTA_TonT/max_test_motion
    BTA_SonS_All[i,:]=BTA_SonS/max_stim_motion

    test_SubsetBTAs_All[:,:,i]=test_SubsetBTAs/max_test_motion
    stim_SubsetBTAs_All[:,:,i]=stim_SubsetBTAs/max_stim_motion

    numBouts_test_All[i] = np.size(bouts_test, 0)
    numBouts_stim_All[i] = np.size(bouts_stim, 0)
    
    # Load All  Bouts
    bouts_test_All = np.vstack((bouts_test_All, bouts_test))    
    bouts_stim_All = np.vstack((bouts_stim_All, bouts_stim))    
    
    
    i=i+1
    print i
    
##Transform the list into a matlab-like array
#All_auto_corr_test_NS = np.array(auto_corr_test_NS_All)
#All_auto_corr_test = np.array(auto_corr_test_All)
#All_auto_corr_stim = np.array(auto_corr_stim_All)
#All_cross_corr = np.array(cross_corr_All)
#All_cross_corr_rev = np.array(cross_corr_rev_All)

# Compute Bout Frequency
all_peaks = bouts_test_All[:, 2] # Load Bout times
all_ibi = np.concatenate(([0], np.diff(all_peaks))) # Compute Inter-Bout Intervals
all_ibi_vis = all_ibi[bouts_test_All[:, 6]==1] # Sort based on social_side
all_ibi_nonvis = all_ibi[bouts_test_All[:, 6]==0] # Sort based on social_side


all_ibi = all_ibi[np.logical_and(all_ibi > 0, all_ibi < 3000)] # Filter inter-fish breaks
all_ibi_vis = all_ibi[np.logical_and(all_ibi_vis > 0, all_ibi_vis < 3000)] # Filter inter-fish breaks
all_ibi_nonvis = all_ibi[np.logical_and(all_ibi_nonvis > 0, all_ibi_nonvis < 3000)] # Filter inter-fish breaks

# Plot Avg. Motion Bouts : NS, SS-Not-Visible, SS-Visible


#Make mean correlations
All_BTA_TonT = np.nanmean(BTA_TonT_All[:,:], axis=0)
All_BTA_SonS = np.nanmean(BTA_SonS_All[:,:], axis=0) 


#Plot average correlations
timeAxis = (np.arange(-corrLength, corrLength) + 1.0)/100.0
plt.figure()            
plt.title('Avg. Motion Bursts')
plt.plot(timeAxis, All_BTA_TonT, 'k')
plt.xlabel('Lag (seconds)')
plt.title('IBI %.2f +\\- %.2f and IBI-nvis %.2f +\\- %.2f' % (np.mean(all_ibi_vis), np.std(all_ibi_vis),np.mean(all_ibi_nonvis), np.std(all_ibi_nonvis)))
plt.axis([-1.0, 1.0, -0.2, 1.0])


# Plot All BTA
plt.figure('Summary')
plt.plot(timeAxis, All_BTA_TonT)
plt.axis([-1.0, 1.0, -0.2, 1.0])
# FIN
    
    