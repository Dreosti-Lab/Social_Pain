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
#analysisFolder = analysisFolder + r'\Mk_801\Correlations_MK801vsWt'

#analysisFolder = analysisFolder + r'\Dark\Correlations_Dark_Dark'
#analysisFolder = analysisFolder + r'\Dark\Correlations_Dark_Light'

#analysisFolder = analysisFolder + r'\Single Fish\Correlations20_22'
#analysisFolder = analysisFolder + r'\Mk_801\Mk801vsMK801andWt\Final'
#analysisFolder = analysisFolder + r'\EtOH_High\Correlations'
#analysisFolder = analysisFolder + r'\EtOH_Low\Correlations'

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
    
    # Normalize by Avg. Bout Peak
    max_test_motion = np.amax(BTA_TonT)    
    max_stim_motion = np.amax(BTA_SonS)
    
    #Make an array with all the SPI NS and S    
    BTA_TonS_All[i,:]=BTA_TonS/max_test_motion
    BTA_SonT_All[i,:]=BTA_SonT/max_stim_motion
    BTA_TonT_All[i,:]=BTA_TonT/max_test_motion
    BTA_SonS_All[i,:]=BTA_SonS/max_stim_motion

    test_SubsetBTAs_All[:,:,i]=test_SubsetBTAs/max_test_motion
    stim_SubsetBTAs_All[:,:,i]=stim_SubsetBTAs/max_stim_motion
 
    # Remove Baseline
    BTA_TonS_All[i,:]=BTA_TonS_All[i,:]-np.mean(BTA_TonS_All[i,:])
    BTA_SonT_All[i,:]=BTA_SonT_All[i,:]-np.mean(BTA_SonT_All[i,:])
 
 
    numBouts_test_All[i] = np.size(bouts_test, 0)
    numBouts_stim_All[i] = np.size(bouts_stim, 0)
        
    # Speeds
    speeds_test_All[i,:] = speeds_test
    speeds_stim_All[i,:] = speeds_stim
    
    # Orientation Histogrames (after Normlaizing!)
    ortHistograms_test[0,:] = ortHistograms_test[0,:]/np.sum(ortHistograms_test, axis = 1)[0]
    ortHistograms_test[1,:] = ortHistograms_test[1,:]/np.sum(ortHistograms_test, axis = 1)[1]
    ortHistograms_test[2,:] = ortHistograms_test[2,:]/np.sum(ortHistograms_test, axis = 1)[2]

    ortHistograms_stim[0,:] = ortHistograms_stim[0,:]/np.sum(ortHistograms_stim, axis = 1)[0]
    ortHistograms_stim[1,:] = ortHistograms_stim[1,:]/np.sum(ortHistograms_stim, axis = 1)[1]

    ortHistograms_test_All[:,:,i] = ortHistograms_test
    ortHistograms_stim_All[:,:,i] = ortHistograms_stim
    
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


# Good Fish
goodFish = np.where(numBouts_test_All > 100)
goodFish = goodFish[0]

# Plot Bout Details
VisBouts = bouts_test_All[:, 6] == 1
NonVisBouts = bouts_test_All[:, 6] == 0

# Make Bout Angle Change Histogram
angleBins = np.arange(-180,181,1)
[turnHist,binCenters] = np.histogram(bouts_test_All[:,4], bins = angleBins)
plt.figure()
plt.plot(binCenters[1:]-0.5, turnHist)

# Plot Bout Scatter
plt.figure()
#plt.title('Bout Analysis')
#plt.plot(bouts_test_All[NonVisBouts,4], bouts_test_All[NonVisBouts,5], '.', markersize = 1, color = [0.0,1.0,0.0,0.1])
#plt.plot(bouts_test_All[VisBouts,4], bouts_test_All[VisBouts,5], '.', markersize = 1, color = [1.0,0.0,0.0,0.1])
plt.plot(bouts_test_All[:,4], bouts_test_All[:,5], '.', markersize = 1, color = [0.0,0.0,0.0,0.4])
#plt.xlabel('Turn Amplitude (deg)')
#plt.ylabel('Distance (mm)')
plt.axis([-80.0, 80.0, 0.0, 4.0])
plt.axis('off')


# FIN
    
    