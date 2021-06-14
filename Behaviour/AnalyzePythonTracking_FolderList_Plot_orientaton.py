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

#analysisFolder = analysisFolder + r'\Single Fish\Correlations6_8'
#analysisFolder = analysisFolder + r'\Single Fish\Correlations13_15'
#analysisFolder = analysisFolder + r'\Single Fish\Correlations20_22'
#analysisFolder = analysisFolder + r'\Dark\Correlations_Dark_Dark'
#analysisFolder = analysisFolder + r'\Dark\Correlations_Dark_Light'
#analysisFolder = analysisFolder + r'\Mk_801\Correlations_MK801vsWt'
#analysisFolder = analysisFolder + r'\Mk_801\Correlations_MK801vs_MK801vsWT'
#analysisFolder = analysisFolder + r'\Mk_801\Correlations_MK801vsMK801'
#analysisFolder = analysisFolder + r'\Size\Correlations_Big_Small'
analysisFolder = analysisFolder + r'\EtOH_Low\Correlations'
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

test_SubsetNum_Visible_All = np.zeros(numFiles) 
test_SubsetNum_NonVisible_All = np.zeros(numFiles) 

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
    
    # Number of Visible and NonVisible Bouts
    test_SubsetNum_Visible_All[i] = stim_SubsetNum[0] # Number of Stim fish bouts used to trigger when on Visible side
    test_SubsetNum_NonVisible_All[i] = stim_SubsetNum[1]
            
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
    

# Good Fish - Social Side
goodFish = np.where(test_SubsetNum_Visible_All > 100)
goodFish_SS = goodFish[0]

# Good Fish - Non-Social Side
goodFish = np.where(test_SubsetNum_NonVisible_All > 100)
goodFish_NSS = goodFish[0]

# Plot Orientation Histogram
social = 1

plt.figure()
ax = plt.subplot(111, polar=True)
plt.title('Orientation Histograms')
mean_ort_test_VIS = np.nanmean(ortHistograms_test_All[:,:,goodFish_SS], axis=2)
mean_ort_stim_VIS = np.nanmean(ortHistograms_stim_All[:,:,goodFish_SS], axis=2)
err_ort_test_VIS = np.nanstd(ortHistograms_test_All[:,:,goodFish_SS], axis=2)/np.sqrt(np.size(goodFish_SS)-1)
err_ort_stim_VIS = np.nanstd(ortHistograms_stim_All[:,:,goodFish_SS], axis=2)/np.sqrt(np.size(goodFish_SS)-1)

mean_ort_test_NONVIS = np.nanmean(ortHistograms_test_All[:,:,goodFish_NSS], axis=2)
mean_ort_stim_NONVIS = np.nanmean(ortHistograms_stim_All[:,:,goodFish_NSS], axis=2)
err_ort_test_NONVIS = np.nanstd(ortHistograms_test_All[:,:,goodFish_NSS], axis=2)/np.sqrt(np.size(goodFish_NSS)-1)
err_ort_stim_NONVIS = np.nanstd(ortHistograms_stim_All[:,:,goodFish_NSS], axis=2)/np.sqrt(np.size(goodFish_NSS)-1)


xAxis = np.arange(-np.pi,np.pi+np.pi/18.0, np.pi/18.0)
#plt.plot(xAxis, mean_ort_test[0,:], 'k', linewidth = 3) # NS Test
plt.plot(xAxis, np.hstack((mean_ort_test_VIS[social,:]-2*err_ort_test_VIS[social,:], mean_ort_test_VIS[social,0]-err_ort_test_VIS[social,0])), 'r', linewidth = 1) # Vis Test
plt.plot(xAxis, np.hstack((mean_ort_test_VIS[social,:]+2*err_ort_test_VIS[social,:], mean_ort_test_VIS[social,0]+err_ort_test_VIS[social,0])), 'r', linewidth = 1) # Vis Test
plt.plot(xAxis, np.hstack((mean_ort_test_VIS[social,:], mean_ort_test_VIS[social,0])), 'r', linewidth = 3) # Vis Test
#plt.plot(xAxis, mean_ort_test_NONVIS[2,:], 'b', linewidth = 3) # NonVis Test
#plt.plot(xAxis, mean_ort_stim[0,:], 'g', linewidth = 3) # Vis Stim
#plt.plot(xAxis, mean_ort_stim[1,:], 'y', linewidth = 3) # NonVis Stim
plt.xlabel('Angle') 
plt.ylabel('Frequency') 
plt.xlim(-180, 180)


# Plot SUmmary
plt.figure('Summary')
ax = plt.subplot(111, polar=True)
plt.plot(xAxis, np.hstack((mean_ort_test_VIS[social,:], mean_ort_test_VIS[social,0])), linewidth = 3) # Vis Test


# FIN
    
    