# -*- coding: utf-8 -*-
"""
12/06/22
@alizeekastler
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social Zebrafish Repo
lib_path = r'C:/Repos/Social_Pain/libs'
#-----------------------------------------------------------------------------
# Set "Base Path" for this analysis session
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Set Library Paths
import sys
sys.path.append(lib_path)

# Import useful libraries

import numpy as np
import glob


# -----------------------------------------------------------------------------
# Function to load all summary statistics and make a summary figure
# -----------------------------------------------------------------------------

# Analysis folder lONG isolation
analysisFolder = base_path + r'\Isolation_Experiments\Social_Brain_Areas_Analisys\Analysis_Folder\Isolation_7dpf'
reportFilename = analysisFolder + r'\report.txt'


# Find all the npz files saved for each group and fish with all the information
npzFiles = glob.glob(analysisFolder+'\*.npz')

#Calculate how many files
numFiles = np.size(npzFiles, 0)
# Allocate space for summary data
BPS_NS_ALL = np.zeros(numFiles)
BPS_S_ALL = np.zeros(numFiles)
avgPosition_NS_ALL = np.zeros(numFiles)
avgPosition_S_ALL = np.zeros(numFiles)
avg_cue_motion_ALL = np.zeros(numFiles)
numFreezes_NS_ALL = np.zeros(numFiles)
numFreezes_S_ALL = np.zeros(numFiles)
Binned_Freezes_NS_ALL = np.zeros((numFiles,14))
Binned_Freezes_S_ALL = np.zeros((numFiles,14))
Binned_DistanceT_NS_ALL = np.zeros((numFiles,15))
Binned_DistanceT_S_ALL = np.zeros((numFiles,15))
Percent_Moving_NS_ALL = np.zeros(numFiles)
Percent_Moving_S_ALL = np.zeros(numFiles)
Percent_Paused_NS_ALL = np.zeros(numFiles)
Percent_Paused_S_ALL = np.zeros(numFiles)
DistanceT_NS_ALL = np.zeros(numFiles)
DistanceT_S_ALL = np.zeros(numFiles)
Bouts_NS_ALL = np.zeros((0,9))
Bouts_S_ALL = np.zeros((0,9))
Pauses_NS_ALL = np.zeros((0,9))   
Pauses_S_ALL = np.zeros((0,9))
Freezes_S_ALL = np.zeros((0,4))
Freezes_NS_ALL = np.zeros((0,4))
OrtHist_NS_Cool_ALL = np.zeros((numFiles,36))
OrtHist_NS_Hot_ALL = np.zeros((numFiles,36))
OrtHist_NS_Noxious_ALL = np.zeros((numFiles,36))
OrtHist_S_Cool_ALL = np.zeros((numFiles,36))
OrtHist_S_Hot_ALL = np.zeros((numFiles,36))
OrtHist_S_Noxious_ALL = np.zeros((numFiles,36))
Position_NS_ALL = np.zeros((numFiles,3))
Position_S_ALL = np.zeros((numFiles,3))


# Go through all the files contained in the analysis folder
reportFile = open(reportFilename, 'w')
for f, filename in enumerate(npzFiles):

    # Load each npz file
    dataobject = np.load(filename)
    
    # Extract from the npz file 
    BPS_NS = dataobject['BPS_NS']   
    BPS_S = dataobject['BPS_S']
    Bouts_NS = dataobject['Bouts_NS']   
    Bouts_S = dataobject['Bouts_S']
    Pauses_NS = dataobject['Pauses_NS']   
    Pauses_S = dataobject['Pauses_S']
    Freezes_NS = dataobject['Freezes_NS']
    Freezes_S = dataobject['Freezes_S']
    Percent_Moving_NS = dataobject['Percent_Moving_NS']
    Percent_Moving_S = dataobject['Percent_Moving_S']
    Percent_Paused_NS = dataobject['Percent_Paused_NS']
    Percent_Paused_S = dataobject['Percent_Paused_S']
    numFreezes_NS = dataobject['numFreezes_NS']
    numFreezes_S = dataobject['numFreezes_S']
    Binned_Freezes_NS = dataobject['Binned_Freezes_NS']
    Binned_Freezes_S = dataobject['Binned_Freezes_S']
    DistanceT_NS = dataobject['DistanceT_NS']
    DistanceT_S = dataobject['DistanceT_S']
    Binned_DistanceT_NS = dataobject['Binned_DistanceT_NS']
    Binned_DistanceT_S = dataobject['Binned_DistanceT_S']
    OrtHist_NS_Cool = dataobject['OrtHist_NS_Cool']
    OrtHist_NS_Hot = dataobject['OrtHist_NS_Hot']
    OrtHist_NS_Noxious = dataobject['OrtHist_NS_Noxious']
    OrtHist_S_Cool = dataobject['OrtHist_S_Cool']
    OrtHist_S_Hot = dataobject['OrtHist_S_Hot']
    OrtHist_S_Noxious = dataobject['OrtHist_S_Noxious']
    Position_NS = dataobject['Position_NS']
    Position_S = dataobject['Position_S']
    avgPosition_NS = dataobject['avgPosition_NS']
    avgPosition_S = dataobject['avgPosition_S']
    avg_cue_motion = dataobject['avg_cue_motion']
    
    # Make an array with all summary stats
    BPS_NS_ALL[f] = BPS_NS
    BPS_S_ALL[f] = BPS_S
    numFreezes_NS_ALL[f] = numFreezes_NS
    numFreezes_S_ALL[f] = numFreezes_S
    Binned_Freezes_NS_ALL[f,:] = Binned_Freezes_NS
    Binned_Freezes_S_ALL[f,:] = Binned_Freezes_S
    Binned_DistanceT_NS_ALL[f,:] = Binned_DistanceT_NS
    Binned_DistanceT_S_ALL[f,:] = Binned_DistanceT_S
    Percent_Moving_NS_ALL[f] = Percent_Moving_NS
    Percent_Moving_S_ALL[f] = Percent_Moving_S
    Percent_Paused_NS_ALL[f] = Percent_Paused_NS
    Percent_Paused_S_ALL[f] = Percent_Paused_S
    Position_NS_ALL[f] = Position_NS
    Position_S_ALL[f] = Position_S
    avgPosition_NS_ALL[f] = avgPosition_NS
    avgPosition_S_ALL[f] = avgPosition_S
    avg_cue_motion_ALL[f] = avg_cue_motion
    DistanceT_NS_ALL[f] = DistanceT_NS
    DistanceT_S_ALL[f] = DistanceT_S
    OrtHist_NS_Cool_ALL[f,:] = OrtHist_NS_Cool
    OrtHist_NS_Hot_ALL[f,:] = OrtHist_NS_Hot
    OrtHist_NS_Noxious_ALL[f,:] = OrtHist_NS_Noxious
    OrtHist_S_Cool_ALL[f,:] = OrtHist_S_Cool
    OrtHist_S_Hot_ALL[f,:] = OrtHist_S_Hot
    OrtHist_S_Noxious_ALL[f,:] = OrtHist_S_Noxious
    # Concat all Pauses/Bouts
    Bouts_NS_ALL = np.vstack([Bouts_NS_ALL, Bouts_NS])
    Bouts_S_ALL = np.vstack([Bouts_S_ALL, Bouts_S])
    Pauses_NS_ALL = np.vstack([Pauses_NS_ALL, Pauses_NS])
    Pauses_S_ALL = np.vstack([Pauses_S_ALL, Pauses_S]) 
    Freezes_NS_ALL = np.vstack([Freezes_NS_ALL, Freezes_NS])
    Freezes_S_ALL = np.vstack([Freezes_S_ALL, Freezes_S])
    


    # Save to report file
    reportFile.write(filename + '\n')
    reportFile.write('-------------------\n')
    reportFile.write('avgPosition_NS:\t' + format(np.float64(avgPosition_NS), '.3f') + '\n')
    reportFile.write('avgPosition_S:\t' + format(np.float64(avgPosition_S), '.3f') + '\n')
    reportFile.write('BPS_NS:\t' + format(np.float64(BPS_NS), '.3f') + '\n')
    reportFile.write('BPS_S:\t' + format(np.float64(BPS_S), '.3f') + '\n')
    reportFile.write('DistanceT_NS:\t' + format(np.float64(DistanceT_NS), '.3f') + '\n')
    reportFile.write('DistanceT_S:\t' + format(np.float64(DistanceT_S), '.3f') + '\n')
    reportFile.write('Freezes_NS:\t' + format(np.float64(Freezes_NS), '.3f') + '\n')
    reportFile.write('Freezes_S:\t' + format(np.float64(Freezes_S), '.3f') + '\n')
    reportFile.write('Perc_Moving_NS:\t' + format(np.float64(Percent_Moving_NS), '.3f') + '\n')
    reportFile.write('Perc_Moving_S:\t' + format(np.float64(Percent_Moving_S), '.3f') + '\n')
    reportFile.write('-------------------\n\n')
    

# Close report
reportFile.close()

# FIN
