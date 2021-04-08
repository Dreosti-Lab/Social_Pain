# -*- coding: utf-8 -*-
"""
Social Pain "Utilities"

"""
# -----------------------------------------------------------------------------
# Detect Platform
import platform
if(platform.system() == 'Linux'):
    # Set "Repos Library Path" - Social_Pain Repos
    lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
else:
    # Set "Repos Library Path" - Social_Pain Repos
    lib_path = r'C:/Repos/Social_Pain/libs'

# Set Library Paths
import sys
sys.path.append(lib_path)
# -----------------------------------------------------------------------------

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal

#-----------------------------------------------------------------------------
# Utilities for loading and ploting "social pain" data

# Read Folder List file 6 fish 
def read_folder_list(folderlistFile): 
    folderFile = open(folderlistFile, "r") #"r" means read the file
    folderList = folderFile.readlines() # returns a list containing the lines
    folderPath = folderList[0][:-1] # Read Data Path which is the first line
    folderList = folderList[1:] # Remove first line because it contains the path

    # Set Data Path where the experiments are located
    data_path = folderPath
    
    numFolders = len(folderList) 
    groups = np.zeros(numFolders)
    ages = np.zeros(numFolders)
    folderNames = [] # We use this because we do not know the exact lenght
    fishStatus = np.zeros((numFolders, 6))
    
    for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'
        stringLine = f[:-1].split()
        groups[i] = int(stringLine[0])
        ages[i] = int(stringLine[1])
        expFolderName = data_path + stringLine[2]
        folderNames.append(expFolderName)
        fishStat = [int(stringLine[3]), int(stringLine[4]), int(stringLine[5]), int(stringLine[6]), int(stringLine[7]), int(stringLine[8])]    
        fishStatus[i,:] = np.array(fishStat)
        
    return groups, ages, folderNames, fishStatus
 
def get_folder_names(folder):
    # Specifiy Folder Names
    NS_folder = folder + '/Non_Social_1'
    S_folder = folder + '/Social_1'
    Analysis = folder + '/Analysis'
    
    
    return NS_folder, S_folder, Analysis



# Peak Detection
def find_peaks(values, threshold, refract):    
    over = 0
    r = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[2:-100]:
        if over == 0:
            if values[i] > threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i-1)
        else: #This is what happens when over the threshold
            if r < refract:
                r = r + 1
                if values[i] > curPeakVal:
                    curPeakVal = values[i]
                    curPeakIdx = i
            else:
                if values[i] > curPeakVal:
                    curPeakVal = values[i]
                    curPeakIdx = i
                elif values[i] < threshold:
                    over = 0
                    r = 0
                    curPeakVal = 0
                    peaks.append(curPeakIdx)
                    stops.append(i)
    
    return starts, peaks, stops

# Peak Detection
def find_peaks_dual_threshold(values, upper_threshold, lower_threshold):    
    over = 0
    starts = []
    peaks = []
    stops = []
    curPeakVal = 0
    curPeakIdx = 0
    
    numSamples = np.size(values)
    steps = range(numSamples)
    for i in steps[5:-100]:
        if over == 0:
            if values[i] > upper_threshold:
                over = 1
                curPeakVal = values[i]
                curPeakIdx = i                                
                starts.append(i)
        else: #This is what happens when over the upper_threshold
            if values[i] > curPeakVal:
                curPeakVal = values[i]
                curPeakIdx = i
            elif values[i] < lower_threshold:
                over = 0
                curPeakVal = 0
                peaks.append(curPeakIdx)
                stops.append(i)
    
    return starts, peaks, stops

def diffAngle(Ort):
    dAngle = np.diff(Ort)
    new_dAngle = [0]    
    for a in dAngle:
        if a < -270:
            new_dAngle.append(a + 360)
        elif a > 270:
            new_dAngle.append(a - 360)
        else:
            new_dAngle.append(a)
    
    return np.array(new_dAngle)

def filterTrackingFlips(dAngle):
    new_dAngle = []    
    for a in dAngle:
        if a < -90:
            new_dAngle.append(a + 180)
        elif a > 90:
            new_dAngle.append(a - 180)
        else:
            new_dAngle.append(a)
            
    return np.array(new_dAngle)


def compute_speed(X,Y):
    # Compute Speed (X-Y)    
    speed = np.sqrt(np.diff(X)*np.diff(X) + np.diff(Y)*np.diff(Y)) 
    speed = np.append([0], speed)
    return speed

# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals_calibrated(X, Y, Ort, ROI, test):
    
    # Calibrate X and Y in ROI units
    offX = ROI[0]
    offY = ROI[1]
    width = ROI[2]
    height = ROI[3] 
    X = (X - offX)/width
    Y = (Y - offY)/height
    if test:
        X = X * 14; # Convert to mm
        Y = Y * 42; # Convert to mm
    else:
        X = X * 14; # Convert to mm
        Y = Y * 14; # Convert to mm
        

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
#    # Filter Speed for outliers
#    sigma = np.std(speedXY)
#    baseline = np.median(speedXY)
#    speedXY[speedXY > baseline+10*sigma] = -1.0

    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
    
    return speedXY, speedAngle


# Extract Bouts from Motion Signal
def extract_bouts_from_motion(X, Y, Ort, motion, upper_threshold, lower_threshold, ROI):

    SpeedXY, SpeedAngle = compute_bout_signals_calibrated(X, Y, Ort, ROI, True)
      
     
    # Find Peaks in Motion Signal 
    starts, peaks, stops = find_peaks_dual_threshold(motion, upper_threshold, lower_threshold)
    numBouts = np.size(peaks)    
    bouts = np.zeros([numBouts, 6])

    for i in range(numBouts):
        bouts[i, 0] = starts[i]-4 # Point 4 frames (40 ms) before Upper threshold crossing 
        bouts[i, 1] = peaks[i] # Peak
        bouts[i, 2] = stops[i]+1 # Point 1 frame (10 ms) after lower threshold crossing
        bouts[i, 3] = stops[i]-starts[i] # Durations
        bouts[i, 4] = np.sum(SpeedAngle[starts[i]:stops[i]]) # Net angle change  
        bouts[i, 5] = np.sum(SpeedXY[starts[i]:stops[i]]) # Net distance change

    return bouts
           



# FIN