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
    lib_path = r'C:/Repos/Dreosti-Lab/Social_Pain/libs'

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
    Analysis_folder = folder + '/Analysis'
    
    return NS_folder, S_folder, Analysis_folder
           



# FIN