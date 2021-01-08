# -*- coding: utf-8 -*-
"""
Social Pain "Utilities"

@author: dreostilab (Elena Dreosti)
"""
# -----------------------------------------------------------------------------
# Detect Platform
import platform
if(platform.system() == 'Linux'):
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'/home/kampff/Repos/Dreosti-Lab/Social_Zebrafish/libs'
else:
    # Set "Repo Library Path" - Social Zebrafish Repo
    lib_path = r'C:/Repos/Dreosti-Lab/Social_Zebrafish/libs'

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

def utility_function(inputs):
    
    return

# FIN