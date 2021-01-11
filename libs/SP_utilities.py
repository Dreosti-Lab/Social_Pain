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

def utility_function(inputs):
    
    return

# FIN