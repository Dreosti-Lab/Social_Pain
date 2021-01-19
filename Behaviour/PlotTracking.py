# -*- coding: utf-8 -*-
"""
Plot Fish Tracking (example)
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social_Pain Repos
lib_path = r'C:/Repos/Dreosti-Lab/Social_Pain/libs'

# Set Library Path
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
base_path = r'V:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats



# Set folder
input_folder = base_path + r'/Experiment_3/Behaviours/2020_12_27/Fish5_26dpf/Non_Social_1'

# for i in range(6):
    #fish_number = i + 1

fish_number = 5

# Analyze folder
tracking_file = input_folder + r'/tracking' + str(fish_number) +'.npz'
data = np.load(tracking_file)
tracking = data['tracking']

# Extract tracking data
fx = tracking[:,0] 
fy = tracking[:,1]
bx = tracking[:,2]
by = tracking[:,3]
ex = tracking[:,4]
ey = tracking[:,5]
area = tracking[:,6]
ort = tracking[:,7]
motion = tracking[:,8]

# Filter out bad data
min_x = 275
max_x = 980
min_y = 680
max_y = 725


# Find good samples
num_total_samples = len(fx)
good_samples = (fx > min_x) * (fx < max_x)* (fy > min_y) * (fy < max_y) 
num_good_samples = np.sum(good_samples)
lost_samples = num_total_samples-num_good_samples



# Plot motion
plt.figure()
plt.plot(motion)

# Plot trajectory
plt.figure()
plt.plot(fx[good_samples], fy[good_samples], 'b.', alpha = 0.15)
#plt.plot(good_fx, good_fy, 'r.', alpha = 0.15)
plt.title("Fish #{0}- Lost Frames {1}".format(fish_number, lost_samples))

#good_fx = fx[good_samples]
#good_fy = fy[good_samples]



# SPM Score
SPM = np.mean(fx[good_samples]) - np.mean(good_fx)

