# -*- coding: utf-8 -*-
"""
Make overlay demo video from tracking and raw movie
"""

# Set Library Path - Social_Pain Repos
lib_path = r'/Users/alizeekastler/Documents/GitHub/Social_Pain/libs'
#lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import glob
import cv2


# Import local modules

import SP_utilities as SPU
import SP_Analysis as SPA
import SP_video_TRARK as SPV
import BONSAI_ARK

# Specify paths

movie_file = '/Volumes/T7 Touch/Behaviour_Heat_Gradient/Gradient/Experiments/2022_03_10/Fish2_23dpf/Non_Social/Non_Social.avi'
tracking_file = '/Volumes/T7 Touch/Behaviour_Heat_Gradient/Gradient/Experiments/2022_03_10/Fish2_23dpf/Non_Social/tracking3.npz'
# Set threshold
freeze_threshold = 300
motionStartThreshold = 0.03
motionStopThreshold = 0.015
FPS = 100



# Load tracking data
data = np.load(tracking_file)
tracking = data['tracking']
fx = tracking[:,0] 
fy = tracking[:,1]
bx = tracking[:,2]
by = tracking[:,3]
ex = tracking[:,4]
ey = tracking[:,5]
area = tracking[:,6]
ort = tracking[:,7]
motion = tracking[:,8]


#Load Crop regions NS
bonsaiFiles = glob.glob('/Volumes/T7 Touch/Behaviour_Heat_Gradient/Gradient/Experiments/2022_03_10/Fish2_23dpf/Non_Social/'+ '/*.bonsai')
bonsaiFiles = bonsaiFiles[0]
NS_ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)


x=NS_ROIs[:,0]
y=NS_ROIs[:,1]
width=NS_ROIs[:,2]
height=NS_ROIs[:,3]

    
#15min Movie
fx_NS = fx[12000:13000]
fy_NS = fy[12000:13000]
bx_NS = bx[12000:13000]
by_NS = by[12000:13000]
ex_NS = ex[12000:13000]
ey_NS = ey[12000:13000]
area_NS = area[12000:13000]
ort_NS = ort[12000:13000]
motion_NS = motion[12000:13000]
  

numFrames = 2500 
FPS= 100


fx_NS_mm, fy_NS_mm = SPU.convert_mm(fx_NS, fy_NS, NS_ROIs[0])

#--------------------------------------------------------------------------------------------------------            

#Smooth Motion
Smotion_NS =SPU.smoothSignal(motion_NS,N=3)

# Analyze "Bouts" and "Pauses" 
Bouts_NS, Pauses_NS = SPA.analyze_bouts_and_pauses(fx_NS_mm, fy_NS_mm,ort_NS, Smotion_NS, NS_ROIs[0,1],motionStartThreshold, motionStopThreshold)


Track= plt.figure(figsize=(6,3), dpi=300)
plt.plot(fx_NS_mm, fy_NS_mm)
plt.scatter(Bouts_NS[:,1],Bouts_NS[:,2])


Track.savefig('/Volumes/T7 Touch/Behaviour_Heat_Gradient/Figure_Nox/' + 'Gradient.eps', format='eps', dpi=300,bbox_inches= 'tight')









