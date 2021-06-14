# -*- coding: utf-8 -*-
"""

@author: Alizee
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal

    
# Analyze bouts and pauses (individual stats)
def analyze_bouts_and_pauses(fx, fy, ort, motion, startThreshold, stopThreshold):
    
    
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for i, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(i)
        else:
            if m < stopThreshold:
                moving = 0
                boutStops.append(i)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]

    # Extract all bouts (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numBouts= len(boutStarts)
    bouts = np.zeros((numBouts, 9))
    for i in range(0, numBouts):
        bouts[i, 0] = boutStarts[i]
        bouts[i, 1] = fx[boutStarts[i]]
        bouts[i, 2] = fy[boutStarts[i]]
        bouts[i, 3] = ort[boutStarts[i]]
        bouts[i, 4] = boutStops[i]
        bouts[i, 5] = fx[boutStops[i]]
        bouts[i, 6] = fy[boutStops[i]]
        bouts[i, 7] = ort[boutStops[i]]
        bouts[i, 8] = boutStops[i] - boutStarts[i]
        
    # Analyse all pauses (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numPauses = numBouts+1
    pauses = np.zeros((numPauses, 9))

    
    for i in range(0, numBouts):
        pauseStarts = boutStops[i-1]
        pauseStops = boutStarts[i]
        
        pauses[i, 0] = pauseStarts
        pauses[i, 1] = fx[pauseStarts]
        pauses[i, 2] = fy[pauseStarts]
        pauses[i, 3] = np.median(ort[pauseStarts:pauseStops])
        pauses[i, 4] = pauseStops
        pauses[i, 5] = fx[pauseStops]
        pauses[i, 6] = fy[pauseStops]
        pauses[i, 7] = ort[pauseStops]
        pauses[i, 8] = pauseStops - pauseStarts

    return bouts, pauses



# FIN
    