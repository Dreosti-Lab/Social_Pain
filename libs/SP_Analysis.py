# -*- coding: utf-8 -*-
"""

@author: Alizee
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal




# Measure ditance traveled during experiment (in mm)
def distance_traveled(bx, by, ROI, numFrames):

    # Rescale by chamber dimensions
    chamber_Width_px = ROI[2]
    chamber_Height_px = ROI[3]
    chamber_Width_mm = 140
    chamber_Height_mm = 10
    
    # Sample position every 10 frames (10 Hz) and accumulate distance swum
    # - Only add increments greater than 0.5 mm
    prev_x = bx[0]
    prev_y = by[0]
    distanceT = 0
    for f in range(9,numFrames,10):
        dx = ((bx[f]-prev_x)/chamber_Width_px) * chamber_Width_mm
        dy = ((by[f]-prev_y)/chamber_Height_px) * chamber_Height_mm
        d = np.sqrt(dx*dx + dy*dy)
        if(d > 0.5):
            distanceT = distanceT + d
            prev_x = bx[f]
            prev_y = by[f]           
    
    return distanceT

# Compute activity level of the fish in bouts per second (BPS)
def measure_BPS(motion, startThreshold, stopThreshold):
                   
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

    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion)/100   ## Assume 100 Frames per Second
    print(numberOfSeconds)
    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds

    return boutsPerSecond
    
# Analyze bouts and pauses (individual stats)
def analyze_bouts_and_pauses(bx, by, ort, motion, startThreshold, stopThreshold):
    
    
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
        bouts[i, 1] = bx[boutStarts[i]]
        bouts[i, 2] = by[boutStarts[i]]
        bouts[i, 3] = ort[boutStarts[i]]
        bouts[i, 4] = boutStops[i]
        bouts[i, 5] = bx[boutStops[i]]
        bouts[i, 6] = by[boutStops[i]]
        bouts[i, 7] = ort[boutStops[i]]
        bouts[i, 8] = boutStops[i] - boutStarts[i]
        
    # Analyse all pauses (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numPauses = numBouts+1
    pauses = np.zeros((numPauses, 9))

    # -Include first and last as pauses (clipped in video)
    # First Pause
    pauses[0, 0] = 0
    pauses[0, 1] = bx[0]
    pauses[0, 2] = by[0]
    pauses[0, 3] = ort[0]
    pauses[0, 4] = boutStarts[0]
    pauses[0, 5] = bx[boutStarts[0]]
    pauses[0, 6] = by[boutStarts[0]]
    pauses[0, 7] = ort[boutStarts[0]]
    pauses[0, 8] = boutStarts[0]
    # Other pauses
    for i in range(1, numBouts):
        pauses[i, 0] = boutStops[i-1]
        pauses[i, 1] = bx[boutStops[i-1]]
        pauses[i, 2] = by[boutStops[i-1]]
        pauses[i, 3] = ort[boutStops[i-1]]
        pauses[i, 4] = boutStarts[i]
        pauses[i, 5] = bx[boutStarts[i]]
        pauses[i, 6] = by[boutStarts[i]]
        pauses[i, 7] = ort[boutStarts[i]]
        pauses[i, 8] = boutStarts[i] - boutStops[i-1]
    # Last Pause
    pauses[-1, 0] = boutStops[-1]
    pauses[-1, 1] = bx[boutStops[-1]]
    pauses[-1, 2] = by[boutStops[-1]]
    pauses[-1, 3] = ort[boutStops[-1]]
    pauses[-1, 4] = len(motion)-1
    pauses[-1, 5] = bx[-1]
    pauses[-1, 6] = by[-1]
    pauses[-1, 7] = ort[-1]
    pauses[-1, 8] = len(motion)-1-boutStops[-1]

    return bouts, pauses
    
# FIN
    