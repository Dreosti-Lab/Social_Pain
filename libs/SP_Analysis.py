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
def distance_traveled(fx, fy, ROI, numFrames):

    # Rescale by chamber dimensions
    chamber_Width_px = ROI[2]
    chamber_Height_px = ROI[3]
    chamber_Width_mm = 109
    chamber_Height_mm = 14
    
    # Sample position every 10 frames (10 Hz) and accumulate distance swum
    # - Only add increments greater than 0.5 mm
    prev_x = fx[0]
    prev_y = fy[0]
    distanceT = 0
    for f in range(9,numFrames,10):
        dx = ((fx[f]-prev_x)/chamber_Width_px) * chamber_Width_mm
        dy = ((fy[f]-prev_y)/chamber_Height_px) * chamber_Height_mm
        d = np.sqrt(dx*dx + dy*dy)
        if(d > 0.5):
            distanceT = distanceT + d
            prev_x = fx[f]
            prev_y = fy[f]           
    
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
    #print(numberOfSeconds)
    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds

    return boutsPerSecond
    
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

    for i in range(1, numBouts):
        pauses[i, 0] = boutStops[i-1]
        pauses[i, 1] = fx[boutStops[i-1]]
        pauses[i, 2] = fy[boutStops[i-1]]
        pauses[i, 3] = ort[boutStops[i-1]]
        pauses[i, 4] = boutStarts[i]
        pauses[i, 5] = fx[boutStarts[i]]
        pauses[i, 6] = fy[boutStarts[i]]
        pauses[i, 7] = ort[boutStarts[i]]
        pauses[i, 8] = boutStarts[i]- boutStops[i-1]
  
    return bouts, pauses

# Build a histogram of all orientation values
def ort_histogram(ort):

    # ORIENTATION ---------------------------
    numOrts = 36
    interval = 360/numOrts
    ortRange = np.arange(-180,180+interval, interval)    
    ortHistogram, bins = np.histogram(ort, ortRange)

    return ortHistogram

# FIN
    