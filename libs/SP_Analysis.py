# -*- coding: utf-8 -*-
"""

@author: Alizee
"""

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import math




# Measure ditance traveled during experiment (in mm)
def distance_traveled(fx, fy, ROI, numFrames):

    # Rescale by chamber dimensions
    chamber_Width_px = ROI[2]
    chamber_Height_px = ROI[3]
    chamber_Width_mm = 100
    chamber_Height_mm = 14
    
    # Sample position every 10 frames (10 Hz) and accumulate distance swum
    # - Only add increments greater than 0.5 mm
    
    distance_frame = np.zeros((90000,2)) 
    prev_x = fx[0]
    prev_y = fy[0]
    distanceT = 0
    for f in range(9,numFrames,10):
        
        distance_frame[f,0] = f
        
        dx = ((fx[f]-prev_x)/chamber_Width_px) * chamber_Width_mm
        dy = ((fy[f]-prev_y)/chamber_Height_px) * chamber_Height_mm
        d = np.sqrt(dx*dx + dy*dy)
        
        if d > 0.5:
            
            distance_frame[f,1]=d
            
            distanceT = distanceT + d
            prev_x = fx[f]
            prev_y = fy[f] 

            
    return distanceT, distance_frame

# Compute activity level of the fish in bouts per second (BPS)
def measure_BPS(motion, startThreshold, stopThreshold):
                   
    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    # f=frames, m=motion
    for f, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(f)
        else:
            if np.sum(motion[f:(f+30)]) == stopThreshold:
            #bout stops only if at 0 for more than 30 frames (eliminate lost tracking)    
                moving = 0
                boutStops.append(f)
    
    # Extract all bouts (ignore last, if clipped)
    boutStarts = np.array(boutStarts)
    boutStops = np.array(boutStops)
    if(len(boutStarts) > len(boutStops)):
        boutStarts = boutStarts[:-1]

    # Count number of bouts
    numBouts= len(boutStarts)
    numberOfSeconds = np.size(motion)/100
    # Set the bouts per second (BPS)
    boutsPerSecond = numBouts/numberOfSeconds
    
    # Measure averge bout trajectory
    boutStarts = boutStarts[(boutStarts > 25) * (boutStarts < (len(motion)-75))]
    allBouts = np.zeros([len(boutStarts), 100])
    for b in range(0,len(boutStarts)):
        allBouts[b,:] = motion[(boutStarts[b]-25):(boutStarts[b]+75)];
    avgBout = np.mean(allBouts,0);

    return boutsPerSecond, avgBout
    
# Analyze bouts and pauses (individual stats)
def analyze_bouts_and_pauses(fx, fy, ort, motion, ROI, startThreshold, stopThreshold):
    

    # Find bouts starts and stops
    boutStarts = []
    boutStops = []
    moving = 0
    for f, m in enumerate(motion):
        if(moving == 0):
            if m > startThreshold:
                moving = 1
                boutStarts.append(f)
        else:
            if np.sum(motion[f:(f+30)]) == stopThreshold:
                moving = 0
                boutStops.append(f)
    
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
        bouts[i, 2] = (fy[boutStarts[i]])-ROI
        bouts[i, 3] = ort[boutStarts[i]]
        bouts[i, 4] = boutStops[i]
        bouts[i, 5] = fx[boutStops[i]]
        bouts[i, 6] = (fy[boutStops[i]])-ROI
        bouts[i, 7] = ort[boutStops[i]]
        bouts[i, 8] = boutStops[i] - boutStarts[i]
        
    # Analyse all pauses (startindex, startx, starty, startort, stopindex, stopx, stopy, stoport, duration)
    numPauses = numBouts+1
    pauses = np.zeros((numPauses, 9))

    for i in range(1, numBouts):
        pauses[i, 0] = boutStops[i-1]
        pauses[i, 1] = fx[boutStops[i-1]]
        pauses[i, 2] = (fy[boutStops[i-1]])-ROI
        pauses[i, 3] = ort[boutStops[i-1]]
        pauses[i, 4] = boutStarts[i]
        pauses[i, 5] = fx[boutStarts[i]]
        pauses[i, 6] = (fy[boutStarts[i]])-ROI
        pauses[i, 7] = ort[boutStarts[i]]
        pauses[i, 8] = boutStarts[i]- boutStops[i-1]
  
    return bouts, pauses


def analyze_freezes(pauses, freeze_threshold):
    
    numFreezes = np.sum(pauses[:,8]>freeze_threshold)
    
    freezeStart = pauses[:,0][(pauses[:,8]> freeze_threshold)]
    fx_freezeStart = pauses[:,1][(pauses[:,8]> freeze_threshold)]
    fy_freezeStart = pauses[:,2][(pauses[:,8]> freeze_threshold)]
    ort_freezeStart = pauses[:,3][(pauses[:,8]> freeze_threshold)]

    freezes = np.stack((freezeStart,fx_freezeStart, fy_freezeStart, ort_freezeStart), axis=-1)
   
    return freezes, numFreezes

# Analyze temporal bouts
def analyze_temporal_bouts(bouts, binning):

    # Determine total bout counts
    num_bouts = bouts.shape[0]

    # Determine largest frame number in all bouts recordings (make multiple of 100)
    max_frame = np.int(np.max(bouts[:, 4]))
    max_frame = max_frame + (binning - (max_frame % binning))
    max_frame = 100 * 60 * 15

    # Temporal bouts
    bout_hist = np.zeros(max_frame)
    frames_moving = 0
    
    for i in range(0, num_bouts):
        # Extract bout params
        start = np.int(bouts[i][0])
        stop = np.int(bouts[i][4])
        duration = np.int(bouts[i][8])
  
        # Ignore bouts beyond 15 minutes
        if stop >= max_frame:
            continue

        # Accumulate bouts in histogram
        bout_hist[start:stop] = bout_hist[start:stop] + 1
        frames_moving += duration


    plt.figure()
    plt.plot(bout_hist, 'b')
    plt.show()

    # Bin bout histogram
    bout_hist_binned = np.sum(np.reshape(bout_hist.T, (binning, -1), order='F'), 0)

    plt.figure()
    plt.plot(bout_hist_binned, 'r')
    plt.show()


    return bout_hist, frames_moving

def Binning (parameter_2D_array, binsize_frames):
    
            f,d = np.transpose(parameter_2D_array) 
            f = (f//binsize_frames). astype(int) 
            binned_parameter = np.bincount(f,d).astype(np.int64)
            
            return binned_parameter
        
        
def Bin_Freezes(Freeze_Start, FPS, movieLength, binsize) :
    
    FPS = 100
    
    movieLengthFrames= movieLength*60*FPS #in frames
    binsizeFrames = binsize*60*FPS

    Binned_Freezes=[]
    for x in range(0, movieLengthFrames,binsizeFrames):
      
        if x>0:
            boo=Freeze_Start<x
            Binned_Freezes.append(np.sum(boo[Freeze_Start>(x-binsizeFrames)]))
            
    
    return Binned_Freezes       



def compute_speed(X,Y):
    # Compute Speed (X-Y)    
    speed = np.sqrt(np.diff(X)*np.diff(X) + np.diff(Y)*np.diff(Y)) 
    speed = np.append([0], speed)
    return speed


def computeDistPerBout(fx,fy,boutStarts,boutEnds):
## Computes total straight line distance travelled over the course of individual bouts
## Returns a distance travelled for each bout, and a cumDist    
    absDiffX=np.abs(fx[boutStarts]-fx[boutEnds])
    absDiffY=np.abs(fy[boutStarts]-fy[boutEnds])
    
    cumDistPerBout=np.zeros(len(boutStarts)-1)
    distPerBout=np.zeros(len(boutStarts))
    
    for i in range(len(boutStarts)):
        distPerBout[i]=math.sqrt(np.square(absDiffX[i])+np.square(absDiffY[i]))
#        if distPerBout[i]>100:distPerBout[i]=0
        if i!=0 and i!=len(boutStarts)-1:
            cumDistPerBout[i]=distPerBout[i]+cumDistPerBout[i-1]
    
    return distPerBout,cumDistPerBout


def computeDistPerFrame(fx,fy):
## Computes straight line distance between every frame, given x and y coordinates of tracking data
    cumDistPerFrame=np.zeros(len(fx)-1)
    distPerFrame=np.zeros(len(fx))
    absDiffX=np.abs(np.diff(fx))
    absDiffY=np.abs(np.diff(fy))
    for i in range(len(fx)-1):
        if i!=0:
            distPerFrame[i]=math.sqrt(np.square(absDiffX[i])+np.square(absDiffY[i]))
            if distPerFrame[i]>100:distPerFrame[i]=0
            cumDistPerFrame[i]=distPerFrame[i]+cumDistPerFrame[i-1]
    return distPerFrame,cumDistPerFrame

# Build a histogram of all orientation values
def ort_histogram(ort):

    # ORIENTATION ---------------------------
    numOrts = 36
    interval = 360/numOrts
    ortRange = np.arange(-180,180+interval, interval)    
    ortHistogram, bins = np.histogram(ort, ortRange)

    return ortHistogram

# FIN
    