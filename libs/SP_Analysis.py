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
import glob
import cv2
import pandas as pd

# Import useful libraries
import SP_video_TRARK as SPV
import SP_utilities as SPU


# Measure ditance traveled during experiment (in mm)
def distance_traveled(fx, fy, ROI, numFrames):

    # Rescale by chamber dimensions
    chamber_Width_px = ROI[2]
    chamber_Height_px = ROI[3]
    chamber_Width_mm = 100
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
        
        if d > 0.5:
            
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


def compute_motion(folder,ROIs,change_threshold=0,stepFrames=1000,bFrames = 50):
    
    # First steps are same as usual tracking
    background_ROIs = SPV.compute_initial_backgrounds(folder, ROIs,change_threshold=change_threshold,stepFrames=stepFrames,bFrames=bFrames)
    
    aviFiles = glob.glob(folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    
    print('Processing' + aviFile)
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-200 # skip lat 100 frames (we take 100 off the start later)
    
    ## debug / testing
    numFrames=10000
    ##
    
    previous_ROIs = []
    for i in range(0,6):
        w, h = SPV.get_ROI_size(ROIs, i)
        previous_ROIs.append(np.zeros((h, w), dtype = np.uint8))
    motS = np.zeros((numFrames,6))
        
    for f in range(0,numFrames): 
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # loop through ROIs
        for i in range(0,6):
            # Extract Crop Region
            crop, xOff, yOff = SPV.get_ROI_crop(current, ROIs, i)
            crop_height, crop_width = np.shape(crop)

            # Difference from current background
            diff = background_ROIs[i] - crop
            
            # Determine current threshold
            threshold_level = np.median(diff)+(3*np.std(diff)) # 3 standard deviations above the median (yours might be the median + the median/7 or similar)

            
            # if not on the first frame, compute the absolute frame by frame difference across the whole ROI
            if (f != 0):
                absdiff = np.abs(diff)
                absdiff[absdiff < threshold_level] = 0
                totalAbsDiff = np.sum(np.abs(absdiff))
                frame_by_frame_absdiff = np.abs(np.float32(previous_ROIs[i]) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff
            else:
                motion = 0
            motS[f,i]= motion
               
            # keep track of previous ROI within loop for subsequent frame motion computation (because of the way we have to cycle through ROIs each frame)
            previous_ROIs[i] = np.copy(crop)
            
    return motS,background_ROIs

def compute_speed(X,Y):
    # Compute Speed (X-Y)    
    speed = np.sqrt(np.diff(X)*np.diff(X) + np.diff(Y)*np.diff(Y)) 
    speed = np.append([0], speed)
    return speed


def computeDistPerBout(fx_boutStarts, fy_boutStarts, fx_boutEnds,fy_boutEnds, ROI):
## Computes total straight line distance travelled over the course of individual bouts

    fx_boutStarts_mm, fy_boutStarts_mm = SPU.convert_mm(fx_boutStarts,fy_boutStarts, ROI)
    fx_boutEnds_mm, fy_boutEnds_mm = SPU.convert_mm(fx_boutEnds,fy_boutEnds, ROI)
    
    absDiffX=np.abs(fx_boutStarts_mm - fx_boutEnds_mm)
    absDiffY=np.abs(fy_boutStarts_mm - fy_boutEnds_mm)
    

    Bout_dist=np.zeros(len(fx_boutStarts))
    
    for i in range(len(fx_boutStarts)):
        
        Bout_dist[i]= np.sqrt(absDiffX[i]*absDiffX[i] + absDiffY[i]* absDiffY[i])
    
    #Concatenate bout type to dataFrame
    xStart_NS = pd.Series(fx_boutStarts, name = 'fxStart')
    yStart_NS = pd.Series(fy_boutStarts, name = 'fyStart')
    dist = pd.Series(Bout_dist, name= 'distT')
    distPerBout = pd.concat([xStart_NS,yStart_NS, dist], axis=1)
    
    return distPerBout


def computeDistPerFrame(fx,fy):
## Computes straight line distance between every frame, given x and y coordinates of tracking data
    cumDistPerFrame=np.zeros(len(fx)-1)
    distPerFrame=np.zeros(len(fx))
    absDiffX=np.abs(np.diff(fx))
    absDiffY=np.abs(np.diff(fy))
    for i in range(len(fx)-1):
        if i!=0:
            distPerFrame[i]=math.sqrt(np.square(absDiffX[i])+np.square(absDiffY[i]))
            if distPerFrame[i]>200:distPerFrame[i]=0
            cumDistPerFrame[i]=distPerFrame[i]+cumDistPerFrame[i-1]
    return distPerFrame,cumDistPerFrame


# Compute Dynamic Signal for Detecting Bouts (swims and turns)
def compute_bout_signals(X, Y, Ort):

    # Compute Speed (X-Y)    
    speedXY = compute_speed(X,Y)
    
    # Filter Speed for outliers
    sigma = np.std(speedXY)
    baseline = np.median(speedXY)
    speedXY[speedXY > baseline+10*sigma] = -1.0
    
    # Compute Speed (Angular)
    speedAngle = diffAngle(Ort)
    speedAngle = filterTrackingFlips(speedAngle)
        
    return speedXY, speedAngle

# Build a histogram of all orientation values
def ort_histogram(ort):

    # ORIENTATION ---------------------------
    numOrts = 36
    interval = 360/numOrts
    ortRange = np.arange(-180,180+interval, interval)    
    ortHistogram, bins = np.histogram(ort, ortRange)

    return ortHistogram

def diffAngle(Ort):
## Computes the change in angle over all frames of given Ort tracking data
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
## Identifies and reverses sudden flips in orientation caused by errors in tracking the eyes vs the body resulting in very high frequency tracking flips    
    new_dAngle = []    
    for a in dAngle:
        if a < -100:
            new_dAngle.append(a + 180)
        elif a > 100:
            new_dAngle.append(a - 180)
        else:
            new_dAngle.append(a)
            
    return np.array(new_dAngle)


# rotate the orientation trace so that the initial heading is zero
def rotateOrt(ort):
    ort_rot=np.zeros(len(ort))
    ort_init=ort[0]
    for i,thisOrt in enumerate(ort):
        o=thisOrt-ort_init
        oAbs=np.abs(o)
        if o>180:
            o=(180-(o-180))*-1
        elif o<-180:
            o=180-(oAbs-180)
            
        ort_rot[i]=o
    return ort_rot


# Label Bouts
def label_bouts(bouts,ort):

    # Parameters
    preWindow = 10
    postWindow = 80

    num_bouts = bouts.shape[0]
    boutStarts = bouts[:,1]
       
    Bout_ort = np.zeros([num_bouts,(preWindow+postWindow)])
    Bout_Angles = np.zeros(num_bouts)
    
    for b in range(0,num_bouts):
        Bout_ort[b,:] = rotateOrt(ort[int(boutStarts[b]-preWindow):int(boutStarts[b]+postWindow)]) # extract heading for this bout (rotated to zero initial heading)
        Bout_Angles[b] = np.mean(Bout_ort[b,-2:-1])-np.mean(Bout_ort[b,0:1]) # take the heading before and after


    labels=np.zeros((len(Bout_Angles),3))

    
    for i,angle in enumerate(Bout_Angles):
        if angle > 10: 
            labels[i,0]=1
        elif angle < -10:
            labels[i,1]=1
        else:
            labels[i,2]=1
    L=labels[:,0]!=0
    R=labels[:,1]!=0
    F=labels[:,2]!=0
    
    LTurn = pd.Series(R, name='RTurn')
    RTurn = pd.Series(L, name='LTurn')
    FSwim = pd.Series(F, name='FSwim')
    Bout_labels = pd.concat([LTurn, RTurn, FSwim], axis=1)
           

    return Bout_labels

# FIN
    