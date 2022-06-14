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

# Import useful libraries
import SP_video_TRARK as SPV


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


def computeDistPerBout(fx_boutStarts, fy_boutStarts, fx_boutEnds,fy_boutEnds):
## Computes total straight line distance travelled over the course of individual bouts

    absDiffX=np.abs(fx_boutStarts-fx_boutEnds)
    absDiffY=np.abs(fy_boutStarts-fy_boutEnds)
    

    distPerBout=np.zeros(len(fx_boutStarts))
    
    for i in range(len(fx_boutStarts)):
        distPerBout[i]=math.sqrt(np.square(absDiffX[i])+np.square(absDiffY[i]))
        if distPerBout[i]>200:distPerBout[i]=0
        
    
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


# Label Bouts
def label(bouts,speed,speedAngle, numFrames, FPS):

    # Parameters
    pre_window = 10
    post_window = 80

    num_bouts = bouts.shape[0]

    # Turn PC constant
    turn_pc = np.array( 
                        [4.45784725e-06,  7.29697833e-06,  8.34722354e-06,  7.25639602e-06,
                        6.83773435e-06,  1.05799488e-05,  9.59485594e-06,  1.04996460e-05,
                        9.50693646e-06,  6.68761575e-06,  1.74239537e-06, -5.13269107e-06,
                        -1.30955946e-05, -2.93123632e-05, -5.16772503e-05, -6.59745678e-05,
                        -6.24515957e-05, -6.82989320e-05, -5.84883171e-05, -5.49322933e-05,
                        -4.75273440e-05, -5.97750465e-05, -5.50942353e-05, -4.32771920e-05,
                        -4.53841833e-05, -4.39441043e-05, -4.29799500e-05, -3.66285781e-05,
                        -2.74927325e-05, -2.79482710e-05, -2.77149944e-05, -3.01089122e-05,
                        -2.69092862e-05, -2.75200069e-05, -3.25928317e-05, -3.87474743e-05,
                        -4.24973212e-05, -4.47429213e-05, -4.64712226e-05, -4.89719267e-05,
                        -5.91676326e-05, -6.22191781e-05, -6.21876092e-05, -6.47945016e-05,
                        -7.40367790e-05, -7.80097327e-05, -7.82331054e-05, -8.03180239e-05,
                        -8.55250976e-05, -8.88741024e-05, -8.93264800e-05, -9.13412355e-05,
                        -9.33324008e-05, -9.54639901e-05, -9.98497139e-05, -1.03221121e-04,
                        -1.08970275e-04, -1.13959552e-04, -1.20395095e-04, -1.22240153e-04,
                        -1.25032979e-04, -1.26145560e-04, -1.21958655e-04, -1.21565879e-04,
                        -1.21595218e-04, -1.18114363e-04, -1.17635286e-04, -1.12130918e-04,
                        -1.12562112e-04, -1.14707619e-04, -1.16066511e-04, -1.17252020e-04,
                        -1.22045156e-04, -1.22450517e-04, -1.25711027e-04, -1.25607020e-04,
                        -1.23958304e-04, -1.19578445e-04, -1.18268675e-04, -1.20917093e-04,
                        -1.23308934e-04, -1.18843590e-04, -1.19599994e-04, -1.20606743e-04,
                        -1.19085433e-04, -1.17407301e-04, -1.11223481e-04, -1.03411623e-04,
                        -9.72959419e-05, -9.09072743e-05, -3.92279029e-04, -8.75810372e-04,
                        -1.47534021e-03, -1.88185473e-03, -2.22179113e-03, -2.55991823e-03,
                        -2.84555972e-03, -3.18082206e-03, -3.41233583e-03, -3.70544285e-03,
                        -4.73103364e-03, -5.97680392e-03, -9.40038181e-03, -2.37417237e-02,
                        -5.71414180e-02, -7.90270203e-02, -8.59715002e-02, -8.39164195e-02,
                        -8.26775443e-02, -8.46991182e-02, -8.87082454e-02, -9.20826611e-02,
                        -9.44035333e-02, -9.58685766e-02, -9.77270940e-02, -9.94995655e-02,
                        -1.01423412e-01, -1.02874920e-01, -1.04038069e-01, -1.05218456e-01,
                        -1.06468904e-01, -1.07616346e-01, -1.08377944e-01, -1.09295619e-01,
                        -1.10020168e-01, -1.11017271e-01, -1.11630187e-01, -1.12289358e-01,
                        -1.13028781e-01, -1.13582258e-01, -1.14247743e-01, -1.14925706e-01,
                        -1.15475069e-01, -1.15872550e-01, -1.16510964e-01, -1.16891761e-01,
                        -1.17313917e-01, -1.17903131e-01, -1.18225351e-01, -1.18641475e-01,
                        -1.19053891e-01, -1.19258273e-01, -1.19559753e-01, -1.19870835e-01,
                        -1.20140247e-01, -1.20378214e-01, -1.20636915e-01, -1.20902923e-01,
                        -1.21193316e-01, -1.21443497e-01, -1.21709187e-01, -1.21760193e-01,
                        -1.21973109e-01, -1.22152281e-01, -1.22344918e-01, -1.22531978e-01,
                        -1.22724310e-01, -1.22906534e-01, -1.23223312e-01, -1.23339858e-01,
                        -1.23424650e-01, -1.23665608e-01, -1.23838407e-01, -1.24060679e-01,
                        -1.24108222e-01, -1.24361033e-01, -1.24545660e-01, -1.24807371e-01,
                        -1.25075108e-01, -1.25255340e-01, -1.25288654e-01, -1.25387074e-01,
                        -1.25516014e-01, -1.25501054e-01, -1.25552951e-01, -1.25657374e-01,
                        -1.25660401e-01, -1.25796678e-01, -1.25729603e-01, -1.25808149e-01]
                        )
   

    # Label bouts as turns (-1 = Left, 1 = Right) and swims (0)
    labels = np.zeros(num_bouts)
    for i, bout in enumerate(bouts):
        index = np.int(bout[0]) # Align to start
        if(index < pre_window):
            continue
        if(index > (numFrames-post_window)):
            continue
        tdD = speed[(index-pre_window):(index+post_window)]
        tD = np.cumsum(tdD)
        tdA = speedAngle[(index-pre_window):(index+post_window)]
        tA = np.cumsum(tdA)

        # Compare bout trajectory to Turn PC
        bout_trajectory = np.hstack((tD, tA))
        turn_score = np.sum(turn_pc * bout_trajectory)

        # Label
        if(turn_score < -90):
            labels[i] = -1
        if(turn_score > 90):
            labels[i] = 1

    return labels

# FIN
    