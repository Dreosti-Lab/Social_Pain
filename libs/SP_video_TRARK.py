# -*- coding: utf-8 -*-
"""
Processing Social_Pain videos : 
    - Compute Backgrounds for each ROIs 
    - PreProcessing Average difference between Background and Frame 
    - ImprovedTracking Current Frame - Background Image / find fish Eyes (dark) + Body (Bright)
"""
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import glob
import cv2
import imageio

def findBodyFromSeed(numPixels,im,seed):
    
    reg = np.zeros(im.shape)
    
    #parameters
    mean_reg = float(im[seed[1], seed[0]])
    size = 1
    contour = [] 
    contour_val = []
    dist = 0
    spreadCoord = [(1, 0), (0, 1), (-1, 0), (0, -1)] # 4 connectivity... should really be 8...?
    
    #Spreading
    while(dist<1 and size<numPixels):
    
        for j in range(4):
            #select next pixel
            cPix = [seed[0] +spreadCoord[j][0], seed[1] +spreadCoord[j][1]]
            if cPix[0]>im.shape[0]: cPix[0]=im.shape[0]-1
            if cPix[1]>im.shape[1]: cPix[1]=im.shape[1]-1
            
            if reg[cPix[1], cPix[0]]==0:
                contour.append(cPix)
                contour_val.append(im[cPix[1], cPix[0]] )
                reg[cPix[1], cPix[0]] = 255
        #add the nearest pixel of the contour in it
        dist = abs(int(np.mean(contour_val)) - mean_reg)
    
        dist_list = [abs(i - mean_reg) for i in contour_val ]
        dist = min(dist_list)    #get min distance
        index = dist_list.index(dist) #mean distance index
        size += 1 # updating region size

        #updating mean MUST BE FLOAT
        mean_reg = (mean_reg*size + float(contour_val[index]))/(size+1)
        #updating seed
        cPix = contour[index]
    
        #removing pixel from neigborhood
        del contour[index]
        del contour_val[index]
    
    return reg

## Identifies and reverses sudden flips in orientation caused by errors in tracking the eyes vs the body resulting in very high frequency tracking flips     
def filterTrackingFlips(dAngle):
    new_dAngle = []    
    for a in dAngle:
        if a < -100:
            new_dAngle.append(a + 180)
        elif a > 100:
            new_dAngle.append(a - 180)
        else:
            new_dAngle.append(a)
            
    return np.array(new_dAngle)

# Scripts to find circle edges given origin and radius
def removeDuplicates(lst):
      
    return [t for t in (set(tuple(i) for i in lst))]

def findCircleEdge(xo=50,yo=50,r=5,sampleN=144):

    angles = np.linspace(0, 2 * np.pi, sampleN, endpoint=False)
    points_list=[]
    for a in angles:
        x=int(xo + np.sin(a) * r)
        y=int(yo + np.cos(a) * r)
        points_list.append([x,y]) 
        
#    points_list=removeDuplicates(points_list)
    return points_list

# Process Video : Make Summary Images
def pre_process_video(folder, social):
    
     # Load Video
     aviFiles = glob.glob(folder+'/*.avi')
     aviFile = aviFiles[0]
     vid = cv2.VideoCapture(aviFile)
     numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
   
     # Read First Frame
     ret, im = vid.read()
     previous = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
     width = np.size(previous, 1)
     height = np.size(previous, 0)
   
     # Alloctae Image Space
     stepFrames = 250 # Add a background frame every 2.5 seconds for 50 seconds
     bFrames = 50
     thresholdValue=10
     accumulated_diff = np.zeros((height, width), dtype = float)
     backgroundStack = np.zeros((height, width, bFrames), dtype = float)
     background = np.zeros((height, width), dtype = float)
     bCount = 0
     for i, f in enumerate(range(0, numFrames, stepFrames)):
       
         vid.set(cv2.CAP_PROP_POS_FRAMES, f)
         ret, im = vid.read()
         current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
         absDiff = cv2.absdiff(previous, current)
         level, threshold = cv2.threshold(absDiff,thresholdValue,255,cv2.THRESH_TOZERO)
         previous = current
      
         # Accumulate differences
         accumulated_diff = accumulated_diff + threshold
       
         # Add to background stack
         if(bCount < bFrames):
             backgroundStack[:,:,bCount] = current
             bCount = bCount + 1
       
         print (numFrames-f)
         print (bCount)

     vid.release()  
    # Normalize accumulated difference image
     accumulated_diff = accumulated_diff/np.max(accumulated_diff)
     accumulated_diff = np.ubyte(accumulated_diff*255)
    
    # Enhance Contrast (Histogram Equalize)
     equ = cv2.equalizeHist(accumulated_diff)

     # Compute Background Frame (median or mode)
     background = np.median(backgroundStack, axis = 2)

     saveFolder = folder
     imageio.imwrite(saveFolder + r'/difference.png', equ)    
     cv2.imwrite(saveFolder + r'/background.png', background)
     # Using SciPy to save caused a weird rescaling when the images were dim.
     # This will change not only the background in the beginning but the threshold estimate

     return 0

#------------------------------------------------------------------------------
    
# Compute the initial background for each ROI
def compute_initial_backgrounds(folder, ROIs, divisor=7):

    # Load Video
    aviFiles = glob.glob(folder+'/*.avi')#finds any avi file in the folder
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Allocate space for all ROI backgrounds
    background_ROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i)#get height and width of all 6 ROIs
        background_ROIs.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for each ROI
    for i in range(0,6):

        # Allocate space for background stack
        crop_width, crop_height = get_ROI_size(ROIs, i)
        stepFrames = 1000 # Check background frame every 10 seconds
        bFrames = 50
        backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)
        background = np.zeros((crop_height, crop_width), dtype = np.float32)
        previous = np.zeros((crop_height, crop_width), dtype = np.float32)
        
        # Store first frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
        backgroundStack[:,:,0] = np.copy(crop)
        previous = np.copy(crop)
        bCount = 1
        
        # Search for useful background frames (significantly different than previous)
        changes = []
        for f in range(stepFrames, numFrames, stepFrames):

            # Read frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, im = vid.read()
            current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
        
            # Measure change from current to previous frame
            absdiff = np.abs(previous-crop)
            level = np.median(crop)/divisor
            change = np.mean(absdiff > level)
            changes.append(change)
            previous = np.copy(crop)
            
            # If significant, add to stack...possible finish
            if(change > 0.0075):
                backgroundStack[:,:,bCount] = np.copy(crop)
                bCount = bCount + 1
                if(bCount == bFrames):
                    print("Background for ROI(" + str(i) + ") found on frame " + str(f))
                    break
        
        # Compute background
        backgroundStack = backgroundStack[:,:, 0:bCount]
        background_ROIs[i] = np.median(backgroundStack, axis=2)
    # median value of 20 frames with significant differences gives a background with no fish
                        
    # Return initial background
    return background_ROIs
#------------------------------------------------------------------------------

# Process Video : Track fish in AVI
def fish_tracking(input_folder, output_folder, ROIs, divisor=7, kSize=3):
    
    # Compute a "Starting" Background
    # - Median value of 20 frames with significant difference between them
    background_ROIs = compute_initial_backgrounds(input_folder, ROIs,divisor=divisor)
    # raster=background_ROIs[0][:]
    # centres,hist=np.histogram(raster)
    # plt.plot(centres[1:],hist)
    
    
    # Algorithm
    # 1. Find initial background guess for each ROI
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using median/5 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
    
    # Load Video_ROI
    aviFiles = glob.glob(input_folder+'/*.avi')
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display?
    display = True
    if display:
        cv2.namedWindow("Tracking")

    # Allocate ROI (crop region) space
    previous_ROIs = []
    for i in range(0,6):
        w, h = get_ROI_size(ROIs, i)
        previous_ROIs.append(np.zeros((h, w), dtype = np.uint8))
    
    # Allocate Tracking Data Space
    fxS = np.zeros((numFrames,6))           # Fish X
    fyS = np.zeros((numFrames,6))           # Fish Y
    bxS = np.zeros((numFrames,6))           # Body X
    byS = np.zeros((numFrames,6))           # Body Y
    exS = np.zeros((numFrames,6))           # Eye X
    eyS = np.zeros((numFrames,6))           # Eye Y
    areaS = np.zeros((numFrames,6))         # area (-1 if error)
    ortS = np.zeros((numFrames,6))          # heading/orientation (angle from body to eyes)
    motS = np.zeros((numFrames,6))          # frame-by-frame change in segmented particle

    # Track    
    for f in range(0,numFrames):  
        
        #### DEBUG ######
        badFrameList=[82376,82776,82176,81876,81176,80376,80176,79276,79376]
        badFrameList = [numFrames-x for x in badFrameList]
        if f in badFrameList:
            print('BadFrame')
        # Read next frame        
        ret, im = vid.read()
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # Process each ROI
        for i in range(0,6):
            
            # Extract Crop Region
            crop, xOff, yOff = get_ROI_crop(current, ROIs, i)
            crop_height, crop_width = np.shape(crop)

            # Difference from current background: within crop, the fish appears brighter than the ROI background
            # The difference betwwen the crop and the computed background is the fish
            diff = background_ROIs[i] - crop

            # Determine current threshold (Sensitivity : detected range of pixel change)
            # How different does it need to be for it to be picked up as the fish, lower threshold if you want to track fish that are not moving much 
            threshold_level = np.median(background_ROIs[i])/divisor # <-----
            
            # Within the diff (fish) our threshold value is the median background and max value is 255 (White)
            #if pixel > threshold_level = white and if < threshold_level = Black
            level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
            # Convert to uint8 : instead of black or white : 0 and 1
            threshold = np.uint8(threshold)
            
            # Binary Close: Closing eliminates noise ie. closing small holes inside object (black spots)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kSize,kSize))
            closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # DEBUG
            #if i == 1:
            #    debug_image = np.copy(closing)

            # Find Binary Contours 
            contours, hierarchy = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            # Create Binary Mask Image
            mask = np.zeros(crop.shape,np.uint8)
            
            # If there are NO contours, then skip tracking
            if len(contours) == 0:
                if f!= 0:
                    area = -1.0
                    fX = fxS[f-1, i] - xOff
                    fY = fyS[f-1, i] - yOff
                    bX = bxS[f-1, i] - xOff
                    bY = byS[f-1, i] - yOff
                    eX = exS[f-1, i] - xOff
                    eY = eyS[f-1, i] - yOff
                    heading = ortS[f-1, i]
                    motion = -1.0
                else:
                    area = -1.0
                    fX = xOff
                    fY = yOff
                    bX = xOff
                    bY = yOff
                    eX = xOff
                    eY = yOff
                    heading = -181.0
                    motion = -1.0
            
            else:
                # Get Largest Contour (fish, ideally)
                largest_cnt, area = get_largest_contour(contours)
                
                # If the particle is too small to consider, skip frame
                if area == 0.0:
                    if f!= 0:
                        fX = fxS[f-1, i] - xOff
                        fY = fyS[f-1, i] - yOff
                        bX = bxS[f-1, i] - xOff
                        bY = byS[f-1, i] - yOff
                        eX = exS[f-1, i] - xOff
                        eY = eyS[f-1, i] - yOff
                        heading = ortS[f-1, i]
                        motion = -1.0
                    else:
                        area = -1.0
                        fX = xOff
                        fY = yOff
                        bX = xOff
                        bY = yOff
                        eX = xOff
                        eY = yOff
                        heading = -181.0
                        motion = -1.0
                        
                else:
                    # Draw contours into Mask Image (1 for Fish, 0 for Background)
                    cv2.drawContours(mask,[largest_cnt],0,1,-1) # -1 draw the contour filled
                    pixelpoints = np.transpose(np.nonzero(mask))
                    
                    # Get Area (again)
                    area = np.size(pixelpoints, 0)
                    
                    # ---------------------------------------------------------------------------------
                    # Compute Frame-by-Frame Motion (absolute changes above threshold)
                    if (f != 0):
                        absdiff = np.abs(diff)
                        absdiff[absdiff < threshold_level] = 0
                        totalAbsDiff = np.sum(np.abs(absdiff))
                        frame_by_frame_absdiff = np.abs(np.float32(previous_ROIs[i]) - np.float32(crop)) / 2 # Adjust for increases and decreases across frames
                        frame_by_frame_absdiff[frame_by_frame_absdiff < threshold_level] = 0
                        motion = np.sum(np.abs(frame_by_frame_absdiff))/totalAbsDiff
                    else:
                        motion = 0
                    
                    # Save Masked Fish Image from ROI (for subsequent frames motion calculation)
                    previous_ROIs[i] = np.copy(crop)
                    
                    # ---------------------------------------------------------------------------------
                    # Find Body and Eye Centroids
                    area = np.float(area)

                    # Highlight 50% of the most different pixels (body)                    
                    numBodyPixels = np.int(np.ceil(area*0.5))
                    
                    # Highlight 8% of the ??? pixels (eyes)     
                    numEyePixels = np.int(np.ceil(area*0.08))
                    
                    # Fish Pixel Values (difference from background)
                    fishValues = diff[pixelpoints[:,0], pixelpoints[:,1]]
                    sortedFishValues = np.sort(fishValues)
                    
#                    bodyThreshold = sortedFishValues[-numBodyPixels]                    
                    eyeThreshold = sortedFishValues[-numEyePixels]

                    # Compute Binary/Weighted Centroids
                    r = pixelpoints[:,0]
                    c = pixelpoints[:,1]
                    all_values = diff[r,c]
                    all_values = all_values.astype(float)
                    r = r.astype(float)
                    c = c.astype(float)
                    
                    # Fish Centroid
                    values = np.copy(all_values)
                    values = (values-threshold_level+1)
                    acc = np.sum(values)
                    fX = np.float(np.sum(c*values))/acc
                    fY = np.float(np.sum(r*values))/acc
                    
                    # Eye Centroid (a weighted centroid) 
                    values = np.copy(all_values)                   
                    values = (values-eyeThreshold+1)
                    values[values < 0] = 0
                    acc = np.sum(values)
                    eX = np.float(np.sum(c*values))/acc
                    eY = np.float(np.sum(r*values))/acc
                    
                    # Find points of a circle around the located eye centroid (7 pixels to get beyond eyes)
                    # Careful! Y-origin is reversed
                    sampleAngle=2.5
                    circleCoords=findCircleEdge(xo=eY,yo=eX,r=7,sampleN=int(360/sampleAngle))
#                    circleValsX=[]
                    circleValsY=[]
                    for idx,a in enumerate(circleCoords):
#                        circleValsX.append(idx*sampleAngle)
                        if a[0]>=crop_height: a[0]=crop_height-1
                        if a[1]>=crop_width: a[1]=crop_width-1
                        circleValsY.append(diff[tuple(a)])
                    # Find peak of circular profile 
                    seed=[]
                    seed.append(circleCoords[np.argmax(circleValsY)][1])
                    seed.append(circleCoords[np.argmax(circleValsY)][0])
                    
                    # Now use this as the seed to find body contour (might be a better way to do this in cv2)
#                    reg=findBodyFromSeed(60,diff,seed)
                    bX = seed[0]
                    bY = seed[1]
#                    # Body Centroid (a binary centroid, excluding "eye" pixels)
#                    values = np.copy(all_values)                   
#                    values[values < bodyThreshold] = 0 #black
#                    values[values >= bodyThreshold] = 1 #white                                                         
#                    values[values > eyeThreshold] = 0  #black                                                          
#                    acc = np.sum(values)
#                    bX = np.float(np.sum(c*values))/acc
#                    bY = np.float(np.sum(r*values))/acc
                    
                    # ---------------------------------------------------------------------------------
                    # Heading (0 deg to right, 90 deg up)
                    if (bY != eY) or (eX != bX):
                        heading = math.atan2((bY-eY), (eX-bX)) * (360.0/(2*np.pi))
                    else:
                        heading = -181.00
            
            # ---------------------------------------------------------------------------------
            # Store data in arrays
            
            # Shift X,Y Values by ROI offset and store in Matrix
            fxS[f, i] = fX + xOff
            fyS[f, i] = fY + yOff
            bxS[f, i] = bX + xOff
            byS[f, i] = bY + yOff
            exS[f, i] = eX + xOff
            eyS[f, i] = eY + yOff
            areaS[f, i] = area
            ortS[f, i] = heading
            motS[f, i] = motion

            # Display?
            if display:
                im = cv2.circle(im, (int(bX + xOff), int(bY + yOff)), 1, (0, 0, 255), 1)
                im = cv2.circle(im, (int(eX + xOff), int(eY + yOff)), 1, (0, 255, 255), 1)
                start_line = (int(bX + xOff), int(bY + yOff))
                heading_radians = 2.0 * np.pi * (heading/360.0)
                end_line = (int(bX + xOff) + int(30 * np.cos(heading_radians)), int(bY + yOff) + int(-30 * np.sin(heading_radians)))
                im = cv2.line(im, start_line, end_line, (0,255,0), 1)

            # -----------------------------------------------------------------
            # Update this ROIs background estimate (everywhere except the (dilated) Fish)
            current_background = np.copy(background_ROIs[i])            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            dilated_fish = cv2.dilate(mask, kernel, iterations = 2)           
            updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
            updated_background[dilated_fish==1] = current_background[dilated_fish==1]            
            background_ROIs[i] = np.copy(updated_background)
                    
        # Report Progress
        if (f%100) == 0:
            bs = '\b' * 1000            # The backspace
            print(bs)
            print (numFrames-f)

            # Display?
            if display:
                debug_image = np.copy(im)
                cv2.imshow("Tracking", debug_image)
                cv2.waitKey(1)

    
    # -------------------------------------------------------------------------
    # Close Video File
    vid.release()

    # Display?
    if display:
        cv2.destroyAllWindows()
    
    # Return tracking data
    return fxS, fyS, bxS, byS, exS, eyS, areaS, ortS, motS
#------------------------------------------------------------------------------


# Return cropped image from ROI list
def get_ROI_crop(image, ROIs, numROi):
    r1 = np.int(ROIs[numROi, 1])
    r2 = np.int(r1+ROIs[numROi, 3])
    c1 = np.int(ROIs[numROi, 0])
    c2 = np.int(c1+ROIs[numROi, 2])
    crop = image[r1:r2, c1:c2]
    
    return crop, c1, r1
    
# Return ROI size from ROI list
def get_ROI_size(ROIs, numROi):
    width = np.int(ROIs[numROi, 2])
    height = np.int(ROIs[numROi, 3])
    
    return width, height

# Return largest (area) contour from contour list
def get_largest_contour(contours):
    # Find contour with maximum area and store it as best_cnt
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    if max_area > 0:
        return best_cnt, max_area
    else:
        return cnt, max_area

def intensities_dist (input_folder, ROIs, numROI, divisor=7):
    background_ROIs = compute_initial_backgrounds(input_folder, ROIs, divisor=divisor)
    raster=background_ROIs[numROI][:]
    centres,hist=np.histogram(raster)
    plt.plot(centres[numROI:],hist)
    

# FIN
