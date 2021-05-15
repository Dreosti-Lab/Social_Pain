# -*- coding: utf-8 -*-
"""
Processing Social_Pain videos : 
    - Compute Backgrounds for each ROIs 
    - PreProcessing Average difference between Background and Frame 
    - ImprovedTracking Current Frame - Background Image / find fish Eyes (dark) + Body (Bright)
"""
# Set Library Path - Social_Pain Repos
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)
# Set Base Path
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'

# Import useful libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import glob
import cv2

import BONSAI_ARK
import SP_Utilities as SPU
import SP_Video as SPV
                      
# Read Folder List
FolderlistFile = base_path + r'\Experiment_17\Folderlist\debug.txt'
groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)

    # ---------------------
    # Process Video (NS)
    bonsaiFiles = NS_folder + r'/Bonsai_ROI_Analysis.bonsai'
    ROIs = BONSAI_ARK.read_bonsai_crop_rois(bonsaiFiles)
    input_folder = NS_folder
    output_folder = NS_folder
#------------------------------------------------------------------------------
    # Load Video
    aviFiles = glob.glob(NS_folder+'/*.avi')#finds any avi file in the folder
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-200 # Skip, possibly corrupt, last 200 frames (2 seconds)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Allocate space for all ROI backgrounds
    background_ROIs = []
    for i in range(0,1):
         w = np.int(ROIs[i, 2])
         h = np.int(ROIs[i, 3])
      
         background_ROIs.append(np.zeros((h, w), dtype = np.float32))
    
    # Find initial background for each ROI
    for i in range(0,1):  


        # Allocate space for background stack
        crop_width = np.int(ROIs[i, 2])
        crop_height = np.int(ROIs[i, 3])
        stepFrames = 1000 # Check background frame every 10 seconds
        bFrames = 20
        backgroundStack = np.zeros((crop_height, crop_width, bFrames), dtype = np.float32)
        background = np.zeros((crop_height, crop_width), dtype = np.float32)
        previous = np.zeros((crop_height, crop_width), dtype = np.float32)
        
        # Store first frame
        vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, im = vid.read()
        current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        yOff = np.int(ROIs[i, 1])
        r2 = np.int(yOff+ROIs[i, 3])
        xOff = np.int(ROIs[i, 0])
        c2 = np.int(xOff+ROIs[i, 2])
        crop = current[yOff:r2, xOff:c2]
        backgroundStack[:,:,0] = np.copy(crop)
        previous = np.copy(crop)
        bCount = 1
    
        
        # Search for useful background frames (significantly different than previous)
        changes = []
        for f in range(stepFrames, numFrames, stepFrames):

            # Read frame
            vid.set(cv2.CAP_PROP_POS_FRAMES, 52078)
            ret, im = vid.read()
            current = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
            yOff = np.int(ROIs[i, 1])
            r2 = np.int(yOff+ROIs[i, 3])
            xOff = np.int(ROIs[i, 0])
            c2 = np.int(xOff+ROIs[i, 2])
            crop = current[yOff:r2, xOff:c2]
        
            # Measure change from current to previous frame
            absdiff = np.abs(previous-crop)
            level = np.median(crop)/7
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


    # Algorithm
    # 1. Find initial background guess for each ROI
    # 2. Extract Crop regions from ROIs
    # 3. Threshold ROI using median/5 of each crop region, Binary Close image using 5 rad disc
    # 4. Find largest particle (Contour)
    # 5. - Compute Weighted Centroid (X,Y) for Eye Region (10% of brightest pixels)
    # 6. - Compute Binary Centroid of Body Region (50% of brightest pixels - eyeRegion)
    # 7. - Compute Heading
        # Allocate ROI (crop region) space
    previous_ROIs = []
    for i in range(0,1):
        w = np.int(ROIs[i, 2])
        h = np.int(ROIs[i, 3])
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
        
    
 
    vid.set(cv2.CAP_PROP_POS_FRAMES, 52078)
    # Read next frame        
    ret, im = vid.read()
    # Convert to grayscale (uint8)
    current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
    plt.figure()
    # Process each ROI
    for i in range(0,1):
        
        print('Processing ROI ' + str(i+1))            
        yOff = np.int(ROIs[i, 1])
        r2 = np.int(yOff+ROIs[i, 3])
        xOff = np.int(ROIs[i, 0])
        c2 = np.int(xOff+ROIs[i, 2])
        crop = current[yOff:r2, xOff:c2]
        crop_height, crop_width = np.shape(crop)

        # Difference from current background: within crop, the fish appears brighter than the ROI background
        # The difference betwwen the crop and the computed background is the fish
        diff = crop - background_ROIs[i]

        # Determine current threshold (Sensitivity : detected range of pixel change)
        # How different does it need to be for it to be picked up as the fish, lower threshold if you want to track fish that are not moving much 
        threshold_level = np.median(background_ROIs[i])/4        
              
        # Within the diff (fish) our threshold value is the median background and max value is 255 (White)
        #if pixel > threshold_level = white and if < threshold_level = Black
        level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
        # Convert to uint8 : instead of black or white : 0 and 1
        threshold = np.uint8(threshold)
        
        # Binary Close: Closing eliminates noise ie. closing small holes inside object (black spots)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
        
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
            largest_cnt, area = SPV.get_largest_contour(contours)
            
            
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
        # plt.subplot(2,3,i+1)
        # plt.imshow(mask)
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
                numBodyPixels = np.int(np.ceil(area/1.2))
                
                # Highlight 10% of the ??? pixels (eyes)     
                numEyePixels = np.int(np.ceil(area/20))
                
                # Fish Pixel Values (difference from background)
                fishValues = diff[pixelpoints[:,0], pixelpoints[:,1]]
                sortedFishValues = np.sort(fishValues)
                
                bodyThreshold = sortedFishValues[-numBodyPixels]                    
                eyeThreshold = sortedFishValues[numEyePixels]

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

                # Body Centroid (a binary centroid, excluding "eye" pixels)
                values = np.copy(all_values)                   
                values[values < bodyThreshold] = 0 #black
                values[values >= bodyThreshold] = 1 #white                                                         
                #values[values > eyeThreshold] = 1  #black                                                          
                acc = np.sum(values)
                
                bX = np.float(np.sum(c*values))/acc
                bY = np.float(np.sum(r*values))/acc
                
                    
                
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
            
            # -----------------------------------------------------------------
            # Update this ROIs background estimate (everywhere except the (dilated) Fish)
            current_background = np.copy(background_ROIs[i])            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            dilated_fish = cv2.dilate(mask, kernel, iterations = 2)           
            updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
            updated_background[dilated_fish==1] = current_background[dilated_fish==1]            
            background_ROIs[i] = np.copy(updated_background)
            
        # ---------------------------------------------------------------------------------
    # Plot All Fish in Movie with Tracking Overlay
    if (f % 100 == 0):
        plt.clf()
        enhanced = cv2.multiply(current, 1)
        color = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        plt.imshow(color)
        plt.axis('image')
        for fish_index in range(0,6):
            plt.scatter(fxS[f, fish_index],fyS[f, fish_index],s=10, c='b')
            #plt.plot(exS[f, i],eyS[f, i],'r', markersize = 3)
            plt.scatter(bxS[f, i],byS[f, fish_index],s=5, c='r')
           
        plt.draw()
        plt.pause(0.001)
            
        # ---------------------------------------------------------------------------------
        # Save Tracking Summary
        if(f == 0):
            plt.savefig(output_folder+'/initial_tracking.png', dpi=300)
            plt.figure('backgrounds')
            for fish_index in range(0,6):
                plt.subplot(2,3,i+1)
                plt.imshow(background_ROIs[fish_index])
            plt.savefig(output_folder+'/initial_backgrounds.png', dpi=300)
            plt.close('backgrounds')
        if(f == numFrames-1):
            plt.savefig(output_folder+'/final_tracking.png', dpi=300)
            plt.figure('backgrounds')
            for fish_index in range(0,6):
                plt.subplot(2,3,i+1)
                plt.imshow(background_ROIs[fish_index])
            plt.savefig(output_folder+'/final_backgrounds.png', dpi=300)
            plt.close('backgrounds')

        # Report Progress
        if (f%100) == 0:
            bs = '\b' * 1000            # The backspace
            print(bs)
            print (numFrames-f)
    
    # -------------------------------------------------------------------------
    # Close Video File
    vid.release()


#FIN
