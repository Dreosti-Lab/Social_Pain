# -*- coding: utf-8 -*-
"""
Created on Mon May 23 12:07:19 2022

@author: Tom
"""
# import libraries, including AZU
import BONSAI_ARK as BON # bonsai first to grab local version (not from Lonely_Fish or Arena_Zebrafish)
TR_lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Arena_Zebrafish\libs'
SZ_lib_path = r'S:\WIBR_Dreosti_Lab\Tom\Github\Lonely_Fish_TR\Libraries'
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt

sys.path.append(SZ_lib_path)
sys.path.append(TR_lib_path)

import AZ_utilities as AZU
import SZ_utilities as SZU
import SZ_video as SZV
import cv2

# write function to extend ROI to include cue area
# Each ROI is 4 numbers: x start position, y start position; top left, width (x) and height (y)
def get_cue_ROI(ROIs,w=65):
    cueROIs=[]
    for roi in ROIs:
        x=roi[0]+roi[2]-10
        y=roi[1]
        h=roi[3]
        cueROIs.append([x,y,w,h])
        
    return np.array(cueROIs)

def compute_motion(folder,ROIs):
    
    # First steps are same as usual tracking
    background_ROIs = SZV.compute_intial_backgrounds(folder, ROIs)
    
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
        w, h = SZV.get_ROI_size(ROIs, i)
        previous_ROIs.append(np.zeros((h, w), dtype = np.uint8))
    motS = np.zeros((numFrames,6))
        
    # skip sometimes corrupt first 100 frames
    AZU.setFrame(vid,100)
    for f in range(0,numFrames): 
        # Read next frame        
        ret, im = vid.read()
        
        # Convert to grayscale (uint8)
        current = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # loop through ROIs
        for i in range(0,6):
            # Extract Crop Region
            crop, xOff, yOff = SZV.get_ROI_crop(current, ROIs, i)
            crop_height, crop_width = np.shape(crop)

            # Difference from current background
            diff = background_ROIs[i] - crop
            
            # Determine current threshold
            threshold_level = np.median(diff)+(3*np.std(diff)) # 3 standard deviations above the median (yours might be the median + the median/7 or similar)
   
            # # Threshold and find contours - use if you want to filter out where there was no fish found (not sure how this will behave with multiple fish)           
            # level, threshold = cv2.threshold(diff,threshold_level,255,cv2.THRESH_BINARY)
            
            # # Convert to uint8
            # threshold = np.uint8(threshold)
            
            # # Binary Close
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            # closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel)
            
            # # Find Binary Contours            
            # contours, _ = cv2.findContours(closing,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            
            # # if no contours, skip
            # if len(contours) == 0:
            #     motion=-1
            # possibly put in an area check to make sure particle is big enough...
            
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
            
            # Update this ROIs background estimate (everywhere except the (dilated) Fish)
            
            # current_background = np.copy(background_ROIs[i])            
            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
            # dilated_fish = cv2.dilate(mask, kernel, iterations = 2)           
            # updated_background = (np.float32(crop) * 0.01) + (current_background * 0.99)
            # updated_background[dilated_fish==1] = current_background[dilated_fish==1]            
            # background_ROIs[i] = np.copy(updated_background)
    return motS


plot = False
# cycle through folderList
folderListFile='S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient/Heat/Folderlist.txt'

# Read Folder List
_, _, folderNames, fishStatus, _ = SZU.read_folder_list(folderListFile)

for idx,folder in enumerate(folderNames):

    save_path=folder+'\\ROIs'
    AZU.cycleMkDir(save_path)
    # Get Folder Names
    _, S_folder, _ = SZU.get_folder_names(folder)
    
    # extract ROIs
    ROI_filename=glob.glob(S_folder+'\*.bonsai')
    ROIs=BON.read_bonsai_crop_rois(ROI_filename[0])

    # construct 'cue' ROI
    cue_ROIs=get_cue_ROI(ROIs)
    
    # print an image with ROIs to check
    if plot:
        # grab the image
        aviFiles = glob.glob(S_folder+'/*.avi')
        aviFile = aviFiles[0]
        vid = cv2.VideoCapture(aviFile)
        img=AZU.grabFrame(aviFile,500)
        
        color = (255,0)
        weight=10
        for roi in cue_ROIs:
            
            # find corners
            tl = (int(roi[0]),int(roi[1]))
            tr = (int(roi[0]+roi[2]), int(roi[1]))
            bl = (int(roi[0]), int(roi[1]+roi[3]))
            br = (int(roi[0]+roi[2]),int(roi[1]+roi[3]))

            # draw the lines            
            cv2.line(img,tl,tr,color,thickness=weight)
            cv2.line(img,tl,bl,color,thickness=weight)
            cv2.line(img,tr,br,color,thickness=weight)
            cv2.line(img,bl,br,color,thickness=weight)
            
            plt.figure()
            plt.imshow(img)
            savename=save_path+'\\cue_ROIs.png'
            plt.savefig(savename,dpi=300)
            plt.close()
            
    # compute motion for all ROIs
    cue_motion_S=compute_motion(S_folder,cue_ROIs)
    avg_cue_motion_S = np.mean(cue_motion_S)

    
    
    # save as seperate npy for each ROI
    for idr,roi in enumerate(ROIs):
        savename=save_path+'\\cue_motion_'+str(idr)+'.npz'
        np.save(savename,np.array(cue_motion_S[:,idr]))
        
        

        
        
       
        
    