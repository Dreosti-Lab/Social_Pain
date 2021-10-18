# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 18:53:41 2021

@author: thoma
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

## Video handling utilities#############################################################    
def cropMovie(aviFile,ROI,outname='Cropped.avi',FPS=100):
    
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    w = np.int(ROI[2])
    h = np.int(ROI[3])
    outFile=r'D:\\Movies\\cache\\'+outname
    out = cv2.VideoWriter(outFile, cv2.VideoWriter_fourcc(*'DIVX'), FPS, (w,h), False)
    for i in range(numFrames):
        im=np.uint8(grabFrame32(vid,i))
        crop, _,_ = get_ROI_cropSingle(im,ROI)
        out.write(crop)
    out.release()
    
    return 0

def trimMovie(aviFile,startFrame,endFrame,saveName):
## Creates a new movie file with 'saveName' and desired start and end frames.    
## INPUTS:  aviFile - string with full path of aviFile
##          startFrame,endFrame - the desired start and end positions of new movie
##          saveName - string with full path of new save location
     
    FPS=120
    vid=cv2.VideoCapture(aviFile)
    width=int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if(endFrame==-1) or (endFrame>numFrames):
        endFrame=numFrames
        
    out = cv2.VideoWriter(saveName,cv2.VideoWriter_fourcc(*'DIVX'), FPS, (width,height), False)
    setFrame(vid,startFrame)
    
    for i in range(endFrame-startFrame):
        ret, im = vid.read()
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        out.write(im)
        
    out.release()
    vid.release()
    return 0

def setFrame(vid,frame):
## set frame of a cv2 loaded movie without having to type the crazy long cv2 command
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    return 0

def grabFrame(avi,frame):
# grab frame and return the image from loaded cv2 movie
    vid=cv2.VideoCapture(avi)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid.release()
    im = np.uint8(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im

def grabFrame32(vid,frame):
# grab frame and return the image (float32) from loaded cv2 movie
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    im = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im
   
def showFrame(vid,frame):
# display selected frame (greyscale) of a cv2 loaded movie (for testing)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    im = np.float32(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    plt.figure()
    plt.imshow(im)
    return 0

def get_ROI_cropSingle(image, ROI):
    r1 = np.int(ROI[1])
    r2 = np.int(r1+ROI[3])
    c1 = np.int(ROI[0])
    c2 = np.int(c1+ROI[2])
    crop = image[r1:r2, c1:c2]
    
    return crop, c1, r1