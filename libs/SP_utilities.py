# -*- coding: utf-8 -*-
"""
Social Pain "Utilities"

"""
# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.signal as signal
import glob
import cv2
import imageio
import SP_Analysis as SPA
#-----------------------------------------------------------------------------
# Utilities for loading and ploting "social pain" data
def getTracking(path):
    data = np.load(path)
    tracking = data['tracking']
    fx = tracking[:,0] 
    fy = tracking[:,1]
    bx = tracking[:,2]
    by = tracking[:,3]
    ex = tracking[:,4]
    ey = tracking[:,5]
    area = tracking[:,6]
    ort = tracking[:,7]
    motion = tracking[:,8]
    
    return fx,fy,bx,by,ex,ey,area,ort,motion
# Read Folder List file 6 fish 
def read_folder_list(folderlistFile): 
    folderFile = open(folderlistFile, "r") #"r" means read the file
    folderList = folderFile.readlines() # returns a list containing the lines
    folderPath = folderList[0][:-1] # Read Data Path which is the first line
    folderList = folderList[1:] # Remove first line because it contains the path

    # Set Data Path where the experiments are located
    data_path = folderPath
    
    numFolders = len(folderList) 
    groups = np.zeros(numFolders)
    ages = np.zeros(numFolders)
    folderNames = [] # We use this because we do not know the exact lenght
    fishStatus = np.zeros((numFolders, 6))
    
    for i, f in enumerate(folderList):  #enumerate tells you what folder is 'i'
        stringLine = f[:-1].split()
        groups[i] = int(stringLine[0])
        ages[i] = int(stringLine[1])
        expFolderName = data_path + stringLine[2]
        folderNames.append(expFolderName)
        fishStat = [int(stringLine[3]), int(stringLine[4]), int(stringLine[5]), int(stringLine[6]), int(stringLine[7]), int(stringLine[8])]    
        fishStatus[i,:] = np.array(fishStat)
        
    return groups, ages, folderNames, fishStatus
 
def get_folder_names(folder):
    # Specifiy Folder Names
    NS_folder = folder + '/Non_Social'
    S_folder = folder + '/Social'
    Analysis = folder + '/Analysis'
    
    
    return NS_folder, S_folder, Analysis

def get_folder_name(folder):
    # Specifiy Folder Names
    NS_folder = folder + '/Non_Social'
    S_folder = folder + '/Non_Social'
    
    return NS_folder, S_folder
    

def load_video (folder):
    
    aviFiles = glob.glob(folder+'/*.avi')#finds any avi file in the folder
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return vid, numFrames, width, height


# write function to extend ROI to include cue area
# Each ROI is 4 numbers: x start position, y start position; top left, width (x) and height (y)
def get_cue_ROI(ROIs,w=65,offset=-10):
    cueROIs=[]
    for roi in ROIs:
        x=roi[0]+roi[2]+offset
        y=roi[1]
        h=roi[3]
        cueROIs.append([x,y,w,h])
        
    return np.array(cueROIs)

def grabFrame(avi,frame):
# grab frame and return the image from loaded cv2 movie
    vid=cv2.VideoCapture(avi)
    vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, im = vid.read()
    vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
    vid.release()
    im = np.uint8(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    return im

# Scripts to find circle edges given origin and radius
def removeDuplicates(lst):
      
    return [t for t in (set(tuple(i) for i in lst))]

def smoothSignal(x,N=5):
## performs simple 'box-filter' of any signal, with a defined box kernel size
    xx=np.convolve(x, np.ones((int(N),))/int(N), mode='valid')
    n=N-1
    xpre=np.zeros(n)
    xxx=np.concatenate((xpre,xx))
    return xxx

def convert_mm(XList,YList,ROI):
    
    # Rescale by chamber dimensions
    chamber_Width_px = ROI[2]
    chamber_Height_px = ROI[3]
    chamber_Width_mm = 100
    chamber_Height_mm = 15
    
    XList = XList - ROI[0]
    YList = YList - ROI[1]

    XList_mm = (XList/chamber_Width_px)*chamber_Width_mm
    YList_mm = (YList/chamber_Height_px)*chamber_Height_mm  
    
    return XList_mm, YList_mm

def plotMotionMetrics(fx,fy,bx,by,ex,ey,area,ort,motion,startFrame,endFrame):
## plots tracking trajectory, motion, distance per frame and cumulative distance for defined section of tracking data
    
    plt.figure()
    plt.plot(fx[startFrame:endFrame],fy[startFrame:endFrame])
    plt.title('Tracking')
    
    smoothedMotion=smoothSignal(motion[startFrame:endFrame],100)
    plt.figure()
    plt.plot(smoothedMotion)
    plt.title('Smoothed Motion')
    
    distPerFrame,cumDistPerFrame=SPA.computeDistPerFrame(fx[startFrame:endFrame],fy[startFrame:endFrame])
    plt.figure()
    plt.plot(distPerFrame)
    plt.title('Distance per Frame')
    
    xx=smoothSignal(distPerFrame,30)
    plt.figure()
    plt.plot(xx[startFrame:endFrame])
    plt.title('Smoothed Distance per Frame (30 seconds)')
    
    
    plt.figure()
    plt.plot(cumDistPerFrame)
    plt.title('Cumulative distance')    
    
    return cumDistPerFrame

# FIN