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

def load_video (folder):
    
    aviFiles = glob.glob(folder+'/*.avi')#finds any avi file in the folder
    aviFile = aviFiles[0]
    vid = cv2.VideoCapture(aviFile)
    numFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))-100 # Skip, possibly corrupt, last 100 frames (1 second)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    return vid, numFrames, width, height


## Identifies and reverses sudden flips in orientation caused by errors in tracking the eyes vs the body resulting in very high frequency tracking flips     
def filterTrackingFlips(ort):
    dAngle = np.diff(ort)
    new_dAngle=[]    
    count=0
    for a in dAngle:
        if a < -90:
            new_dAngle.append(a + 180)
            count+=1
        elif a > 90:
            new_dAngle.append(a - 180)
            count+=1
        else:
            new_dAngle.append(a)
    
        new_dAngle = np.array(new_dAngle)
        ort_new = np.r_[ort[0], new_dAngle].cumsum()
        ort_new = ort_new[1:]
        new_dAngle= list(new_dAngle)
        
    return count, ort_new

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