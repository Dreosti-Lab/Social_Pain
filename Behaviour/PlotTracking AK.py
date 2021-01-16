# -*- coding: utf-8 -*-
"""
Plot Fish Tracking (example)
"""

# -----------------------------------------------------------------------------
# Set "Library Path" - Social_Pain Repos
lib_path = r'C:/Repos/Dreosti-Lab/Social_Pain/libs'

# Set Library Path
import sys
sys.path.append(lib_path)

# -----------------------------------------------------------------------------
# Set Base Path
base_path = r'V:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'

# Import useful libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc as misc
from scipy import stats
import seaborn as sns

# Import local modules
import SP_utilities as SPU


# Set folder
FolderlistFile = base_path + r'\Experiment_3\Folderlist\Exp_3.txt'

groups, ages, folderNames, fishStatus = SPU.read_folder_list(FolderlistFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis_folder = SPU.get_folder_names(folder)

    # ---------------------
    # Analyze Tracking for each fish 
    for i in range(0,6):
        fish_number = i + 1
     
     # Extract tracking data (NS)
     
        tracking_file = NS_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
         
        fx_NS = tracking[:,0] 
        fy_NS = tracking[:,1]
        bx_NS = tracking[:,2]
        by_NS = tracking[:,3]
        ex_NS = tracking[:,4]
        ey_NS = tracking[:,5]
        area_NS = tracking[:,6]
        ort_NS = tracking[:,7]
        motion_NS = tracking[:,8]
          
   
        # Extract tracking data (S)
         
        tracking_file = S_folder + r'/tracking' + str(fish_number) +'.npz'
        data = np.load(tracking_file)
        tracking = data['tracking']
         
         # Extract tracking data
        fx_S = tracking[:,0] 
        fy_S = tracking[:,1]
        bx_S = tracking[:,2]
        by_S = tracking[:,3]
        ex_S = tracking[:,4]
        ey_S = tracking[:,5]
        area_S = tracking[:,6]
        ort_S = tracking[:,7]
        motion_S = tracking[:,8] 

         # Filter out bad data
        min_x = 250
        max_x = 1000
      
        if fish_number == 1:
            min_y = 120
            max_y = 165
        if fish_number == 2:
            min_y = 260
            max_y = 305
        if fish_number == 3:
            min_y = 400
            max_y = 445
        if fish_number == 4:
            min_y = 540
            max_y = 585
        if fish_number == 5:
            min_y = 680
            max_y = 725
        if fish_number == 6:
            min_y = 820
            max_y = 865    
 
         #Find good tracking 
        num_total_frames_NS = len(fx_NS)
        good_frame_NS = (fx_NS > min_x) * (fx_NS < max_x) * (fy_NS > min_y) * (fy_NS < max_y)
        num_good_frames_NS = np.sum(good_frame_NS)
        lost_frame_NS = num_total_frames_NS-num_good_frames_NS
        
         
        num_total_frames_S = len(fx_S)
        good_frame_S = (fx_S > min_x) * (fx_S < max_x) * (fy_S > min_y) * (fy_S < max_y)
        num_good_frames_S = np.sum(good_frame_S)
        lost_frame_S = num_total_frames_S-num_good_frames_S
         
        lost_frame = lost_frame_S + lost_frame_NS

        plt. figure ()
        plt.plot(fx_NS[good_frame_NS], fy_NS[good_frame_NS], 'b.', alpha = 0.15)
        plt.plot(fx_S[good_frame_S], fy_S[good_frame_S], 'm.', alpha = 0.15)  
        plt.title("Fish #{0}- Lost Frames {1}".format(fish_number, lost_frame))
     
        for i in range(0,6):
            plt.savefig(Analysis_folder + '/tracking_summary'+ str(fish_number) + '.png', dpi=300)
       
        # SPM Score
        SPM = np.mean(fx_NS[good_frame_NS]) - np.mean(fx_S[good_frame_S])
        print(SPM)
        
        x = [-21.56661701,
-9.764888107, 
18.7915388, 
-252.1871488,
-185.654834, 
-94.85879695,
99.12376909, 
-26.74959896, 
52.36924335, 
-10.26304963
-327.9338558,
55.21552345,
0.327159166,
2.685205203,
-51.47853443,
-57.48042053,
-148.9089504,
-2.804369766,
25.09621421,
0.979425583,
-123.9042566,
-8.712607414,
-177.2677854,
105.6516658,
-55.54692246,
-92.92125595,
-81.4303552,
-208.6113573,
77.33483599,
-43.40925851,
-6.654612779,
-58.04324023,
0.088248158,
-570.452956,
-532.4735466,
-504.6501395,
]
        
        plt.hist(x, bins = 30, color = '#468189')
        plt.title('SPM for each Fish')
        sns.despine()
        plt.show()




