# -*- coding: utf-8 -*-
"""
Check immediately if the tracking has worked: generates a summary background image + Difference
"""
# Set Library Path
lib_path = r'C:/Repos/Social_Pain/libs'
import sys
sys.path.append(lib_path)

# Set Base Path
base_path = r'S:/WIBR_Dreosti_Lab/Alizee/Behaviour_Heat_Gradient'
#base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'

# Import local modules
import SP_video_TRARK as SPV
import SP_utilities as SPU

# Read Folder List
folderListFile = base_path + r'/NewChamber/Isolated_Habituation_NewChamber/Folderlist_Habituation.txt'

control = False
groups, ages, folderNames, fishStatus = SPU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)
            
    # Process Video (Non_Social)
    SPV.pre_process_video(NS_folder, False)
    # Process Video (Social)
    SPV.pre_process_video(S_folder, True)
       
    # Report Progress
    print (groups[idx])   
    
# FIN
    