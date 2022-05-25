# -*- coding: utf-8 -*-
"""
PreProcessing check immediately if the tracking has worked: Summary Background image + Difference
"""
# Set Library Path - Social Pain Repo
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
folderListFile = base_path + r'\Heat_NewChamber38\Folderlist_NewChamber_HEAT38.txt'

control = False
groups, ages, folderNames, fishStatus = SPU.read_folder_list(folderListFile)

# Bulk analysis of all folders
for idx,folder in enumerate(folderNames):
    
    # Get Folder Names
    NS_folder, S_folder, Analysis = SPU.get_folder_names(folder)
            
    # Process Video (NS)
    SPV.pre_process_video(NS_folder, False)
    # Process Video (S)
    SPV.pre_process_video(S_folder, True)
       
    # Report Progress
    print (groups[idx])   
    
# FIN
    