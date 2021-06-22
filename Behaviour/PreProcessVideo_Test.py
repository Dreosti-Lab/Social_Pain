# -*- coding: utf-8 -*-
"""
PreProcessing check immediately if the tracking has worked: Summary Background image + Difference
"""
# Set Library Path - Social Pain Repo
lib_path = r'C:\Repos\Social_Pain\libs'
import sys
sys.path.append(lib_path)
# Set Base Path
base_path = r'S:\WIBR_Dreosti_Lab\Alizee\Behaviour_Heat_Gradient'

# Import local modules
import SP_Video as SPV
import SP_Utilities as SPU

# Read Folder List
File = base_path + r'\Test_Food\16_06_2021\Dry_Paramecia_Rotifer'

# Process Video (NS)
SPV.pre_process_video(File, False)

# FIN
    