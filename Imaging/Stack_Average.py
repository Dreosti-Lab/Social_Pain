
# -*- coding: utf-8 -*-
"""
This script calculates the average of .nii warped red images

@author: Dreosti Lab
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt


folder_path = 'S:/WIBR_Dreosti_Lab/Alizee/LSZ1/Registration/Peptides/512_2/TRPM3'

folder_list = (folder_path  + '/Average.txt')

folder_file = open(folder_list, "r") #"r" means read the file
file_list = folder_file.readlines() # returns a list containing the lines

num_fish = len(file_list) 
Average_images = np.zeros((512, 512, 319), dtype = np.float32)
Divisor = np.zeros((512, 512, 319), dtype = np.float32)
Valid = np.zeros((512, 512, 319), dtype = np.float32)

for f in file_list:
    img_file = folder_path + f[:-1]  # to remove the space(new line character) in the txt file 

    Image = nib.load(img_file)
    
    Image_size = Image.shape
    Image_type = Image.get_data_dtype()
    #Image_data = Image.dataobj[...,0]
    Image_data = Image.get_fdata()
    
   
    Valid = (Image_data != 0.0).astype(int)
    
    Divisor = Divisor + Valid
    
    Average_images = Average_images + Image_data

Average_images = Average_images/Divisor


new_img = nib.Nifti1Image(Average_images, Image.affine, Image.header)

nib.save(new_img, folder_path +  "/TRPM3.nii.gz")



# FIN
