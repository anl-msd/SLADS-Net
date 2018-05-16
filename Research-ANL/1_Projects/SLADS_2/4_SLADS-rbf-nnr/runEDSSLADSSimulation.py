#! /usr/bin/env python3
import numpy as np
import tensorflow as tf
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('code')
###############################################################################
############## USER INPUTS: L-0 ###############################################
###############################################################################

# Name of folder to save result in    
FolderName = 'Example_1'

# Type of Image: D - for discrete (classified) image; C - for continuous
ImageType = 'D'

# Image extention
ImageExtension = '.tif'

# If TestingImageSet_X used enter 'X'
TestingImageSet = '1'    

# If TrainingDB_Y was used for training enter 'Y'          
TrainingImageSet = '1'

# Image resolution in pixels   
SizeImage = [1024,1024]

# Value of c found in training     
c=8

# Maximum  sampling percentage 
StoppingPercentage = 10
# If you want to use stopping function used, enter threshold (from Training), 
# else leave at 0      
StoppingThrehsold = 0

# Clasification setting
Classify = 'EDS'
# 'N' - no classification needed (already classified or continuous data)
# '2C' - perform 2 class classification using otsu
# 'MC' - perform multi-class classification
# if 'MC' open .code/runSLADSOnce.py and search "elif Classify=='MC':" and  
# include your classification method (in all four locations) as instructed
                            
# Initial Mask for SLADS:
# Percentage of samples in initial mask
PercentageInitialMask = 1
# Type of initial mask   
MaskType = 'R'
# Choices: 
    # 'U': Uniform mask; can choose any percentage
    # 'R': Randomly distributed mask; can choose any percentage
    # 'H': low-dsicrepacy mask; can only choose 1% mask
# Batch Sampling
BatchSample = 'N'           
# If 'Y' set number of samples in each step in L-1 (NumSamplesPerIter)

PlotResult='Y'

###############################################################################
############## USER INPUTS: L-1 ###############################################
############################################################################### 

# The number of samples in each step  
NumSamplesPerIter = 10

# Update ERD or compute full ERD in SLADS
# with Update ERD, ERD only updated for a window surrounding new measurement
Update_ERD = 'Y' 
# Smallest ERD update window size permitted
MinWindSize = 3  
# Largest ERD update window size permitted  
MaxWindSize = 15  

# EDS Data
NumSpectra = 12
Folder = 'PbSn'
NoiseType = 'P'
Noiselambda = 2
ErrorSpectrumProb=0.01

###############################################################################
############################ END USER INPUTS ##################################
############################################################################### 
#global sess
#global W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2
#global x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv

import tfmodel as tfc
sess, W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv = tfc.tfClassify(Folder)

from variableDefinitions import tfclfstruct
        
tfclf = tfclfstruct(sess, W_conv1, W_conv2, W_fc1, W_fc2, W_fc3, W_fco, b_conv1, b_conv2, x_image, x, y_, h_conv1, h_conv2, h_pool1, h_pool2, size_hp, h_flat, h_fc1, h_fc2, h_fc3, keep_prob, h_fc1_drop, y_conv) 

from runEDSSLADSSimulationScript import runEDSSLADSSimulationScript
runEDSSLADSSimulationScript(FolderName,ImageType,ImageExtension,TestingImageSet,TrainingImageSet,SizeImage,c,StoppingPercentage,StoppingThrehsold,Classify,PercentageInitialMask,MaskType,BatchSample,PlotResult,NumSamplesPerIter,Update_ERD,MinWindSize,MaxWindSize,NumSpectra,Folder,NoiseType,Noiselambda,ErrorSpectrumProb, tfclf)






























