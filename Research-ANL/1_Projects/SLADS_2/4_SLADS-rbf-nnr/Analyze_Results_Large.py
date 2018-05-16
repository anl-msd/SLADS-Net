
#! /usr/bin/env python3
import sys
sys.path.append('code')
import numpy as np
from scipy import misc
from computeOrupdateERD import ComputeRecons
from computeOrupdateERD import FindNeighbors
from computeDifference import computeDifference


Path = '/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170413_EDS_Dictionary_CNN/ResultsAndData/SLADSSimulationResults/'
Folder = 'Experiment_Substance_MicroPowder_res_256_InitM_1_Test_4_Trn_4/'


MeasuredValues = np.load(Path + Folder + 'MeasuredValues.npy')

dim = np.size(MeasuredValues)
for i in range(0, dim):
    if MeasuredValues[i] == 9:
        MeasuredValues[i] = 1
    


MeasuredIdxs = np.load(Path + Folder + '/MeasuredIdxs.npy')
UnMeasuredIdxs = np.load(Path + Folder + '/UnMeasuredIdxs.npy')
PercentageVect = np.array([1,2,3,4,5,10,15,20])
Img = misc.imread(Path + Folder + 'Image_1.tif')
misc.imsave(Path + Folder + 'Original_1.tif', Img.astype(float))
SizeImage = [256,256]

from variableDefinitions import TrainingInfo
TrainingInfo = TrainingInfo()
TrainingInfo.initialize('DWM','DWM',2,10,'Gaussian',4,0.25,15)
Resolution = 1
ImageType = 'D'

for n in range(0,PercentageVect.shape[0]):
    NumSamples = int(PercentageVect[n]*SizeImage[0]*SizeImage[1]/100)
    MeasuredImage = np.zeros([SizeImage[0],SizeImage[1]])
    for i in range(0,NumSamples):
        MeasuredImage[MeasuredIdxs[i,0],MeasuredIdxs[i,1]]=1
    MeasuredIdxs_Tmp = MeasuredIdxs[0:NumSamples]           
    Tmp = MeasuredIdxs[NumSamples:MeasuredIdxs.shape[0]]
    print(Tmp.shape)
    print(UnMeasuredIdxs.shape)
    UnMeasuredIdxs_Tmp = np.append(Tmp,UnMeasuredIdxs,axis=0)  
    
    MeasuredValues_Tmp =   MeasuredValues[0:NumSamples]             
    NeighborValues,NeighborWeights,NeighborDistances = FindNeighbors(TrainingInfo,MeasuredIdxs_Tmp,UnMeasuredIdxs_Tmp,MeasuredValues_Tmp,Resolution)
    ReconValues,ReconImage = ComputeRecons(TrainingInfo,NeighborValues,NeighborWeights,SizeImage,UnMeasuredIdxs_Tmp,MeasuredIdxs_Tmp,MeasuredValues_Tmp)            
    Difference = np.sum(computeDifference(Img,ReconImage,ImageType))                                     
    TD = Difference/(SizeImage[0]*SizeImage[1])
    
    
    name = 'Measured_' + str(PercentageVect[n]) + '.tif'
    misc.imsave(Path + Folder + name, MeasuredImage.astype(float))
    print(TD)
    name = 'Recon_' + str(PercentageVect[n]) + '.tif'
    misc.imsave(Path + Folder + name, ReconImage.astype(float))
    
    
    
    
    
from matplotlib import pyplot as plt

plt.figure()
plt.imshow(MeasuredImage) 
plt.axis('off')

plt.figure()
plt.imshow(ReconImage)    
plt.axis('off')  
    
plt.figure()
plt.imshow(Img) 
plt.axis('off') 
    
    
    
Img_ori = misc.imread(Path + Folder + 'Orig_Image_SEM.png')

plt.figure()
plt.imshow(Img_ori) 
plt.axis('off')

    




## misclassification rate

k = 0.0
for i in range(0, dim):
    if (Img[MeasuredIdxs[i,0], MeasuredIdxs[i,1]] != MeasuredValues[i]):
        k = k + 1

print(k/dim)

























    