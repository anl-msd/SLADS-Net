
#! /usr/bin/env python3
import sys
sys.path.append('code')
import numpy as np
from scipy import misc
from computeOrupdateERD import ComputeRecons
from computeOrupdateERD import FindNeighbors
from computeDifference import computeDifference



path = 'Example_1'
ImagePath = 'TestingImageSet_1'
SizeImage = [128,128]


MeasuredValues = np.load('ResultsAndData/SLADSSimulationResults/' + path + '/MeasuredValues.npy')
MeasuredIdxs = np.load('ResultsAndData/SLADSSimulationResults/' + path + '/MeasuredIdxs.npy')
UnMeasuredIdxs = np.load('ResultsAndData/SLADSSimulationResults/' + path + '/UnMeasuredIdxs.npy')
PercentageVect = np.array([5,10,15,20,25,30,35,40])
Img = misc.imread('ResultsAndData/TestingImages/' + ImagePath + '/Image_1.png')
misc.imsave('ResultsAndData/SLADSSimulationResults/' + path + '/Original.png',Img)

from variableDefinitions import TrainingInfo
TrainingInfo = TrainingInfo()
TrainingInfo.initialize('DWM','DWM',2,10,'Gaussian',4,0.25,15)
Resolution = 1
ImageType = 'C'


TDvec = np.zeros(np.size(PercentageVect))
PSNRvec = np.zeros(np.size(PercentageVect))

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
    
    Difference2 = np.sum(np.square(computeDifference(Img,ReconImage,ImageType))) 
    mse = Difference2/(SizeImage[0]*SizeImage[1])
    PSNR = 10*np.log10(np.square(255)/mse)
    
    
    
    name = 'ResultsAndData/SLADSSimulationResults/' + path + '/Measured_' + str(PercentageVect[n]) + '.png'
    misc.imsave(name,MeasuredImage)
    print(TD)
    name = 'ResultsAndData/SLADSSimulationResults/' + path + '/Recon_' + str(PercentageVect[n]) + '.png'
    misc.imsave(name,ReconImage)
    
    TDvec[n] = TD
    PSNRvec[n] = PSNR

from matplotlib import pyplot as plt    
plt.plot(TDvec)
plt.plot(PSNRvec)

    
np.save('ResultsAndData/SLADSSimulationResults/' + path + '/TDvec.npy', TDvec)    
np.save('ResultsAndData/SLADSSimulationResults/' + path + '/PSNRvec.npy', PSNRvec)    












    