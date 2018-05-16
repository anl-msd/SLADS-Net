#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 13:28:33 2017

@author: yzhang
"""

import numpy as np
from matplotlib import pyplot as plt
import sys
import os



EDStrain = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDStrain.npy')
EDStest = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDStest.npy')
EDSlabel = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/EDSlabel.npy')

num = EDStrain.shape[0]
dim = EDStrain.shape[1]




from sklearn.neural_network import MLPClassifier as nn

clf = nn(activation='logistic', solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 3), random_state=1, max_iter=1000)
clf.fit(EDStrain, EDSlabel)

t = len(EDSlabel)
tlabel = np.zeros(t)
tlabel_prob = np.zeros((t,2))
e_rate = np.zeros(t)

x_test = (EDStest)

for i in range(0,t):
    tlabel[i] = clf.predict(x_test[i,:])
    #tlabel_prob[i] = clf.predict(x_test[i,:])
    if tlabel[i].astype(int) == EDSlabel[i].astype(int):
        e_rate[i] = 0
    else:
        e_rate[i] = 1
error = np.sum(e_rate)/t*100

print('success rate =', 100-error, '%')
print(tlabel)
print(tlabel_prob)






y_tar = np.zeros((24, 100))
for i in range(0, 100):
    y_tar[:, i] = i

from sklearn.neural_network import MLPRegressor as nnr

reg = nnr(activation='relu', solver='adam', alpha=1e-5, hidden_layer_sizes=(100, 5), random_state=1, max_iter=500)
reg.fit(EDStrain, y_tar)


noise3 = np.random.poisson(lam=20.0, size=(EDStest.shape))


y_reg = reg.predict(EDStest)
y_regnoise = reg.predict(noise3)

t=1
t1=t
t2=t+12

plt.figure()
plt.plot(y_tar[t, :], label='target')
plt.plot(y_reg[t1,:], label='EDS-class-1')
#plt.figure()   
plt.plot(y_reg[t2,:], label='EDS-class-2')
plt.plot(y_regnoise[t,:], label='Error case')
#plt.figure() 
plt.legend(bbox_to_anchor=(0.01, 0.98), loc=2, borderaxespad=0.)  
plt.show()



sig_true = np.var(np.abs(y_reg[t1,:]-y_tar))

sig_false = np.var(np.abs(y_regnoise[t1,:]-y_tar))



print(sig_true)
print(sig_false)






import cPickle
#import _pickle as cPickle
# save the classifier
with open('Classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid, protocol = 2)    


import cPickle
#import _pickle as cPickle
# save the classifier
with open('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/skRegTrain/Regressor.pkl', 'wb') as fid:
    cPickle.dump(reg, fid, protocol = 2)    
    
    
    
    
    
    
# load it again
import cPickle
with open('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-EDS_ClfReg_20170407/training_nn/skRegTrain/Regressor.pkl', 'rb') as fid:
    reg = cPickle.load(fid)    




































