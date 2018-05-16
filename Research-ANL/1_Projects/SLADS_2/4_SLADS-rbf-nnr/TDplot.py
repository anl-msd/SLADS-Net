#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:23:13 2017

@author: yzhang
"""

import numpy as np
from matplotlib import pyplot as plt


TD_rbf_lsr = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-yz-nonlinear/ResultsAndData/SLADSSimulationResults/Example_1_rbf_lsr/TDvec.npy')

TD_lin_svr = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-yz-nonlinear/ResultsAndData/SLADSSimulationResults/Example_2_lin_svr/TDvec.npy')

TD_lin_nnr = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-yz-nonlinear/ResultsAndData/SLADSSimulationResults/Example_3_lin_nnr/TDvec.npy')

TD_rbf_nnr = np.load('/home/yzhang/Research-ANL/1_Projects/SLADS-Python-v4-yz-nonlinear/ResultsAndData/SLADSSimulationResults/Example_4/TDvec.npy')




plt.figure()
plt.plot(TD_rbf_lsr, label='TD_rbf_lsr')
plt.plot(TD_lin_svr, label='TD_lin_svr')
plt.plot(TD_lin_nnr, label='TD_lin_nnr')
plt.plot(TD_rbf_nnr, label='TD_rbf_nnr')
plt.legend()










