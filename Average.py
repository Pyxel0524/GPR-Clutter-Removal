#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 19:35:33 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot
import skimage.transform

# import T_PowerGain

def Average(data):
    m,n=data.shape
    ave=np.sum(data,axis=1)/n
    for idx in range(n):
        data[:,idx]-=ave
    return data
if __name__=='__main__':
    data=np.load('Complex900MHz.npy')
    # data=np.load('Test_Real_Data.npy')
    data=skimage.transform.resize(data,(1024,300),mode='edge')

    # data=T_PowerGain.tpowGain(data,np.arange(1500)/4,1.2)
    data_=Average(data.copy())
    
    pyplot.figure()
    pyplot.imshow(data,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))
    pyplot.axis('off')
    
    pyplot.figure()
    pyplot.imshow(data_,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))
    pyplot.axis('off')