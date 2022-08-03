#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 19:56:32 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot
import sklearn.decomposition
import T_PowerGain

def PCA_(data,n_components):
    data-=np.mean(data,axis=0)
    pca=sklearn.decomposition.PCA(n_components=n_components,svd_solver='randomized',whiten=True)
    pca_data=pca.fit_transform(data)
    print(np.sum(pca.explained_variance_ratio_))
    pca_data=pca.inverse_transform(pca_data)
    return pca_data
    
if __name__=='__main__':
    data=np.load('0_iter_record_0_comp.npy')
    data=T_PowerGain.tpowGain(data,np.arange(1500)/4,0)
    data_=PCA_(data.copy(),1)
    
    pyplot.figure()
    pyplot.imshow(data,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))
    
    pyplot.figure()
    pyplot.imshow(data-data_,extent=(0,1,0,1),vmin=0.1*np.min(data),vmax=0.1*np.max(data))