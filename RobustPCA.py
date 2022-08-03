#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 17:13:38 2020

@author: nephilim
"""

import numpy as np
from matplotlib import pyplot
import skimage.transform

def shrink(X,tau):##软阈值算法
    return np.sign(X)*np.maximum((np.abs(X)-tau),np.zeros(X.shape))

def svd_threshold(X,tau):
    U,S,V=np.linalg.svd(X,full_matrices=False)
    return np.dot(U,np.dot(np.diag(shrink(S,tau)),V))

def RobustPCA(M,tau,mu,tol=1e-7,max_iter=1000):#tol:误差优化目标
    Lk=Sk=Yk=np.zeros_like(M)
    error=np.inf
    iter_=0
    while error>tol and iter_<max_iter:
        Lk=svd_threshold(M-Sk+1/mu*Yk,1/mu)
        Sk=shrink(M-Lk+1/mu*Yk,tau/mu)
        Yk=Yk+mu*(M-Lk-Sk)
        error=np.linalg.norm(M-Lk-Sk,ord='fro')
        iter_+=1
    return Lk,Sk

def PreProcessGPR(X_data,tol,max_iter):
    if np.min(X_data)<0:
        min_X_data=np.min(X_data)
    else:
        min_X_data=0
    X_data-=min_X_data
    tau=1/np.sqrt(np.max(X_data.shape))/2
    mu=np.prod(X_data.shape)/(4*np.linalg.norm(X_data,ord=1))
    Lk,Sk=RobustPCA(X_data,tau,mu,tol,max_iter)
    X_data+=min_X_data
    Lk+=min_X_data
    Sk+=min_X_data
    return Lk,Sk

if __name__=='__main__':
    import time
    start_time=time.time()
    X_data=np.load('Complex900MHz.npy')
    import cv2
    X_data = cv2.imread('F:/ZPY/SELF/IMG/test1.png',0)
    # X_data=skimage.transform.resize(X_data,(1000,700),mode='edge')
    
    # X_data=np.load('Test_Real_Data.npy')
    # X_data=skimage.transform.resize(X_data,(1024,300),mode='edge')
    
    tol=1e-7
    max_iter=100
    Lk,Sk=PreProcessGPR(X_data,tol,max_iter)
    print(time.time()-start_time)
    
    pyplot.figure()
    pyplot.imshow(X_data-Lk,extent=(0,1,0,1),cmap='gray')
    pyplot.axis('off')
    