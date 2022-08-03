#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:57:29 2020

@author: nephilim
"""

import numpy as np
from numba import jit

def my_im2col(Image,BlkSize,slidingDis):
    if slidingDis==1:
        blocks=im2col(Image,BlkSize)
        idx=np.arange(blocks.shape[1])
        return blocks,idx
    idxMat=np.zeros(np.array(Image.shape)-BlkSize+1)
    idxMat[:-1:slidingDis,:-1:slidingDis]=1
    idxMat[-1,:None:slidingDis]=1
    idxMat[:None:slidingDis,-1]=1
    idxMat[-1,-1]=1
    cols,rows=np.where(idxMat.T==1)
    idx=cols*idxMat.shape[0]+rows
    blocks=np.zeros((np.prod(BlkSize),len(rows)))
    for index in range(len(rows)):
        currBlock=Image[rows[index]:rows[index]+BlkSize[0],cols[index]:cols[index]+BlkSize[1]]
        blocks[:,index]=currBlock.ravel()
    return blocks,idx
    
@jit(nopython=True)
def im2col(Image,BlkSize):
    idx=0
    blocks=np.zeros((int(BlkSize[0]*BlkSize[1]),int((Image.shape[0]-BlkSize[0]+1)*(Image.shape[1]-BlkSize[1]+1))))
    for idx_col in range(Image.shape[1]-BlkSize[1]+1):
        for idx_row in range(Image.shape[0]-BlkSize[0]+1):
            blocks[:,idx]=Image[idx_row:idx_row+BlkSize[0],idx_col:idx_col+BlkSize[1]].T.ravel()
            idx+=1
    return blocks

def ind2sub(size,index):
    rows=np.mod(index,size[0])
    cols=index//size[0]
    return rows,cols