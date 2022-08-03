#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 18:40:09 2021

@author: wts
"""
from matplotlib import pyplot,cm
import tensorflow as tf
import numpy as np
import skimage.transform
import my_Im2col

class AutoEncoder():
    def __init__(self,ImageShape,filters,kernel_size,latent_dim):
        self.ImageShape=ImageShape
        self.filters=filters
        self.kernel_size=kernel_size
        self.latent_dim=latent_dim
    
    def Encoder(self):
        self.Encoder_Input=tf.keras.Input(shape=self.ImageShape,name='Encoder_Input_2D')
        x=self.Encoder_Input
        for idx,_ in enumerate(self.filters):
            x=tf.keras.layers.Conv2D(filters=self.filters[idx],kernel_size=self.kernel_size[idx],activation='relu',padding='same')(x)
            x=tf.keras.layers.BatchNormalization()(x)
            x=tf.keras.layers.MaxPool2D((2,2))(x)
            x=tf.keras.layers.Dropout(0.2)(x)
            if idx==0:
                residual=tf.keras.layers.Conv2D(filters=self.filters[idx],kernel_size=5,padding='same')(self.Encoder_Input)
                residual=tf.keras.layers.BatchNormalization()(residual)
                residual=tf.keras.layers.MaxPool2D((2,2))(residual)
                x=tf.keras.layers.add([x,residual])
            if idx==1:
                residual=tf.keras.layers.Conv2D(filters=self.filters[idx],kernel_size=5,padding='same')(self.Encoder_Input)
                residual=tf.keras.layers.BatchNormalization()(residual)
                residual=tf.keras.layers.MaxPool2D((4,4))(residual)
                x=tf.keras.layers.add([x,residual])
        residual=tf.keras.layers.Conv2D(filters=self.filters[-1],kernel_size=3,strides=2**len(self.filters),padding='same')(self.Encoder_Input)
        x=tf.keras.layers.add([x,residual])
        self.shape=tf.keras.backend.int_shape(x)
        # print(self.shape)
        x=tf.keras.layers.Flatten()(x)
        Encoder_Output=tf.keras.layers.Dense(self.latent_dim,name='Encoder_Ouput_1D')(x)
        self.EncoderMode=tf.keras.models.Model(inputs=self.Encoder_Input,outputs=Encoder_Output,name='EncoderPart')
        self.EncoderMode.summary()        
        self.EncoderMode.compile(loss='mse',optimizer='adam')

    def Decoder(self):
        Decoder_Input=tf.keras.Input(shape=(self.latent_dim,),name='Decoder_Input_1D')
        x=tf.keras.layers.Dense(self.shape[1]*self.shape[2]*self.shape[3])(Decoder_Input)
        x=tf.keras.layers.Reshape((self.shape[1],self.shape[2],self.shape[3]))(x)
        for idx,_ in enumerate(self.filters):
            x=tf.keras.layers.Conv2DTranspose(filters=self.filters[len(self.filters)-idx-1],kernel_size=self.kernel_size[len(self.kernel_size)-idx-1],activation='relu',padding='same')(x)
            x=tf.keras.layers.BatchNormalization()(x)
            x=tf.keras.layers.UpSampling2D((2,2))(x)
    
        Decoder_Output=tf.keras.layers.Conv2DTranspose(filters=1,kernel_size=3,activation='sigmoid',padding='same',name='Decoder_Output_1D')(x)
        self.DecoderMode=tf.keras.models.Model(inputs=Decoder_Input,outputs=Decoder_Output)
        self.DecoderMode.summary()  
        self.DecoderMode.compile(loss='mse',optimizer='adam')
        

def BuildAutoEncoder(ImageShape=(32,32,1),filters=[32,64,128],kernel_size=[5,5,5],latent_dim=256):
    AutoEncoder_=AutoEncoder(ImageShape,filters,kernel_size,latent_dim)
    AutoEncoder_.Encoder()
    AutoEncoder_.Decoder()
    AutoEncoderMode=tf.keras.models.Model(inputs=AutoEncoder_.Encoder_Input,outputs=AutoEncoder_.DecoderMode(AutoEncoder_.EncoderMode(AutoEncoder_.Encoder_Input)),name='AutoEncoderMode')
    AutoEncoderMode.summary()  
    AutoEncoderMode.compile(loss='mse',optimizer='adam')
    return AutoEncoderMode

def Patch2TrainData(Patch,PatchSize):
    TrainData=np.zeros(((Patch.shape[1],)+PatchSize+(1,)))
    for idx in range(Patch.shape[1]):
        TrainData[idx,:,:,0]=Patch[:,idx].reshape(PatchSize)
    return TrainData

def GetPatch(Image,patch_size,slidingDis):
    blocks,idx=my_Im2col.my_im2col(Image,patch_size,slidingDis)
    return blocks,idx

def NormalizeData(data):
    data_min=np.min(data)
    data_max=np.max(data)
    data_tmp=np.zeros_like(data)
    data_tmp=(data-data_min)/(data_max-data_min)
    return data_tmp,data_max,data_min

def ReverseNormalizeData(data,data_min,data_max):
    data_tmp=np.zeros_like(data)
    data_tmp=data*(data_max-data_min)+data_min
    return data_tmp

def shrink(M,tau):
    return np.sign(M)*np.maximum((np.abs(M)-tau),np.zeros(M.shape))

def DAE_fit(X_data,epochs):
    data,data_max,data_min=NormalizeData(X_data) ## data normalize
    patch_size=(32,32)
    slidingDis=8
    data_input_Patch,Patch_Idx=GetPatch(data,patch_size,slidingDis)# 图像数据分块
    data_input=Patch2TrainData(data_input_Patch,patch_size) #分块
    print(data_max)
    print(data_min)
    print(np.max(data_input))
    print(np.min(data_input))
    
    ImageShape=(32,32,1)
    filters=[32,64,128]
    kernel_size=[3,3,3]
    latent_dim=256
    # DAE_mode=BuildAutoEncoder(ImageShape,filters,kernel_size,latent_dim)
    # DAE_mode.fit(x=data_input,y=data_input,epochs=epochs,batch_size=32)
    # DAE_mode.save('AutoEncoderClutterRemoval.h5')
    
    DAE_mode = tf.keras.models.load_model('AutoEncoderClutterRemoval.h5')
    data_output=DAE_mode.predict(data_input) #预测
    print(np.max(data_output))
    print(np.min(data_output))
    data_output=ReConstruct(X_data.shape,patch_size,Patch_Idx,data_output,data_max,data_min) #数据重建
    print(np.max(data_output))
    print(np.min(data_output))
    return data_output

def ReConstruct(ImageShape,patch_size,Patch_Idx,x_decoded,data_max,data_min):
    rows,cols=my_Im2col.ind2sub(np.array(ImageShape)-patch_size[0]+1,Patch_Idx)
    IMout=np.zeros(ImageShape)
    Weight=np.zeros(ImageShape)
    count=0
    for index in range(len(cols)):
        col=cols[index]
        row=rows[index]
        block=x_decoded[count,:,:,0]
        IMout[row:row+patch_size[0],col:col+patch_size[1]]+=block
        Weight[row:row+patch_size[0],col:col+patch_size[1]]+=np.ones(patch_size)
        count+=1
    PredictData=IMout/Weight
    PredictData=ReverseNormalizeData(PredictData,data_min,data_max)
    return(PredictData)

def RobustDAE(M,rho,lambda_,tol=1e-7,max_iter=5):
    Lk=Sk=Yk=np.zeros_like(M)
    error=np.inf
    iter_=0    

    while error>tol and iter_<max_iter:
        Lk=DAE_fit(M-Sk+1/rho*Yk,300)
        Sk=shrink(M-Lk+1/rho*Yk,lambda_)
        # Lk=DAE_fit(M-Sk+Yk,50)
        # Sk=shrink(M-Lk+Yk,lambda_)
        Yk=Yk+rho*(M-Lk-Sk)
        error=np.linalg.norm(M-Lk-Sk,ord='fro')
        iter_+=1
    return Lk,Sk

def PreProcessGPR(X_data,tol,max_iter):
    if np.min(X_data)<0:
        min_X_data=np.min(X_data)
    else:
        min_X_data=0
    X_data-=min_X_data
    tau = 1/np.sqrt(np.max(X_data.shape))
    rho = np.prod(X_data.shape)/(4*np.linalg.norm(X_data,ord=1))
    # rho=1
    lambda_ = tau/rho
    Lk,Sk=RobustDAE(X_data,rho,lambda_,tol,max_iter)    
    
    
    X_data+=min_X_data
    Lk+=min_X_data
    Sk+=min_X_data
    return Lk,Sk

if __name__=='__main__':
    import time
    # tf.reset_default_graph()
    start_time=time.time()
    # X_data=np.load('Test_Real_Data.npy')
    # X_data=np.load('Complex900MHz.npy')
    # X_data=X_data/np.max(np.abs(X_data))
    # X_data=skimage.transform.resize(X_data,(1024,300),mode='edge')
    import cv2
    X_data = cv2.imread('F:/ZPY/SELF/IMG/test1.png',0)

    tol=1e-7
    max_iter=100
    Lk,Sk=PreProcessGPR(X_data,tol,max_iter)
    print(time.time()-start_time)
    
    # pyplot.figure()
    # pyplot.imshow(Lk,extent=(0,1,0,1),vmin=np.min(Lk),vmax=np.max(Lk))
    # pyplot.figure()
    # pyplot.imshow(Sk,extent=(0,1,0,1),vmin=np.min(Sk),vmax=np.max(Sk))    
    # pyplot.figure()
    # pyplot.imshow(X_data-Lk,extent=(0,1,0,1),vmin=0.05*np.min(X_data),vmax=0.05*np.max(X_data))
    # pyplot.axis('off')
   
    
    xx=X_data-Lk
    pyplot.figure()
    pyplot.imshow(xx,extent=(0,1,0,1),vmin=0.1*np.min(X_data),vmax=0.1*np.max(X_data))
    pyplot.axis('off')
    
    
    # data,data_max,data_min=NormalizeData(X_data)
    # data_input_Patch,Patch_Idx=GetPatch(data,(32,32),8)# 图像数据分块
    # data_input=Patch2TrainData(data_input_Patch,(32,32)) #分块
    # model=tf.keras.models.load_model('AutoEncoderClutterRemoval.h5')
    # pred = model.predict(data_input)
    # data_output = ReConstruct(X_data.shape,(32,32),Patch_Idx,pred,data_max,data_min) #数据重建

    # pyplot.imshow(data_output,extent=(0,1,0,1),vmin=0.1*np.min(X_data),vmax=0.1*np.max(X_data))
    # pyplot.axis('off')