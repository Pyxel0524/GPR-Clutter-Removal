# -*- coding: utf-8 -*-
"""
Created on Sat Jul  2 06:17:40 2022

Train CR-Net

@author: zhangpengyu970524@163.com
"""

import torch 
import numpy as np
from torch import nn
from CRNet import *
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import d2l
from matplotlib import pyplot as plt
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import os


def train(net, optimizer, train_data, train_labels, 
          num_epochs, lr_period, lr_decay, batch_size,device,save_path):
    train_loss = []
    #loss: MAE + MS_SSIM
    loss1 = torch.nn.L1Loss().to(device)
    loss2 = MS_SSIM(win_size=3,win_sigma=1.5, data_range=1, size_average=True, channel=1).to(device)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_period, lr_decay)
    for epoch in range(num_epochs):
        train_step = 0
        for (raw, classes),(free, classes) in zip(train_data,train_labels):
            optimizer.zero_grad()
            raw, free = raw.to(device), free.to(device)
            l1 = loss1(net(raw),free)
            l2 = 1-loss2(net(raw),free)
            l = l1 + l2
            l1.backward()
            optimizer.step()
    # legend = ['train loss', 'train acc']
    # animator = d2l.Animator(xlabel = 'epoch', xlim = [1, num_epochs],
    #                         legend = legend) #可视动画    
            train_step = train_step + 1
            if train_step % 150 == 0:
              #print('raw data:', net(raw))
              print("epoch: {}, train step: [{}/{}], Loss: {:.5f}".format(epoch+1, train_step*batch_size, len(train_data.dataset), l.data))
              save_model(save_path + '/epoch_{}_loss_{:.5f}.pth'.format(epoch+1+66,l), optimizer, net, epoch)
    return train_loss



def save_model(save_path, optimizer, model, epoch):
    torch.save({'epoch': epoch + 1,
          'optimizer_dict': optimizer.state_dict(),
          'model_dict': model.state_dict()},
          save_path)
    print("model save success")


    
def get_dataloader(file_dir, Batch_Size=40):
    transform_fn = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()])
    # 获取数据集
    dataset = datasets.ImageFolder(root = file_dir, transform=transform_fn)
    # 获取数据加载器
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=40)
    return data_loader    
    

if __name__ == '__main__':
    ## parameter
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(torch.cuda.get_device_properties(device))  # 显卡信息   
    
    learning_rate = 0.000001
    lr_period = 30; lr_decay = 0.1
    epoch = 100
    Batch_Size = 40  
    save_path = '/content/drive/MyDrive/Colab_Notebooks/NTU_CRNet/Model'
  
    # model
    if os.path.exists('/content/drive/MyDrive/Colab_Notebooks/NTU_CRNet/Model/epoch_66_loss_0.00258.pth'):
      checkpoint = torch.load('/content/drive/MyDrive/Colab_Notebooks/NTU_CRNet/Model/epoch_66_loss_0.00258.pth')
      CRNet = cr_net().to(device)
      CRNet.load_state_dict(checkpoint['model_dict'])
    else:
      CRNet = cr_net().to(device)
    # optimizer
    optimizer = torch.optim.Adam(CRNet.parameters(), lr = learning_rate)#定义优化器和训练步长
    if os.path.exists('/content/drive/MyDrive/Colab_Notebooks/NTU_CRNet/Model/epoch_66_loss_0.00258.pth'):
      optimizer.load_state_dict(checkpoint['optimizer_dict'])

    ##加载数据
    train_loader = get_dataloader('/content/CLT-GPR-dataset/Train', Batch_Size = Batch_Size)
    label_loader = get_dataloader('/content/CLT-GPR-dataset/Label', Batch_Size = Batch_Size)
    
    train_loss = train(CRNet, optimizer, train_loader,label_loader, epoch, lr_period, lr_decay, Batch_Size,device,save_path)






   




   
       

