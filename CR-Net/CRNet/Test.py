# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 11:40:03 2022

CR_Net test

@author: Administrator
"""
import torch 
import numpy as np
from torch import nn
from CRNet import *
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms
import d2l
from matplotlib import pyplot as plt
import argparse                     
from PIL import Image

 
# loader使用torchvision中自带的transforms函数
loader = transforms.Compose([
    transforms.ToTensor()])  
 
unloader = transforms.ToPILImage()




def PIL_to_tensor(PIL):
    trans = transforms.ToTensor()
    image = trans(PIL)
    return image


# 输入tensor变量
# 输出PIL格式图片
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


if __name__ == '__main__':
    model = cr_net()    
    checkpoint = torch.load('model/epoch_101_loss_0.00188.pth',map_location='cpu')#epoch_66_loss_0.00258,epoch_101_loss_0.00188
    
    #加载参数            
    model.load_state_dict(checkpoint['model_dict'])
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)#加载保存的优化器参数可用于继续训练
    optimizer.load_state_dict(checkpoint['optimizer_dict'])#加载保存的优化器参数可用于继续训练
    
    
    test_path = 'F:\ZPY\github\\NTU_CRNet_withoutTrain\TestingData'
    transform_fn = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        ])#transforms.Normalize((0.5,), (0.5,))
    # 获取数据集
    dataset = datasets.ImageFolder(root = test_path, transform=transform_fn)
    test = dataset[143][0].reshape(1,1,256,64) 
    
    ##命令窗获取数据集的方式
    # parser = argparse.ArgumentParser()  # 创建parser   
    # parser.add_argument('--test', default = 'F:\ZPY\github\\NTU_CRNet_withoutTrain\TestingData\SimulatedTestingData\\624.png',
    #         help='the address of test imgs')# 添加参数
    # args = parser.parse_args()      
    # test_ = args.test
    # test = PIL_to_tensor(Image.open(test_)).reshape(1,1,256,64)
    
    
    
    
    out = model(test)
    test_img = tensor_to_PIL(test);plt.figure();plt.imshow(test_img, cmap = 'gray')
    out_img = tensor_to_PIL(out);plt.figure();plt.imshow(out_img, cmap = 'gray')
    print(np.array(out_img))