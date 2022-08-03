import torch.nn.functional as F
from unet_parts import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init

class cr_net(nn.Module):
    def __init__(self, bilinear=True):
        super(cr_net, self).__init__()
        self.bilinear = bilinear
        self.number = 48

        self.inc = DoubleConv(1, self.number)
        self.down1 = Down(self.number, self.number*2)
        self.down2 = Down(self.number*2, self.number*4)
        factor = 2 if bilinear else 1
        self.dow = Down(self.number*8, self.number*16 // factor)
        self.up1 = Up(self.number*16, self.number*8 // factor, bilinear)
        self.un3 = Down(self.number*4, self.number*8)
        self.down4p2 = Up(self.number*8, self.number*4 // factor, bilinear)
        self.up3 = Up(self.number*4, self.number*2 // factor, bilinear)
        self.up4 = Up(self.number*2, self.number, bilinear)
        self.outc1 = DoubleConv(self.number, self.number//2, self.number)
        self.outc2 = OutConv(self.number//2, 1)

        self.denselayer = 3
        self.dense1 = RDB(self.number,self.denselayer,self.number//self.denselayer)
        self.dense2 = RDB(self.number*2,self.denselayer,self.number*2//self.denselayer)
        self.dense3 = RDB(self.number*4,self.denselayer,self.number*4//self.denselayer)
        self.dense4 = RDB(self.number*8,self.denselayer,self.number*8//self.denselayer)

    def forward(self, x):
        x1 = self.inc(x)
        x1d = self.dense1(x1)

        x2 = self.down1(x1)
        x2d = self.dense2(x2)

        x3 = self.down2(x2)
        x3d = self.dense3(x3)
        
        x4 = self.down3(x3)
        x4d = self.dense4(x4)

        
        x5 = self.down4(x4)


        x6 = self.up1(x5, x4d)

        x7 = self.up2(x6, x3d)

        x8 = self.up3(x7, x2d)

        x9 = self.up4(x8, x1d)

        signal = self.outc1(x9)
        signal = self.outc2(signal)

        return signal

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=(kernel_size-1)//2)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


