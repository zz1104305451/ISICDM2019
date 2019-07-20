#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author: fenfen an
@file: unet.py
@version:
@time: 2019/06/25
@email:
@function：
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torchsummary import summary


class conv_bn_relu(nn.Module):
    def __init__( self, in_channels, out_channels, kernel_size, stride, padding=0, groups=1 ):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.activation = nn.PReLU()

    def forward( self, x ):
        x = self.conv(x)
        x = self.gn(x)
        x = self.activation(x)
        return x


class ResBlock(torch.nn.Module):
    def __init__( self, channels ):
        super(ResBlock, self).__init__()
        self.block_1 = conv_bn_relu(channels, channels, 3, 1, 1, 2)
        self.block_2 = conv_bn_relu(channels, channels, 3, 1, 1, 2)

    def forward( self, x ):
        residual = x
        out = self.block_1(x)
        out = self.block_2(out)
        out = residual + out
        return out


class down_layer(nn.Module):
    def __init__( self ):
        super(down_layer, self).__init__()
        self.conv = nn.Conv3d(1, 16, 3, 1, 1, 1)
        self.res1 = ResBlock(16)
        self.conv1 = nn.Conv3d(16, 32, 3, 1, 1)
        self.res2 = ResBlock(32)
        self.conv2 = nn.Conv3d(32, 64, 3, 1, 1)
        self.res3 = ResBlock(64)
        self.conv3 = nn.Conv3d(64, 128, 3, 1, 1)
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward( self, x ):
        # down layer 1
        x = self.conv(x)
        x = self.res1(x)
        x = self.conv1(x)
        layer1 = x
        x = self.max_pool(x)
        # down layer 2
        x = self.res2(x)
        x = self.conv2(x)
        layer2 = x
        x = self.max_pool(x)
        # down layer3
        x = self.res3(x)
        x = self.conv3(x)
        layer3 = x
        x = self.max_pool(x)
        return x, layer1, layer2, layer3


class deconv(nn.Module):
    def __init__( self, in_channels, out_channels, kernel_size, stride, padding, groups ):
        super(deconv, self).__init__()
        self.deconv_1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding)
        self.gn = nn.GroupNorm(groups, out_channels)
        self.activation = nn.PReLU()

    def forward( self, x ):
        x = self.deconv_1(x)
        x = self.gn(x)
        x = self.activation(x)
        return x


class up_layer(nn.Module):
    def __init__( self ):
        super(up_layer, self).__init__()
        self.up_block3 = deconv(256, 128, 2, 2, 0, 2)
        self.res3 = ResBlock(128)
        self.up_block2 = deconv(128, 64, 2, 2, 0, 2)
        self.res2 = ResBlock(64)
        self.up_block1 = deconv(64, 32, 2, 2, 0, 2)
        self.res1 = ResBlock(32)

    def forward( self, x, layer1, layer2, layer3 ):
        # up layer 3
        x = self.up_block3(x)
        x = x + layer3
        x = self.res3(x)
        # up layer 2
        x = self.up_block2(x)
        x = x + layer2
        x = self.res2(x)
        # up layer 1
        x = self.up_block1(x)
        x = x + layer1
        # x = self.res1(x)
        return x


class UNet3d(nn.Module):
    def __init__( self, num_class, activation ):
        super(UNet3d, self).__init__()
        self.down = down_layer()
        self.bottom = nn.Conv3d(128, 256, 3, 1, 1)
        self.up = up_layer()
        self.conv_1 = nn.Conv3d(32, num_class, 3, 1, 1)
        self.conv_2 = nn.Conv3d(num_class, num_class, 3, 1, 1)
        if activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'Softmax':
            self.activation = nn.Softmax(dim=1)
        else:
            raise NotImplementedError('Not Implement Activation Function')

    def forward( self, x ):
        # bottom layer
        x, layer1, layer2, layer3 = self.down(x)
        x = self.bottom(x)
        x = self.up(x, layer1, layer2, layer3)
        # head
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.activation(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3d(3, 'Softmax').cuda(device)
    # model = UNet3d(2, 'Softmax')

    # summary(model, (1, 32, 128, 128))
    # model = Unet3d(2,'Sigmoid')
    # print(model)

    # input_size_1 = torch.randn((1,1,32,128,128)).cuda()
    # result = model(input_size_1)
    # print(result.shape)
    model.eval()
    for i in range(10):
        input_size_2 = torch.randn((1,1,32,256,256)).cuda()
        result = model(input_size_2)
    print(result)
    print(result.shape)