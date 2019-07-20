#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: rotate.py 
@version: 
@time: 2019/07/16 
@email: tanjiale2018@gmail.com
@function： 
"""
import os
import sys

import numpy as np

from app.augment.basic import rotate
from app.augment.basic import BasicTransform


class RotateTransform(BasicTransform):

    def __init__( self, angle_range, axes, order=1, cval=0.0, mode='constant',seed=None, name=None ):
        super().__init__(name)
        assert type(angle_range) == list or type(angle_range) == tuple, "the range of angle should be a list"
        assert len(angle_range) == 2, "the length of angle range should be 2"
        assert len(axes) == 2, "the length of axes should be 2"

        self.angle_min = angle_range[0]
        self.angle_max = angle_range[1]
        self.axes = axes
        self.order = order
        self.cval = cval
        self.mode = mode
        self.random_state = np.random.RandomState(seed)

    def __call__( self , image, label):
        angle = self.random_state.randint(low=self.angle_min, high=self.angle_max)
        image ,label = rotate(image,label,angle,axes=self.axes, order=self.order,cval=self.cval,mode=self.mode)
        return image, label
