#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: noise.py 
@version: 
@time: 2019/07/16 
@email: tanjiale2018@gmail.com
@function： 
"""

import os
import sys

import numpy as np

from app.augment.basic import BasicTransform


class GaussNoise(BasicTransform):
    def __init__( self, sigma, seed=None, name=None ):
        super().__init__(name)
        self.sigma = sigma
        self.random_state = np.random.RandomState(seed=seed)

    def __call__( self, image, label ):
        assert image.shape == label.shape

        shape = image.shape
        noise = self.random_state.normal(loc=0, scale=self.sigma, size=shape)

        image = image + noise
        label = label

        return image, label
