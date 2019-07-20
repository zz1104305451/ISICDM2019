#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: elastic.py 
@version: 
@time: 2019/07/05 
@email: tanjiale2018@gmail.com
@function： 
"""

import os
import sys

import random

from app.augment.basic import BasicTransform
from app.augment.basic import elastic_transform

class ElasticTransform(BasicTransform):

    def __init__( self , alpha_range, sigma_range, order=1, seed=None,name = None):
        super().__init__(name)
        assert type(alpha_range) == list and len(alpha_range) == 2
        assert type(sigma_range) == list and len(sigma_range) == 2
        self.alpha_min = alpha_range[0]
        self.alpha_max = alpha_range[1]
        self.sigma_min = sigma_range[0]
        self.sigma_max = sigma_range[1]

        self.order = order
        self.seed = seed

    def __call__( self , image, label):
        alpha = random.randint(self.alpha_min, self.alpha_max)
        sigma = random.randint(self.sigma_min, self.sigma_max)

        image, label = elastic_transform(image, label, alpha, sigma, self.order, self.seed)

        return image, label