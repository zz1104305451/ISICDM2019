#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: sequential.py 
@version: 
@time: 2019/07/17 
@email: tanjiale2018@gmail.com
@function： 
"""
import os
import sys

from app.augment.basic import BasicSequential

class TransformSequential(BasicSequential):
    def __init__( self) :
        super().__init__()

    def __call__(self, image, label):
        for transform in self.list:
            image, label = transform(image, label)
        return image, label
