#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: shear.py 
@version: 
@time: 2019/07/16 
@email: tanjiale2018@gmail.com
@function： 
"""
import os
import sys

import numpy as np

from app.augment.basic import BasicTransform
from app.augment.basic import affine_transform
from app.augment.basic import transform_matrix_offset_center


class ShearTransform(BasicTransform):

    def __init__( self, shear_min, shear_max, order=1, cval=0.0, mode='constant', seed=None, name=None ):
        super().__init__(name)
        assert isinstance(shear_min, np.ndarray)
        assert isinstance(shear_max, np.ndarray)
        assert shear_min.shape == (3,3)
        assert shear_max.shape == (3,3)

        for i in range(3):
            shear_min[i][i] = shear_max[i][i] = 0
        self.shear_min = shear_min
        self.shear_max = shear_max

        self.order = order
        self.cval = cval
        self.mode = mode
        self.random_state = np.random.RandomState(seed=seed)

    def __call__( self, image, label ):
        assert image.shape == label.shape, "image shape should equal to label shape"

        shape = image.shape
        assert len(shape) == 3, "image shape should be used be 3-dimension"

        # initialize
        x, y, z = shape
        affine_matrix = np.diag(np.ones(4))

        shear = self.random_state.uniform(low=self.shear_min, high=self.shear_max)
        empty = np.zeros_like(affine_matrix)
        empty[:3,:3] = shear
        affine_matrix = affine_matrix + empty

        affine_matrix = transform_matrix_offset_center(affine_matrix, x, y, z)

        image, label = affine_transform(image, label, affine_matrix, order=self.order, cval=self.cval, mode=self.mode)
        return image, label
