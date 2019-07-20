#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: flip.py 
@version: 
@time: 2019/07/16 
@email: tanjiale2018@gmail.com
@function： 
"""
import os
import sys

import random
import numpy as np

from app.augment.basic import BasicTransform
from app.augment.basic import affine_transform
from app.augment.basic import transform_matrix_offset_center


class FlipTransform(BasicTransform):

    def __init__( self, axes, weights, order=1, cval=0.0, mode='constant', seed=None, name=None ):
        super().__init__(name)
        assert type(axes) == list or type(axes) == tuple
        assert type(weights) == list or type(weights) == tuple

        for a in axes:
            assert type(a) == bool, "axes should be a boolean list"

        for w in weights:
            assert 0.0 <= w <= 1.0, "weight should be float between 0.0 and 1.0"

        self.axes = axes
        self.weights = weights
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
        offset = np.zeros(4)

        # build affine matrix
        for i, (axis, weight) in enumerate(zip(self.axes, self.weights)):
            if axis:
                if self.random_state.uniform() < weight:
                    affine_matrix[i][i] = -1

        affine_matrix = transform_matrix_offset_center(affine_matrix, x, y, z)

        # process
        image, label = affine_transform(image, label,
                                        affine_matrix,
                                        offset,
                                        order=self.order,
                                        cval=self.cval,
                                        mode=self.mode)

        return image, label
