#!usr/bin/env python
# -*- coding:utf-8 _*-

"""
@author: jiale tan
@file: basic.py
@version:
@time: 2019/07/12
@email: tanjiale2018@gmail.com
@function：
"""

import os
import sys

import scipy
import numpy as np

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

#########################################################################
# Super Class For Transform
#########################################################################
class BasicTransform(object):
    def __init__(self, name=None):
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def __call__( self , image, label):
        pass

class BasicSequential(object):
    def __init__(self):
        self.list = list()
        self.weight = list()

    def add( self, transforms, weights):
        if type(transforms) == list and type(weights) == list:
            assert len(transforms) == len(weights)
            self.list.extend(transforms)
            self.weight.extend(weights)
        else:
            assert isinstance(transforms, BasicTransform)
            assert type(weights) == int
            self.list.append(transforms)
            self.weight.append(transforms)


    def __call__( self, image, label):
        pass

#########################################################################
# Basic Transform Function
#########################################################################

def transform_matrix_offset_center(matrix, x, y, z):
    """Return transform matrix offset center.
    Parameters
    ----------
    matrix : numpy array
        Transform matrix
    x, y : int
        Size of image.
    Examples
    --------
    - See ``rotation``, ``shear``, ``zoom``.
    """
    o_x = float(x) / 2
    o_y = float(y) / 2
    o_z = float(z) / 2
    offset_matrix = np.array([[1, 0, 0,o_x], [0, 1, 0,o_y], [0, 0, 1,o_z],[0,0,0,1]])
    reset_matrix = np.array([[1, 0,0, -o_x], [0, 1,0, -o_y], [0, 0, 1,-o_z],[0,0,0,1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def elastic_transform( image, label, alpha, sigma, order=1, seed=None ):
    """
    Elastic deformation of images as described in [Simard2003]_.
    [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    :param image: numpy array, sampled image patch
    :param label: numpy array, sampled label patch
    :param alpha:
    :param sigma:
    :param order:
    :param seed: integer, random seed
    :return: (numpy array, numpy array), (transformed image, transformed label)
    """
    random_state = np.random.RandomState(seed)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z + dz, (-1, 1))

    distored_image = map_coordinates(image, indices, order=order, mode='reflect')

    if label is not None:
        distored_label = map_coordinates(label, indices, order=order, mode='reflect')
        return distored_image.reshape(image.shape), distored_label.reshape(label.shape)
    else:
        return distored_image.reshape(image.shape), None


def affine_transform( image, label, affine_matrix,offset=None, order=1, cval=0.0, mode='constant' ):
    """
    affine transform helper function
    this function can be used to implement shift, rotate, flip, shear
    :param image: numpy array, sampled image patch
    :param label: numpy array, sampled label patch
    :param affine_matrix: list, transform matrix without offset
    :param offset: list, transform offset
    :param order:
    :param cval: float or integer, default number to fill up the blank
    :param mode: string,'constant' or 'reflect' or 'nearest' or 'mirror' or 'wrap'
    :return: (numpy array, numpy array), (transformed image, transformed label)
    """
    image = scipy.ndimage.affine_transform(image, matrix=affine_matrix, order=order,offset=offset, cval=cval, mode=mode)
    if label is not None:
        label = scipy.ndimage.affine_transform(label, matrix=affine_matrix,order=order, offset=offset, cval=cval, mode=mode)
        return image, label
    else:
        return image, None


def zoom_transform( image, label, zoom, order=3,cval=0.0, mode='constant' ):
    """
    zoom transform helper function
    :param image: numpy array, sampled image patch
    :param label: numpy array, sampled label patch
    :param zoom: float, zoom rate
    :param order: int, 1～5
    :param cval: float or integer, default number to fill up the blank
    :param mode: string,'constant' or 'reflect' or 'nearest' or 'mirror' or 'wrap'
    :return: (numpy array, numpy array), (transformed image, transformed label)
    """
    image = scipy.ndimage.zoom(image, zoom=zoom, order=order, cval=cval, mode=mode)
    if label is not None:
        label = scipy.ndimage.zoom(label, zoom=zoom, cval=cval, mode=mode, order=order)
        return image, label
    else:
        return image, None

def crop( image, label, top_left, shape ):
    """
    crop image function
    :param image: numpy array, sampled image patch with shape (D, H, W)
    :param label: numpy array, sampled label patch with shape (D, H, W)
    :param top_left: list with two integer, top left corner coordinate
    :param shape: list or tuple with two integer, target image shape
    :return: (numpy array, numpy array), (transformed image, transformed label)
    """
    assert len(top_left) == 2
    assert len(shape) == 2
    image = image[:, top_left[0]:top_left[0] + shape[0], top_left[1]:top_left[1] + shape[1]]
    if label is not None:
        label = label[:, top_left[0]:top_left[0] + shape[0],top_left[1]:top_left[1] + shape[1]]
        return image, label
    else:
        return image, None


def rotate(image, label, angle, axes=(0,1), order=1, cval=0.0, mode='constant'):
    """

    :param image:
    :param label:
    :param angle:
    :param axes:
    :param order:
    :param cval:
    :param mode:
    :return:
    """
    image = scipy.ndimage.rotate(image,angle,axes=axes,order=order, cval=cval, mode=mode, reshape=False)
    if label is not None:
        label = scipy.ndimage.rotate(label,angle,axes=axes,order=order,mode=mode,cval=cval,reshape=False)
        return image, label
    else:
        return image, None