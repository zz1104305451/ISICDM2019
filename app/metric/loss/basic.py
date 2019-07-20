#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: basic.py
@version: 
@time: 2019/07/04 
@email: tanjiale2018@gmail.com
@function： 
"""


import os
import sys

import torch

from typing import Tuple, List, Callable


def get_confusion_matrix( predict: torch.Tensor, gt: torch.Tensor, axes=None, square=False ) -> Tuple[torch.Tensor]:
    if axes is None:
        axes = tuple(range(2, len(predict.size())))

    predict_shape = predict.shape
    gt_shape = gt.shape

    with torch.no_grad():
        if len(predict_shape) != len(gt_shape):
            gt = gt.view((gt_shape[0], 1, *gt_shape[1:]))

        if all([i == j for i, j in zip(predict.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(predict_shape)
            if predict.device.type == 'cuda':
                y_onehot = y_onehot.cuda(predict.device.index)
            y_onehot.scatter(1, gt, 1)

    tp = predict * y_onehot
    fp = predict * (1 - y_onehot)
    fn = (1 - predict) * y_onehot
    tn = (1 - predict) * (1 - y_onehot)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    tp = tp.sum(dim=axes, keepdim=False)
    fp = fp.sum(dim=axes, keepdim=False)
    fn = fn.sum(dim=axes, keepdim=False)
    tn = tn.sum(dim=axes, keepdim=False)
    return tp, fp, fn, tn

