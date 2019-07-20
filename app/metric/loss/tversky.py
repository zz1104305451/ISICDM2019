#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: tversky.py 
@version: 
@time: 2019/07/20 
@email: tanjiale2018@gmail.com
@function： 
"""

import os
import sys

import torch

from app.metric.loss.basic import get_confusion_matrix

class TverskyIndexSimilarity(torch.nn.Module):

    def __init__(self):
        pass
