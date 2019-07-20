#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: evaluator.py 
@version: 
@time: 2019/07/01 
@email: tanjiale2018@gmail.com
@function： 
"""

import SimpleITK as sitk
import numpy as np
import torch

from app.opt.basic import BaseEvaluation
from tools.utils import load_image, truncation
from tools.utils import sample, assemble_channels


class SimpleEvaluation(BaseEvaluation):
    def __init__( self, model, metrics, images_path, labels_path, input_shape, checkpoint_restore, use_gpu=False,
                  gpu_ids=None ):
        super().__init__(model, metrics, images_path, labels_path, input_shape, checkpoint_restore, use_gpu=use_gpu,
                         gpu_ids=gpu_ids)

    def load_data( self, index ):
        """

        :param index:
        :return: (D, H, W) nd-array
        """
        image_path = self.images_path[index]
        label_path = self.labels_path[index]

        image = load_image(image_path, type=sitk.sitkFloat64)
        label = load_image(label_path, type=sitk.sitkUInt8)

        image = truncation(image, lower=-200, upper=300)

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)

        image = np.transpose(image, axes=(2, 1, 0))
        label = np.transpose(label, axes=(2, 1, 0))
        return image, label

    def eval_one_patient( self, image, label ):
        shape = image.shape

        image_patch, _, locations = sample(image, label, self.sample_shape, overlap=(0, 0, 0))

        patch_result = []
        for index in range(len(image_patch)):
            patch = image_patch[index]
            patch = patch[np.newaxis, np.newaxis, :]
            patch = torch.from_numpy(patch)
            patch = patch.float()

            if self.use_gpu:
                patch = patch.cuda()

            with torch.no_grad():
                output = self.model(patch)

            patch_result.append(output.cpu().numpy()[0, :])

        predict = assemble_channels(patch_result, shape, locations)

        metric_result = dict()
        for key in self.metrics.keys():
            metric_result[key] = self.metrics[key](predict, label)

        return metric_result
