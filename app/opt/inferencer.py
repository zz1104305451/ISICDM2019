#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: inferencer.py 
@version: 
@time: 2019/07/02 
@email: tanjiale2018@gmail.com
@function： 
"""
import os
import sys

import torch
import SimpleITK as sitk
import numpy as np

from app.opt.basic import BaseInferencer
from tools.utils import load_image, truncation
from tools.utils import sample, assemble_channels

class SimpleInferencer(BaseInferencer):

    def __init__( self, model,
                  images_path,
                  labels_path,
                  patient_ids,
                  sample_shape,
                  checkpoint_restore,
                  inference_dir,
                  use_gpu=False,
                  gpu_ids=None ):
        super(SimpleInferencer, self).__init__(model,
                                               images_path,
                                               labels_path,
                                               patient_ids,
                                               sample_shape,
                                               checkpoint_restore,
                                               inference_dir,
                                               use_gpu=use_gpu,
                                               gpu_ids=gpu_ids)

    def __inference__( self, index ):
        image_path = self.images_path[index]
        label_path = self.labels_path[index]

        image = load_image(image_path, type=sitk.sitkFloat64)
        label = load_image(label_path, type=sitk.sitkUInt8)

        origin = label.GetOrigin()
        spacing = label.GetSpacing()
        direction = label.GetDirection()

        image = truncation(image, lower=-200, upper=300)

        image = sitk.GetArrayFromImage(image)
        label = sitk.GetArrayFromImage(label)

        # (W, H, D) -> (D, H, W)
        image = np.transpose(image,axes=(2,1,0))
        label = np.transpose(label,axes=(2,1,0))

        shape = image.shape
        image_patch, _, locations = sample(image, label, self.sample_shape, overlap=(0, 0, 0))

        patch_result = []
        for i in range(len(image_patch)):
            patch = image_patch[i]
            patch = patch[np.newaxis, np.newaxis, :]
            patch = torch.from_numpy(patch)
            patch = patch.float()

            if self.use_gpu:
                patch = patch.cuda()

            with torch.no_grad():
                output = self.model(patch)

            patch_result.append(output.cpu().numpy()[0,:])

        predict = assemble_channels(patch_result, shape, locations)
        predict = np.argmax(predict, axis=0)

        # (D, H, W) -> (W, H, D)
        predict = np.transpose(predict, axes=(2,1,0))
        predict = sitk.GetImageFromArray(predict)
        predict.SetOrigin(origin)
        predict.SetDirection(direction)
        predict.SetSpacing(spacing)

        inference_file_path = os.path.join(self.inference_dir, str(self.patient_ids[index]) + ".nii.gz")
        sitk.WriteImage(predict, inference_file_path)

