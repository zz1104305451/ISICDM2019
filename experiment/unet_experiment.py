#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: unet_experiment.py
@version: 
@time: 2019/06/28 
@email: tanjiale2018@gmail.com
@function： 
"""

import os
import sys

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

from app.model.unet import UNet3d
from app.data.kits import KiTS_Dataset
from app.opt.trainer import SimpleTrainer
from app.opt.evaluator import SimpleEvaluation
from app.opt.inferencer import SimpleInferencer
from app.augment.basic import swap_axis

import logging
logging.basicConfig(level=logging.INFO)

# def transform(image, label):
#     image = to_tensor_shape(image)
#     label = to_tensor_shape(label, newaxis=False)
#
#     return image, label

if __name__ == '__main__':
    # # train settings
    # train_file = '/tmp/pycharm_project_745/data/processed/train/train.hdf5'
    # val_file = '/tmp/pycharm_project_745/data/processed/val/val.hdf5'
    #
    # train_ds = KiTS_Dataset(train_file, 'train', transform=transform)
    # val_ds = KiTS_Dataset(val_file,'val',transform=transform)
    #
    # train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=32, pin_memory=False,drop_last=True)
    # val_loader = DataLoader(val_ds, batch_size=16, shuffle=True, num_workers=32, pin_memory=False, drop_last=True)
    #
    # unet = UNet3d(3, activation='Softmax')
    #
    # train_loss_func = nn.CrossEntropyLoss()
    # train_metric_func = {'ce': nn.CrossEntropyLoss()}
    # optimizer = torch.optim.Adam(unet.parameters())
    #
    # trainer = SimpleTrainer(10, unet, train_loader, train_loss_func,train_metric_func,optimizer,log_dir='logs', checkpoint_dir='checkpoint',
    #                         checkpoint_frequency=1,checkpoint_restore=None, val_dataloader=val_loader, val_metrics_func=train_metric_func,lr_scheduler=None,
    #                         lr_reduce_metric=None, use_gpu=True, gpu_ids=[0,1])
    #
    # trainer.train()

    checkpoint_restore = r'/tmp/pycharm_project_745/experiment/checkpoint/checkpoint_10.pth'
    test_image = r'/tmp/pycharm_project_745/data/original/case_00000/imaging.nii.gz'
    test_label = r'/tmp/pycharm_project_745/data/original/case_00000/segmentation.nii.gz'
    test_pid = r'case_00000'

    unet = UNet3d(3, activation='Softmax')
    # def ce(predict, label):
    #     predict = predict[np.newaxis,:]
    #     label = label[np.newaxis,:]
    #
    #     predict = torch.from_numpy(predict)
    #     label = torch.from_numpy(label)
    #
    #     label = label.long()
    #
    #     metric = nn.CrossEntropyLoss()(predict,label)
    #     metric = metric.cpu().item()
    #
    #     return metric
    #
    # metric_func ={'ce': ce}
    #
    # evaluator = SimpleEvaluation(unet, metric_func, images_path=[test_image],labels_path=[test_label], input_shape=(32, 128, 128),
    #                              checkpoint_restore=checkpoint_restore, use_gpu=True, gpu_ids=[0])
    # evaluator.eval()
    #
    inferencer = SimpleInferencer(unet,[test_image],[test_label],[test_pid],sample_shape=(32, 512, 512),checkpoint_restore=checkpoint_restore,
                                  inference_dir='/tmp/pycharm_project_745/experiment/result',
                                  use_gpu=True,gpu_ids=[0])
    inferencer.inference()