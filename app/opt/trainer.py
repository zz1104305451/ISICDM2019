#!usr/bin/env python
# -*- coding:utf-8 _*-

""" 
@author: jiale tan
@file: basic.py
@version: 
@time: 2019/06/28 
@email: tanjiale2018@gmail.com
@function： 
"""

import os
import sys

import time
import torch
import tqdm
import numpy as np

from app.opt.basic import BaseTrainer

import logging

logging.basicConfig(level=logging.INFO)


# Single Input , Single Output for test
# Note:
# We define the loss function should be only one function
# If you want to implement a multi-output model, you can
# combine these loss in one function and pass it in
class SimpleTrainer(BaseTrainer):

    def __init__( self, epochs,
                  model,
                  train_dataloader,
                  train_loss_func,
                  train_metrics_func,
                  optimizer,
                  log_dir,
                  checkpoint_dir,
                  checkpoint_frequency,
                  checkpoint_restore=None,
                  val_dataloader=None,
                  val_metrics_func=None,
                  lr_scheduler=None,
                  lr_reduce_metric=None,
                  use_gpu=False,
                  gpu_ids=None ):
        super().__init__(epochs, model,
                         train_dataloader,
                         train_loss_func,
                         train_metrics_func,
                         optimizer,
                         log_dir,
                         checkpoint_dir,
                         checkpoint_frequency,
                         checkpoint_restore,
                         val_dataloader,
                         val_metrics_func,
                         lr_scheduler,
                         lr_reduce_metric,
                         use_gpu,
                         gpu_ids)

    def train_epochs( self, epoch ):
        loss_result = list()
        metric_result = dict()
        for index, (image, label) in enumerate(self.train_dataloader):

            label = label.long()

            if self.use_gpu:
                image = image.cuda()
                label = label.cuda()

            output = self.model(image)
            loss = self.train_loss_func(output, label)
            loss_result.append(loss.cpu().item())

            for key in self.train_metrics_func.keys():
                if not key in metric_result.keys():
                    metric_result[key] = list()
                metric = self.train_metrics_func[key](output, label)
                metric_result[key].append(metric.cpu().item())

            logging_metrics = dict()
            for key in metric_result.keys():
                logging_metrics[key] = metric_result[key][-1]

            logging.info(
                '%d-th sub-epoch , loss: %s , metrics: %s' % (index + 1, str(loss_result[-1]), str(logging_metrics)))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # mean for load to tensorboardX
        mean_loss_result = np.mean(loss_result)
        mean_metric_result = {}
        for key in self.train_metrics_func.keys():
            mean_metric_result[key] = np.mean(metric_result[key])
        return mean_loss_result, mean_metric_result

    def val_epochs( self, epoch ):
        loss_result = list()
        metric_result = dict()
        for index, (image, label) in enumerate(self.val_dataloader):
            if self.use_gpu:
                image = image.cuda()
                label = label.cuda()

            label = label.long()
            with torch.no_grad():
                output = self.model(image)
            loss = self.train_loss_func(output, label)

            loss_result.append(loss.cpu().item())
            for key in self.val_metrics_func.keys():
                if not key in metric_result.keys():
                    metric_result[key] = list()
                metric = self.val_metrics_func[key](output, label)
                metric_result[key].append(metric.cpu().item())

            logging_metrics = dict()
            for key in metric_result.keys():
                logging_metrics[key] = metric_result[key][-1]

            logging.info('%d-th sub-epoch , loss: %s , metrics: %s' % (index + 1, str(loss_result[-1]), str(logging_metrics)))

        mean_loss_result = np.mean(loss_result)
        mean_metric_result = {}
        for key in self.val_metrics_func.keys():
            mean_metric_result[key] = np.mean(metric_result[key])
        return mean_loss_result, mean_metric_result
