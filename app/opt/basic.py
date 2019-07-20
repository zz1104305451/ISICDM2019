#!usr/bin/env python  
# -*- coding:utf-8 _*-
""" 
@author: jiale tan & fenfen an
@file: basic.py
@version: 
@time: 2019/06/27 
@email: tanjiale2018@gmail.com
@function： 
"""

import logging
import os

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel

logging.basicConfig(level=logging.INFO)


class BaseTrainer(object):

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
        # train settings
        self.epochs = epochs
        self.model = model
        self.train_dataloader = train_dataloader
        self.train_loss_func = train_loss_func
        self.train_metrics_func = train_metrics_func
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_frequency = checkpoint_frequency
        self.writer = SummaryWriter(logdir=log_dir)

        # validation settings
        if val_dataloader is not None:
            self.validation = True
            self.val_dataloader = val_dataloader
            self.val_metrics_func = val_metrics_func
        else:
            self.validation = False

        # lr scheduler settings
        if lr_scheduler is not None:
            self.lr_schedule = True
            self.lr_scheduler = lr_scheduler
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_reduce_metric = lr_reduce_metric
        else:
            self.lr_schedule = False

        # multi-gpu settings
        self.use_gpu = use_gpu
        gpu_visible = list()
        for index in range(len(gpu_ids)):
            gpu_visible.append(str(gpu_ids[index]))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpu_visible)

        if use_gpu and torch.cuda.device_count() > 0:
            self.model.cuda()
            if gpu_ids is not None:
                if len(gpu_ids) > 1:
                    self.multi_gpu = True
                    self.model = DataParallel(model, gpu_ids)
                else:
                    self.multi_gpu = False
            else:
                if torch.cuda.device_count() > 1:
                    self.multi_gpu = True
                    self.model = DataParallel(model)
                else:
                    self.multi_gpu = False
        else:
            self.multi_gpu = False
            self.device = torch.device('cpu')
            self.model = self.model.cpu()

        # checkpoint settings
        if checkpoint_restore is not None:
            self.model.load_state_dict(torch.load(checkpoint_restore))

    def train( self ):

        for epoch in range(1, self.epochs + 1):
            logging.info('*' * 80)
            logging.info('start epoch %d training loop' % epoch)
            # train
            self.model.train()
            loss, metrics = self.train_epochs(epoch)

            self.writer.add_scalar('train_loss', loss, epoch)
            for key in metrics.keys():
                self.writer.add_scalar(key, metrics[key], epoch)
            if self.lr_schedule:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(loss[self.lr_reduce_metric])
                else:
                    self.lr_scheduler.step()
            logging.info('train loss result: %s' % str(loss))
            logging.info('train metrics result: %s' % str(metrics))

            # validation
            if self.validation:
                logging.info('validation start ... ')
                self.model.eval()
                loss, metrics = self.val_epochs(epoch)
                self.writer.add_scalar('val_loss', loss, epoch)
                for key in metrics.keys():
                    self.writer.add_scalar(key, metrics[key], epoch)
                logging.info('validation loss result: %s' % str(loss))
                logging.info('validation metrics result: %s' % str(metrics))

            # model checkpoint
            if epoch % self.checkpoint_frequency == 0:
                logging.info('saving model...')
                checkpoint_name = 'checkpoint_%d.pth' % epoch
                if self.multi_gpu:
                    torch.save(self.model.module.state_dict(),
                               os.path.join(self.checkpoint_dir, checkpoint_name))
                else:
                    torch.save(self.model.state_dict(),
                               os.path.join(self.checkpoint_dir, checkpoint_name))
                logging.info('model have saved for epoch_%d ' % epoch)
            else:
                logging.info('saving model skipped.')

    def train_epochs( self, epoch ) -> (dict, dict):
        """

        :rtype: loss -> dict , metrics -> dict
        """
        pass

    def val_epochs( self, epoch ) -> (dict, dict):
        """

        :rtype: loss -> dict , metrics -> dict
        """
        pass


class BaseEvaluation(object):

    def __init__( self, model,
                  metrics,
                  images_path,
                  labels_path,
                  sample_shape,
                  checkpoint_restore,
                  use_gpu=False,
                  gpu_ids=None ):
        # model settings
        self.model = model

        # metrics settings
        assert type(metrics) == dict
        self.metrics = metrics

        # data settings
        assert len(images_path) == len(labels_path)
        self.images_path = images_path
        self.labels_path = labels_path
        self.length = len(images_path)
        self.sample_shape = sample_shape

        # gpu settings
        self.use_gpu = use_gpu
        if use_gpu and torch.cuda.device_count() > 0:
            self.model.cuda()
            if gpu_ids is not None:
                if len(gpu_ids) > 1:
                    self.multi_gpu = True
                    self.model = DataParallel(model, gpu_ids)
                else:
                    self.multi_gpu = False
            else:
                if torch.cuda.device_count() > 1:
                    self.multi_gpu = True
                    self.model = DataParallel(model)
                else:
                    self.multi_gpu = False
        else:
            self.multi_gpu = False
            self.model = self.model.cpu()

        self.model.load_state_dict(torch.load(checkpoint_restore))

    def load_data( self, index ):
        """

        :rtype: image -> nd-array, label -> nd-array
        """
        pass

    def eval_one_patient( self, image, label ):
        """

        :rtype: metrics -> dict
        """
        pass

    def eval( self ):
        self.model.eval()

        logging.info('*' * 80)
        logging.info('start evaluation loop')
        logging.info('%d patients need to be evaluated ' % self.length)

        result = dict()
        for index in range(self.length):

            logging.info('start evaluation %d-th patient' % (index + 1))
            image, label = self.load_data(index)
            with torch.no_grad():
                metrics = self.eval_one_patient(image, label)

            for key in metrics.keys():
                if key not in result.keys():
                    result[key] = list()
                result[key].append(metrics[key])

            logging.info('evaluation metrics result: %s' % str(metrics))

        mean_result = dict()
        for key in result.keys():
            mean_result[key] = np.mean(result[key])

        logging.info('*' * 80)
        logging.info('evaluation report: ')
        logging.info('evaluation patient: %d' % self.length)
        logging.info('evaluation metrics %s' % str(mean_result))


class BaseInferencer(object):

    def __init__( self, model,
                  images_path,
                  labels_path,
                  patient_ids,
                  sample_shape,
                  checkpoint_restore,
                  inference_dir,
                  use_gpu=False,
                  gpu_ids=None ):
        # model settings
        self.model = model

        # data settings
        assert len(images_path) == len(labels_path)
        self.images_path = images_path
        self.labels_path = labels_path
        self.patient_ids = patient_ids
        self.length = len(images_path)
        self.sample_shape = sample_shape

        self.inference_dir = inference_dir

        # gpu settings
        self.use_gpu = use_gpu
        if use_gpu and torch.cuda.device_count() > 0:
            self.model.cuda()
            if gpu_ids is not None:
                if len(gpu_ids) > 1:
                    self.multi_gpu = True
                    self.model = DataParallel(model, gpu_ids)
                else:
                    self.multi_gpu = False
            else:
                if torch.cuda.device_count() > 1:
                    self.multi_gpu = True
                    self.model = DataParallel(model)
                else:
                    self.multi_gpu = False
        else:
            self.multi_gpu = False
            self.model = self.model.cpu()

        self.model.load_state_dict(torch.load(checkpoint_restore))

    def inference( self ):
        self.model.eval()

        logging.info('*' * 80)
        logging.info('start inference loop')
        logging.info('%d patients need to be inference ' % self.length)

        for index in range(self.length):
            logging.info('start inference %d-th patient' % (index+1))
            self.__inference__(index)

        logging.info('*' * 80)
        logging.info('inference patient: %d' % self.length)
        logging.info('inference result saved in: %s' % self.inference_dir)

    def __inference__( self, index ):
        """

        :rtype: object
        """
        pass
