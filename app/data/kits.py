import os
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import h5py

import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class KiTS_Dataset(Dataset):

    def __init__( self, file_path, mode, transform=None ):
        self.file_path = file_path
        self.mode = mode
        self.transform = transform

        self.file = None

        with h5py.File(self.file_path, 'r', swmr=True) as f:
            self.dataset_len = len(f["image"])

    def __getitem__( self, index ):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r', swmr=True)
        if self.mode == 'train' or self.mode == 'val':
            image = self.file['image'][index, :]
            label = self.file['mask'][index, :]
            if self.transform is not None:
                image, label = self.transform(image, label)
            return image, label
        elif self.mode == 'test':
            image = self.file['image'][index, :]
            location = self.file['location'][index, :]
            pid = self.file['pid'][index, :]

            return image, location, pid
        else:
            raise NotImplementedError('Specified mode is not support, please check again')

    def __len__( self ):
        return self.dataset_len


if __name__ == '__main__':
    file = '/Users/tanjiale/PycharmProjects/KiTS19-Beta/data/processed/train/train.hdf5'
    # file = h5py.File(file, 'r', swmr=True)
    ds = KiTS_Dataset(file, 'train', transform=None)
    loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=32, pin_memory=True, drop_last=True)
    st = time.time()
    for index, (image, label) in enumerate(loader):
        logging.info(image.shape)
    for index, (image, label) in enumerate(loader):
        logging.info(image.shape)
    for index, (image, label) in enumerate(loader):
        logging.info(image.shape)
    ed = time.time()
    print(st - ed)

    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # image = ds.__getitem__(1)[0][:, :, 16]
    # # image = image.transpose((1,0))
    # image = image[:, :]
    # logging.info(image.shape)
    # plt.imshow(image, cmap='gray')
    # plt.show()
