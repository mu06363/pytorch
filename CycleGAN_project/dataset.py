import os
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from util import *

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, data_type='both'):
        self.data_dir_a = data_dir + 'A'
        self.data_dir_b = data_dir + 'B'
        self.transform = transform
        self.data_type = data_type

        self.to_tensor = ToTensor()

        if os.path.exists(self.data_dir_a):
            lst_data_a = os.listdir(self.data_dir_a)
            lst_data_a = [f for f in lst_data_a if f.endswith('jpg') | f.endswith('png')]
            lst_data_a.sort()
        else:
            lst_data_a = []

        if os.path.exists(self.data_dir_b):
            lst_data_b = os.listdir(self.data_dir_b)
            lst_data_b = [f for f in lst_data_b if f.endswith('jpg') | f.endswith('png')]        
            lst_data_b.sort()
        else:
            lst_data_b = []

        self.lst_data_a = lst_data_a
        self.lst_data_b = lst_data_b

    def __len__(self):
        if self.data_type == 'both':
            if len(self.data_dir_a) < len(self.data_dir_b):
                return len(self.data_dir_a)
            else:
                return len(self.data_dir_b)
        elif self.data_type == 'a':
            return len(self.data_dir_a)
        elif self.data_type == 'b':
            return len(self.data_dir_b)
            
    def __getitem__(self, index):
        
        data = {}

        if self.data_type == 'a' or self.data_type == 'both':
            data_a = plt.imread(os.path.join(self.data_dir_a, self.lst_data_a[index]))

            if data_a.dim == 2:
                data_a = data_a[:, :, np.newaxis]
            if data_a.type == np.uint8:
                data_a = data_a / 255.0
            
            data['data_a'] = data_a

        if self.data_type == 'b' or self.data_type == 'both':
            data_b = plt.imread(os.path.join(self.data_dir_b, self.lst_data_b[index]))

            if data_b.dim == 2:
                data_b = data_b[:, :, np.newaxis]
            if data_b.type == np.uint8:
                data_b = data_b / 255.0

            data['data_b'] = data_b

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data


## 트랜스폼 구현하기

class ToTensor(object):
    def __call__(self, data):
        # label, input = data['label'], data['input']

        # label = label.transpose((2, 0, 1)).astype(np.float32)
        # input = input.transpose((2, 0, 1)).astype(np.float32)

        # data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        # label, input = data['label'], data['input']

        # label = (label - self.mean) / self.std
        # input = (input - self.mean) / self.std

        # data = {'label': label, 'input': input}

        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data
    
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            # label = np.fliplr(label)
            # input = np.fliplr(input)

            for key, value in data.items():
                data[key] = np.flip(value, axis=0)
        
        if np.random.rand() > 0.5:
            # label = np.flipud(label)
            # input = np.flipud(input)

            for key, value in data.items():
                data[key] = np.flip(value, axis=1)
        
        # data = {'label': label, 'input': input}

        return data
    
class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        # input, label = data['input'], data['label']
        # h, w = input.shape[:2]

        key = list(data.keys())[0]
        h, w = data[key].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        # input = input[id_y, id_x]
        # label = label[id_y, id_x]

        # data = {'input': input, 'label': label}

        return data
    
class Resize(object):
    def __init__(self, shape):
        self.shape = shape
    
    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))
                               
        return data