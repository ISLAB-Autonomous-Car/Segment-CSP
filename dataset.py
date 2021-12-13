#!/usr/bin/python -tt
from __future__ import print_function, division

import os
import re
import cv2
import random
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
import torchvision.transforms.functional as ff
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNEL = 3
NUM_CLASSES = 4
BATCH_SIZE = 32
DROPOUT = 0.3

IMG_PATH = "./ThanhDanhVid/image/"
MASK_PATH = "./ThanhDanhVid/labeltrain/"


seed = 42
random.seed = seed
np.random.seed = seed


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans


class LabelProcessor:
    def __init__(self):

        self.colormap = [
            (0,0,0),
            (106, 61, 154),
            (227, 26, 28),
            (31, 120, 180)
            ]

        self.color2label = self.encode_label_pix(self.colormap)

    @staticmethod
    def encode_label_pix(colormap):
        cm2lb = np.zeros(256**3)
        for i, cm in enumerate(colormap):
            cm2lb[(cm[0]*256 + cm[1]) * 256 + cm[2]] = i

        return cm2lb

    def encode_label_img(self, img):
        data = np.array(img, dtype='int32')
        idx = (data[:,:,0] * 256 + data[:,:,1]) * 256 + data[:,:,2]
        label = np.array(self.color2label[idx], dtype='int64')

        return label


p = LabelProcessor()





class Camvid(Dataset):
    """Camvid dataset."""
    def __init__(self, image_path = IMG_PATH, mask_path = MASK_PATH):
        self.crop_size=(296, 280)
        self.image_path = image_path
        self.image_file = os.listdir(self.image_path)
        self.mask_path = mask_path
        self.mask_file = os.listdir(self.mask_path)


    def __len__(self):
        return len(self.image_file)

    def __getitem__(self, idx):
        img_file = self.image_file[idx]
        label_file = self.mask_file[idx]
        label_path = os.path.join(self.mask_path, label_file).replace("\\","/")
        img_path = os.path.join(self.image_path, img_file).replace("\\","/")

        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")

        img, label = self.center_crop(img, label, self.crop_size)

        img, label, y_cls = self.img_transform(img, label, idx)

        return img, label, y_cls

    def center_crop(self, img, label, crop_size):
        img = ff.center_crop(img, crop_size)
        label = ff.center_crop(label, crop_size)

        return img, label

    def img_transform(self, img, label, index):
        label = np.array(label)
        label = Image.fromarray(label.astype('uint8'))

        transform_label = transforms.Compose([
            transforms.ToTensor()]
            )

        transform_img = transforms.Compose(
            [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]
            )

        img = transform_img(img)

        label = p.encode_label_img(label)
        #Image.fromarray(label).save("./results_pic/"+str(index)+".png")
        #print(label.shape)
        y_cls, _ = np.histogram(label, bins=4, range=(-0.5, 9-0.5), )
        # print(y_cls)
        y_cls = np.asarray(np.asarray(y_cls, dtype=np.bool), dtype=np.uint8)

        #label = transform_label(label)
        #label = torch.squeeze(label)

        return img, torch.from_numpy(label), torch.from_numpy(y_cls)

 
