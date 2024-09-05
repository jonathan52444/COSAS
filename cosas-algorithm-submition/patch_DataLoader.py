import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import imageio as iio
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from scipy.ndimage import zoom
from einops import repeat
from scipy import ndimage
import random
from PIL import Image
import cv2
import logging
import torch

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms import InterpolationMode
import pandas as pd
import csv

# Current loader for the endonasal dataset which we need to incorporate into the trainer.py code
# This will be done by having it output the data in the same shape as the synapse dataloader

def normalise_intensity(image, ROI_thres=0.1):
    pixel_thres = np.percentile(image, ROI_thres)
    ROI = np.where(image > pixel_thres, image, 0) # If image value is greater than pixel threshold, return image value, otherwise return 0
    mean = np.mean(ROI)
    std = np.std(ROI)
    ROI_norm = (ROI - mean) / (std + 1e-8) # Normalise ROI
    return ROI_norm

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

def map_labels(label):
    label_map = {0: 0, 85: 1, 128: 1, 170: 2, 255: 2}
    mapped_label = label.copy()
    for k, v in label_map.items():
        mapped_label[label == k] = v
    return mapped_label    

class RandomGenerator(object):
    def __init__(self, output_size, low_res):
        self.output_size = output_size
        self.low_res = low_res

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)  # why not 3?
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label_h, label_w = label.shape
        low_res_label = zoom(label, (self.low_res[0] / label_h, self.low_res[1] / label_w), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image = repeat(image, 'c h w -> (repeat c) h w', repeat=3)
        label = torch.from_numpy(label.astype(np.float32))
        low_res_label = torch.from_numpy(low_res_label.astype(np.float32))
        sample = {'image': image, 'label': label.long(), 'low_res_label': low_res_label.long()}
        return sample

class COSASDataset(Dataset):
    def __init__(self, root, folder_name, transform_img=None, transform_mask=None):
        self.root = root
        self.folder_name = folder_name
        self.transform_img = transform_img
        self.transform_mask = transform_mask
        self.image_mask_pairs = []

        for obj in self.folder_name:
            image_path = os.path.join(self.root, obj, 'image')
            mask_path = os.path.join(self.root, obj, 'mask')
            
            image_files = sorted(glob(os.path.join(image_path, '*.png')))
            #print(image_files)

            for img_file in image_files:
                base_name = os.path.basename(img_file)
                mask_file = os.path.join(mask_path, base_name)
                #print(base_name)
                if os.path.exists(mask_file):
                    self.image_mask_pairs.append((img_file, mask_file))

    def __len__(self):
        #print(len(self.image_mask_pairs))
        #print(self.image_mask_pairs)
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform_img:
            image = self.transform_img(image)
        if self.transform_mask:
            mask = self.transform_mask(mask) 

        mask = torch.from_numpy(np.array(mask)).long()
        
        mask[mask == 255] = 0
        return {'image': image, 'mask': mask}