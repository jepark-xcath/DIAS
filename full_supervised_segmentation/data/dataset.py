import sys
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import torch
from torch.utils.data import Dataset
from utils.data_augmentation import Compose, ToTensor, CropToFixed, HorizontalFlip, VerticalFlip, RandomRotate90
import cv2
import torch.nn.functional as F


class Train_dataset(Dataset):
    def __init__(self, config, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.size = config.DATASET.PATCH_SIZE
        self.num_each_epoch = config.DATASET.NUM_EACH_EPOCH
        self.image_list = os.listdir(self.images_path)

        seed = np.random.randint(123)

        self.seq_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

        self.gt_DA = Compose([
            CropToFixed(np.random.RandomState(seed), size=self.size),
            HorizontalFlip(np.random.RandomState(seed)),
            VerticalFlip(np.random.RandomState(seed)),
            RandomRotate90(np.random.RandomState(seed)),
            ToTensor(False)
        ])

    def __getitem__(self, idx):
        id = np.random.randint(len(self.image_list))
        img_id = self.image_list[id]
        img =  np.load(os.path.join(self.images_path,img_id))
        if img.ndim == 2:
            img = img[np.newaxis]
        gt = cv2.imread(os.path.join(
                self.labels_path, f"label_{img_id.split('_')[1].split('.')[0]}.png"), 0)
        gt = np.array(gt/255)[np.newaxis]
        # gt = np.load(os.path.join(self.labels_path,img_id))[np.newaxis]
        # print('dataset @46', img.shape, gt.shape)
        img = self.seq_DA(img)
        gt = self.gt_DA(gt)
        # print('dataset @48', img.shape, gt.shape)
        return img, gt.long()

    def __len__(self):
        return self.num_each_epoch


class Valid_dataset(Train_dataset):
    def __init__(self, config, images_path, labels_path):
        self.images_path = images_path
        self.labels_path = labels_path
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        self.img_list, self.gt_list = self.read_image(
            self.images_path, self.labels_path)
        self.img_patch = self.get_patch(
            self.img_list, self.patch_size, self.stride)
        self.gt_patch = self.get_patch(
            self.gt_list, self.patch_size, self.stride)

    def read_image(self, images_path, label_path):
        image_files = list(sorted(os.listdir(images_path)))
        print('dataset @72 ', len(image_files), label_path, image_files)
        images = []
        gts = []
        for image_id in image_files:               
            img = np.load(os.path.join(images_path,image_id))
            if img.ndim == 2:
                img = img[np.newaxis]
            images.append(img)            

            image_id = image_id.split('s')[-1].split('_')[0].split('.')[0]            
            gt = cv2.imread(os.path.join(
                self.labels_path, f"label_s{image_id}.png"), 0)
            print('dataset @84', image_id, gt.shape)
            gt = np.array(gt/255)[np.newaxis]
            # gt = np.load(os.path.join(self.labels_path,f"{i}.npy"))[np.newaxis]
            gts.append(gt)

        return images, gts

    def get_patch(self, image_list, patch_size, stride):
        patch_list = []
        _, h, w = image_list[0].shape

        pad_h = stride - (h - patch_size[0]) % stride
        pad_w = stride - (w - patch_size[1]) % stride
        for image in image_list:
            image = F.pad(torch.from_numpy(image).float(),
                          (0, pad_w, 0, pad_h), "constant", 0)
            image = image.unfold(1, patch_size[0], stride).unfold(
                2, patch_size[1], stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(
                image.shape[0] * image.shape[1], image.shape[2], patch_size[0], patch_size[1])
            for sub in image:
                patch_list.append(sub)
        return patch_list

    def __getitem__(self, idx):
        img = self.img_patch[idx]
        gt = self.gt_patch[idx]
        # print('dataset @104', img.shape, gt.shape)
        return img, gt.long()

    def __len__(self):
        return len(self.img_patch)

class Test_dataset(Train_dataset):
    def __init__(self, config, images_path):
        self.images_path = images_path
        self.patch_size = config.DATASET.PATCH_SIZE
        self.stride = config.DATASET.STRIDE
        self.image_files = list(sorted(os.listdir(images_path)))[:10]
        self.img_list = self.read_image(self.image_files)
        self.img_patch = self.get_patch(
            self.img_list, self.patch_size, self.stride)

    def read_image(self, image_files):
        images = []
        for image_id in image_files:
            file_path = os.path.join(self.images_path,image_id)
            if not os.path.isfile(file_path):
                continue
            img = np.load(file_path)
            img = (img - np.mean(img)) / np.std(img)
            # img = ((img + 1) / 2)
            img = (img - np.min(img))*255 / (np.max(img) - np.min(img))
            
            if img.ndim == 2:
                img = img[np.newaxis]
            if img.shape[0] > 4: 
                frames = img.shape[0]
                gap = np.around(frames-4)-1
                img = img[gap:gap+4]
            images.append(img)    
        return images

    def get_patch(self, image_list, patch_size, stride):
        patch_list = []

        _, h, w = image_list[0].shape
        
        pad_h = stride - (h - patch_size[0]) % stride
        pad_w = stride - (w - patch_size[1]) % stride
        # print('dataset @150', h, w, patch_size, stride, pad_h, pad_w)
        for i, image in enumerate(image_list):
            image = F.pad(torch.from_numpy(image).float(),
                          (0, pad_w, 0, pad_h), "constant", 0)
            image = image.unfold(1, patch_size[0], stride).unfold(
                2, patch_size[1], stride).permute(1, 2, 0, 3, 4)
            image = image.contiguous().view(
                image.shape[0] * image.shape[1], image.shape[2], patch_size[0], patch_size[1])
            for sub in image:
                patch_list.append(sub)
        return patch_list

    def __getitem__(self, idx):     
        img = self.img_patch[idx]
        # print('dataset @104', img.shape, gt.shape)
        return img

    def __len__(self):
        return len(self.img_patch)
