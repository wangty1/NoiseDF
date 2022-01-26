""" Util file. """

import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class SiameseDataset(Dataset):
    """ A dataset containing (face, background, label) tuples. """

    def __init__(self, imageFolderDataset_face, imageFolderDataset_background, transform=None, should_invert=True):
        self.imageFolderDataset_face = imageFolderDataset_face
        self.imageFolderDataset_background = imageFolderDataset_background
        self.transform = transform
        self.should_invert = should_invert
        self.img_size = 64

    def __getitem__(self, index):
        img_face_tuple = self.imageFolderDataset_face.imgs[index]
        img_background_tuple = self.imageFolderDataset_background.imgs[index]
        
        img_face = img_face_tuple[0]
        img_background = img_background_tuple[0]

        img0 = cv2.resize(cv2.imread(img_face), (self.img_size, self.img_size))
        img1 = cv2.resize(cv2.imread(img_background), (self.img_size, self.img_size))

        img0 = torch.tensor(img0).permute(2, 0, 1).float()
        img1 = torch.tensor(img1).permute(2, 0, 1).float()

        label = img_face_tuple[1]

        return img0, img1, label

    def __len__(self):
        return len(self.imageFolderDataset_face.imgs)
