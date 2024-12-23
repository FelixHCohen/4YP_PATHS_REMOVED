import numpy as np
import cv2
import torch
import random
from torch.utils.data import Dataset
import albumentations as A
import pandas as pd
from albumentations import (
    Compose,
    CLAHE,
    ColorJitter,
    Equalize,
    FancyPCA,
    HueSaturationValue,
    ISONoise,
    ImageCompression,
    MultiplicativeNoise,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise,
    Blur,
    MedianBlur,
    RGBShift,
    ChannelShuffle,
    ElasticTransform
)

""" hard transformation sequence for interactive model"""
def intensity_aug(p=0.8):# old vals scale 0.1 elasrtic 0.2
    return Compose([
        A.ShiftScaleRotate(p=0.1, shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=0,
                           border_mode=cv2.BORDER_CONSTANT, mask_value=0, value=0
                           ),
        A.ElasticTransform(p=0.2, border_mode=cv2.BORDER_CONSTANT, value=0, alpha_affine=30, mask_value=0),
        A.Rotate(limit=270, p=0.4, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0)
,
        CLAHE(p=0.1),
        ColorJitter(p=0.2),
        Equalize(p=0.05),
        FancyPCA(p=0.1),
        HueSaturationValue(p=0.1),
        ISONoise(p=0.1),
        MultiplicativeNoise(p=0.1),
        RandomBrightnessContrast(p=0.3),
        RandomGamma(p=0.2),
        GaussNoise(p=0.1),
        Blur(p=0.1),
        MedianBlur(p=0.01),
        RGBShift(p=0.05),
        #ChannelShuffle(p=0.001),
        A.HorizontalFlip(p=0.25),
        A.VerticalFlip(p=0.05),
        ], p=p)


"""transformation sequence for unet"""
basic_transform = A.Compose([A.VerticalFlip(p=0.1),A.HorizontalFlip(p=0.25),A.Rotate(limit=45, p=0.55,border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0)
           ]) #prev probs were 0.15 0.45, 25 degree limit, for gs1/gamma 0.25,0.6,25 degree limit

"""baseline dataset class (note: all disc only functionality has been removed from this repo)"""
class train_test_split(Dataset):
    def __init__(self, images_path, masks_path, disc_only=False,transform=False,return_path=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        self.return_path = return_path #for when a particular image is causing issues and u want path
        if transform == 'baseline':
            self.transform = basic_transform
        elif transform == 'interactive':
            print('interactive transforms')
            self.transform = intensity_aug()
        else:
            self.transform = False


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask < 128, 2, mask)  # set cup values to 2
        mask = np.where(mask == 128, 1, mask)  # disc pixels sset to 1
        mask = np.where(mask > 128, 0, mask)  # background pixels set to 0
        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """


        if self.disc_only:
            mask = np.where(mask == 2, 1, mask)  # set cup values to 2

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)
        if self.return_path == True:
            res = image, mask,self.images_path[index],self.masks_path[index]
        else:
            res = image,mask

        return res

    def __len__(self):
        return self.num_samples






"""class for all RIGA datasets (messidor, binrushed, magrabia) """
class RIGA_dataset(train_test_split):

    def __init__(self, images_path, masks_path, disc_only=False, transform=False, return_path=False):
        super().__init__(images_path,masks_path,disc_only,transform,return_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask == 128, 2, mask)  # set cup values to 2
        mask = np.where(mask == 255, 1, mask)  # disc pixels sset to 1

        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """


        if self.disc_only:
            mask = np.where(mask == 2, 1, mask)  # set cup values to 2

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)
        if self.return_path == True:
            res = image, mask,self.images_path[index],self.masks_path[index]
        else:
            res = image,mask

        return res
class GS1_dataset(Dataset):
    def __init__(self, images_path, cup_path,disc_path,return_path = False,disc_only=False,transform=False):

        self.images_path = images_path
        self.cup_path = cup_path
        self.disc_path = disc_path
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        self.return_path = return_path
        if transform:
            self.transform = basic_transform
        else:
            self.transform = False


    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        disc_mask = cv2.imread(self.disc_path[index], cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(self.cup_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(disc_mask >= 191, 1, disc_mask)  # set disc values to 1 w 75% agreement
        mask = np.where(disc_mask < 191, 0, mask)  # background to 0

        mask = np.where(cup_mask >= 191, 2, mask)  # set cup values to 2
        if self.disc_only:
            mask = np.where(mask>=2,1,mask)
        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """




        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)

        if self.return_path == True:
            res = image, mask,self.images_path[index],self.cup_path[index]
        else:
            res = image,mask

        return res

    def __len__(self):
        return self.num_samples



"""never ended up using the RIMDL dataset"""
class RIMDL_dataset(GS1_dataset):

    def __init__(self,images_path,disc_path,cup_path,disc_only=False,transform=False):
        super().__init__(images_path,disc_path,cup_path,disc_only,transform)

    def __getitem__(self,index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        disc_mask = cv2.imread(self.disc_path[index], cv2.IMREAD_GRAYSCALE)
        cup_mask = cv2.imread(self.cup_path[index], cv2.IMREAD_GRAYSCALE)
        mask = np.where(disc_mask >= 255, 1, disc_mask)  # set disc vals to 1
        mask = np.where(disc_mask < 255, 0, mask)  # background to 0

        if not self.disc_only:
            mask = np.where(cup_mask >= 255, 2, mask)  # set cup values to 2
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image - 127.5) / 127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)  # (3,512,512)

        """ Reading mask """

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)  # (1,512,512)

        return image, mask

"""didnt have time to use this dataset (each image corresponds to two masks from two different experts, tricky to decide how to combine these)"""
class PAPILA_dataset(Dataset):
    def __init__(self, images_path, masks_path,masks_path_2, disc_only=False,transform=False,return_path=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.masks_path_2 = masks_path_2
        self.num_samples = len(images_path)
        self.disc_only = disc_only
        self.return_path = return_path
        if transform:
            self.transform = intensity_aug()
        else:
            self.transform = False


    def __getitem__(self, index):
        choice = random.randint(0,1)

        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        if choice ==0:
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        else:
            mask = cv2.imread(self.masks_path_2[index], cv2.IMREAD_GRAYSCALE)

        mask = np.where(mask < 128, 2, mask)  # set cup values to 2
        mask = np.where(mask == 128, 1, mask)  # disc pixels sset to 1
        mask = np.where(mask > 128, 0, mask)  # background pixels set to 0
        if self.transform:
            augmented = self.transform(image=image,mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        '''Normalise tensity in range [-1,-1]'''
        image = (image-127.5)/127.5
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)      # (3,512,512)

        """ Reading mask """


        if self.disc_only:
            mask = np.where(mask == 2, 1, mask)  # set cup values to 2

        mask = mask.astype(np.int64)
        mask = np.expand_dims(mask, axis=0)
        mask = torch.from_numpy(mask)        # (1,512,512)
        if self.return_path == True:
            res = image, mask,self.images_path[index],self.masks_path[index]
        else:
            res = image,mask

        return res

    def __len__(self):
        return self.num_samples




