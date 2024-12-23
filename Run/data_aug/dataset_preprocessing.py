from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import re

def resize_data(images,masks,save_path,name_add = '',pad=True):
    size = (512,512)
    size_reg = set()
    for idx, (x,y) in tqdm(enumerate(zip(images,masks))):
        name = x.split("/")[-1].split(".")[0]
        name = f'{name_add}{name}'
        img = cv2.imread(x,cv2.IMREAD_COLOR)
        mask = cv2.imread(y)

        delta = img.shape[1]-img.shape[0]
        size_reg.add(img.shape)
        d1 = delta//2
        d2 = delta - delta//2

        """all fundus images had wider aspect ratios therefore either crops horizontally or pads vertically"""
        if not pad:
            img = img[:,d1:-1*d2,:] # cut off sides to make aspect ratio more similar to 1:1 to avoid distortion
            mask = mask[:,d1:-1*d2,:]
        else:
            img = np.pad(img, ((d1, d2), (0, 0), (0, 0)), mode='constant', constant_values=0)
            mask = np.pad(mask, ((d1, d2), (0, 0), (0, 0)), mode='constant', constant_values=0)

        tmp_image_name = f"{name}.png"
        tmp_mask_name = f"{name}_mask.png"
        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)
        img = cv2.resize(img,size)
        mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_path,mask)
        cv2.imwrite(image_path,img)

    print(f'original image sizes: {size_reg}')


    """ insert image path and mask path like sorted(glob(f'{path}*.jpg)) 
        ensure images are named so sorting both image paths and mask paths means img_paths[i] and mask_paths[i] corresponds to an image and its corresponding mask 
    """
image_paths = sorted(glob.glob())
# dataset_name = ""
# create_dir(f'/home/kebl6872/Desktop/new_data/{dataset_name}}/image/')
# create_dir(f'/home/kebl6872/Desktop/new_data/{dataset_name}/mask/')
# save_path = f'/home/kebl6872/Desktop/new_data/{dataset_name}/train'
#
# resize_data(train_x,train_y,save_path)