from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

#/data_hd1/data/eyes/drishti-GS/Drishti-G
start_path = '/data_hd1/data/eyes/drishti-GS/Drishti-GS1_files/Drishti-GS1_files'
path = os.path.join(start_path,'Training')
test_path =  os.path.join(start_path,'Test')
test_im_path = os.path.join(test_path,'Images')
test_mask_path = os.path.join(test_path,'Test_GT')
im_path = os.path.join(path,'Images')
mask_path = os.path.join(path,'GT')
train_x = sorted(glob(f'{im_path}/*'))
test_x = sorted(glob(f'{test_im_path}/*'))
test_y = sorted(glob(f'{test_mask_path}/*/SoftMap/*'))
train_y = sorted(glob(f'{mask_path}/*/SoftMap/*'))

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def resize_data(images, masks, save_path):
    size = (512, 512)
    cup_masks = masks[1::2]
    disc_masks = masks[::2]
    paired_masks = [[a,b] for a,b in zip(disc_masks,cup_masks)]

    for idx, (x, y) in tqdm(enumerate(zip(images, paired_masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0] # extracts name from path/name.jpg
        y_disc,y_cup = y

        """ Reading image and mask """
        img = cv2.imread(x, cv2.IMREAD_COLOR)
        y_disc = cv2.imread(y_disc)
        y_cup = cv2.imread(y_cup)


        tmp_image_name = f"{name}.png"
        tmp_cup_name = f"{name}_cup_mask.png"
        tmp_disc_name =  f"{name}_disc_mask.png"

        image_path = os.path.join(save_path, "image", tmp_image_name)
        cup_path = os.path.join(save_path, "cup_mask", tmp_cup_name)
        disc_path = os.path.join(save_path, "disc_mask", tmp_disc_name)


        delta = img.shape[1]-img.shape[0]
        d1 = delta//2
        d2 = delta - d1

        padded_img = np.pad(img, ((d1, d2), (0, 0), (0, 0)), mode='constant')
        padded_y_disc = np.pad(y_disc, ((d1, d2), (0, 0), (0, 0)), mode='constant')
        padded_y_cup =np.pad(y_cup,((d1,d2),(0,0),(0,0)),mode='constant')


        img = cv2.resize(padded_img, size)
        disc_mask = cv2.resize(padded_y_disc, size, interpolation=cv2.INTER_NEAREST)
        cup_mask = cv2.resize(padded_y_cup,size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(image_path, img)
        cv2.imwrite(cup_path, cup_mask)
        cv2.imwrite(disc_path,disc_mask)

create_dir('/home/kebl6872/Desktop/new_data/GS1/train/image/')
create_dir('/home/kebl6872/Desktop/new_data/GS1/train/cup_mask/')
create_dir('/home/kebl6872/Desktop/new_data/GS1/train/disc_mask/')
create_dir('/home/kebl6872/Desktop/new_data/GS1/test/image/')
create_dir('/home/kebl6872/Desktop/new_data/GS1/test/cup_mask/')
create_dir('/home/kebl6872/Desktop/new_data/GS1/test/disc_mask/')
save_path = '/home/kebl6872/Desktop/new_data/GS1/train'
test_save_path = '/home/kebl6872/Desktop/new_data/GS1/test'
resize_data(train_x,train_y,save_path)
resize_data(test_x,test_y,test_save_path)


