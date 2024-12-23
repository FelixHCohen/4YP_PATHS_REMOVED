import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

path = '/data_hd1/data/eyes/refuge-challenge/REFUGE2-Training/REFUGE2-Training'
refuge_tr = os.path.join(path,'REFUGE1-Train-400')
refuge_val = os.path.join(path,'REFUGE1-Val-400')
refuge_test = os.path.join(path,'REFUGE1-Test-400')
tr_image_path = os.path.join(refuge_tr,'Images')
tr_mask_path = os.path.join(refuge_tr,'Annotation-Training400/Disc_Cup_Masks')
val_image_path = os.path.join(refuge_val,'REFUGE-Validation400')
val_mask_path = os.path.join(refuge_val,'REFUGE-Validation400-GT/Disc_Cup_Masks')
test_image_path = os.path.join(refuge_test,'Images')
test_mask_path = os.path.join(refuge_test,'Annotation/Disc_Cup_Masks')
train_x = sorted(glob(os.path.join(tr_image_path,'**/*.jpg'),recursive=True))
train_y = sorted(glob(os.path.join(tr_mask_path,'**/*.bmp'),recursive=True))
val_x = sorted(glob(os.path.join(val_image_path,'*')))
val_y = sorted(glob(os.path.join(val_mask_path,'*')))
test_x = sorted(glob(os.path.join(test_image_path,'*')))
test_y = sorted(glob(os.path.join(test_mask_path,'**/*.bmp'),recursive=True))
print(len(val_x))
print(len(val_y))

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
def resize_data(images,masks,save_path):
    size = (512,512)
    size_reg = set()
    for idx, (x,y) in tqdm(enumerate(zip(images,masks))):
        name = x.split("/")[-1].split(".")[0]

        img = cv2.imread(x,cv2.IMREAD_COLOR)
        mask = cv2.imread(y)

        delta = img.shape[1]-img.shape[0]
        size_reg.add(img.shape)

        if delta!=0:
            d1 = delta//2
            d2 = delta - delta//2
            img = img[:,d1:-1*d2,:] # cut off sides to make aspect ratio more similar to 1:1 to avoid distortion
            mask = mask[:,d1:-1*d2,:]

        tmp_image_name = f"{name}.png"
        tmp_mask_name = f"{name}_mask.png"
        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)
        img = cv2.resize(img,size)
        mask = cv2.resize(mask,size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(mask_path,mask)
        cv2.imwrite(image_path,img)

    print(size_reg)

paths = [[train_x,train_y],[val_x,val_y],[test_x,test_y]]


for dataset,xy in zip(['train','val','test'],paths):
    create_dir(f'/home/kebl6872/Desktop/new_data/REFUGE2/{dataset}/image/')
    create_dir(f'/home/kebl6872/Desktop/new_data/REFUGE2/{dataset}/mask/')
    save_path = f'/home/kebl6872/Desktop/new_data/REFUGE2/{dataset}/'
    resize_data(xy[0],xy[1],save_path)

