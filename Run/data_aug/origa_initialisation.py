
import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

"""Source for origa masks wasnt binary, as origa masks were determined by ellipse fitting I deemed this an acceptable way of determining the binary masks: all it does is set all non 0 values to 1 and then smooths the result using nearest neighbours """

def smooth(array):
  # create a copy of the input array
  result = array.copy()
  # get the shape of the array
  rows, cols,channels = array.shape
  # loop through each element of the array
  for i in range(rows):
    for j in range(cols):
      # if the element is 255, check its 4 directional neighbours
      if array[i, j,0] == 255:
        # count the number of neighbours that are also 255
        count = 0
        # check the left neighbour
        if j > 0 and array[i, j-1,0] == 255:
          count += 1
        # check the right neighbour
        if j < cols-1 and array[i, j+1,0] == 255:
          count += 1
        # check the top neighbour
        if i > 0 and array[i-1, j,0] == 255:
          count += 1
        # check the bottom neighbour
        if i < rows-1 and array[i+1, j,0] == 255:
          count += 1
        # if the count is less than 2, make the element 0
        if count < 2:
          result[i, j,:] = np.array([0,0,0])
  # return the smoothed array
  return result
def resize_data(images, disc_masks,cup_masks, save_path):
    size = (512,512)

    paired_masks = [[a,b] for a,b in zip(disc_masks,cup_masks)]

    for idx, (x, y) in tqdm(enumerate(zip(images, paired_masks)), total=len(images)):
        """ Extracting the name """
        name = x.split("/")[-1].split(".")[0] # extracts name from path/name.jpg
        y_disc,y_cup = y

        """ Reading image and mask """
        img = cv2.imread(x, cv2.IMREAD_COLOR)
        y_disc = cv2.imread(y_disc)

        y_disc = smooth(np.where(y_disc != 0,255,y_disc))
        y_cup = cv2.imread(y_cup)
        y_cup = smooth(np.where(y_cup != 0,255,y_cup))

        y_disc = np.where(y_cup==255,2,y_disc)

        y_disc = np.where(y_disc==255,1,y_disc)

        y_disc = np.where(y_disc==0,255,y_disc)
        y_disc = np.where(y_disc==1,128,y_disc)
        y_disc = np.where(y_disc==2,0,y_disc)
        mask = y_disc
        tmp_image_name = f"{name}.png"
        tmp_mask_name = f"{name}_mask.png"


        image_path = os.path.join(save_path, "image", tmp_image_name)
        mask_path = os.path.join(save_path, "mask", tmp_mask_name)

        cv2.imwrite(mask_path,mask)
        cv2.imwrite(image_path, img)