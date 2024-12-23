import os
import random
import numpy as np
import torch
import cv2
import sys
import torch.nn as nn
from glob import glob
import random
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from pode import Contour, Polygon, divide, Requirement,Point
from pode import joined_constrained_delaunay_triangles

def seeding(seed):  # seeding the randomness
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_file(file):
    if not os.path.exists(file):
        open(file, "w")
        return file
    else:
        print(f"{file} Exists")
        base, extension = os.path.splitext(file)
        counter = 1
        new_file_path = f"{base}_{counter}{extension}"

        while os.path.exists(new_file_path):
            counter += 1
            new_file_path = f"{base}_{counter}{extension}"

        open(new_file_path,"w")
        print(f"File created: {new_file_path}")
        return new_file_path


def train_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



def f1_valid_two_classes(y_true,y_pred):
    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    # 0 == background class - will classify disk class as not 0 so I don't have to change the combined f1
    tp = np.sum(np.logical_and(y_true ==1, y_pred ==1))
    fp = np.sum(np.logical_and(y_true != 1, y_pred == 1))
    fn = np.sum(np.logical_and(y_true==1, y_pred==0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    return f1
def f1_valid_score(y_true, y_pred):
    if y_true.size() != y_pred.size():
        print(f' y true size: {y_true.size()} y_pred size: {y_pred.size()}')
        raise Exception(f'Check dimensions of y_true {y_true.size()} and y_pred {y_pred.size()}')

    smooth = 0.00001
    y_true = y_true.cpu().numpy().astype(int)
    y_pred = y_pred.cpu().numpy().astype(int)
    score_matrix = np.zeros(4)
    for i in range(3):
        tp = np.sum(np.logical_and(y_true == i, y_pred == i))
        fp = np.sum(np.logical_and(y_true != i, y_pred == i))
        fn = np.sum(np.logical_and(y_true == i, y_pred != i))
        f1 = 2*tp/(2*tp+fp+fn+smooth)
        score_matrix[i] = f1
    tp = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), np.logical_or(y_pred == 1, y_pred == 2)))
    fp = np.sum(np.logical_and(y_true == 0, np.logical_or(y_pred == 1, y_pred == 2)))
    fn = np.sum(np.logical_and(np.logical_or(y_true == 1, y_true == 2), y_pred == 0))
    f1 = 2 * tp / (2 * tp + fp + fn + smooth)
    score_matrix[3] = f1

    return score_matrix
def f1(tp,fp,fn):
    return 2*tp/(2*tp + (fp+fn))




def evaluate_centroids(map,indices,val):
    res = list()

    if len(indices) == 0: # no misclassifications therefore append an empty list
        return res

    for i in range(indices.shape[0]):
        map[indices[i, 0], indices[i, 1]] = 1



    (totalLabels, label_map, stats, centroids) = cv2.connectedComponentsWithStats(map, 8, cv2.CV_32S)



    for a, b, c in zip(stats[1:], centroids[1:], list(range(1, totalLabels))): #0th index corresponds to background component

        centroid_i,centroid_j = pick_rand(label_map,c)
        res.append([np.array([centroid_i, centroid_j]), stats[c, cv2.CC_STAT_AREA], val, ])



    return res

def pick_rand(map,label):
    indices = np.argwhere(map == label)
    l = list(range(indices.shape[0]))
    l_i = random.choice(l)
    return indices[l_i, :]



def generate_points(y_true,y_pred,num=1,detach=False):
    """Point generation shceme for teating"""


    """In case soft predictions have been passed through """
    if y_pred.shape[1] != 1:
        y_pred = y_pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)


    y_true = y_true.cpu().numpy().astype(int)

    if detach:
        y_pred = y_pred.detach().cpu().numpy().astype(int)
    else:
        y_pred = y_pred.cpu().numpy().astype(int)

    combined_results = list()
    #each of the following misclassifications will affect avg f1 score differently
    maps = [np.zeros((512,512) ).astype(np.uint8) for _ in range(6)]
    dc_misclass = np.argwhere(np.logical_and(y_true==1,y_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]
    cd_misclass = np.argwhere(np.logical_and(y_true==2,y_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    db_misclass = np.argwhere(np.logical_and(y_true==1,y_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    cb_misclass = np.argwhere(np.logical_and(y_true==2,y_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bd_misclass = np.argwhere(np.logical_and(y_true==0,y_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bc_misclass = np.argwhere(np.logical_and(y_true==0,y_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]





    """for plotting error maps"""
    # count = 1
    # map = np.zeros((512, 512))
    # for m1,m2 in [[dc_misclass,db_misclass],[cd_misclass,cb_misclass],[bd_misclass,bc_misclass]]:
    #
    #
    #     for i in range(m1.shape[0]):
    #         map[m1[i, 0], m1[i, 1]] += count
    #
    #     count+=1
    #
    #     for i in range(m2.shape[0]):
    #         map[m2[i, 0], m2[i, 1]] += count
    #     count+=1
    #
    # plt.figure(figsize=(10, 10))
    # hex_colors = ['#000000','#059000',  '#3cb371', '#0000FF','#05c6ff','#FF0000','#FF6666', ]  # lighter blue
    # custom_cmap = mcolors.ListedColormap(hex_colors, name='custom_cmap')
    # x, y = np.indices(map.shape)  # Create a grid of x, y indices
    # points = np.column_stack((x.flatten(), y.flatten()))
    # labels = map.flatten()  # Flatten the label array as well
    # plt.scatter(points[:, 1], points[:, 0], c=labels, cmap=custom_cmap)
    # plt.show()




    dc_centroids = evaluate_centroids(maps[0],dc_misclass,1)
    combined_results.extend(dc_centroids)
    cd_centroids = evaluate_centroids(maps[1],cd_misclass,2,)
    combined_results.extend(cd_centroids)
    db_centroids = evaluate_centroids(maps[2],db_misclass,1,)
    combined_results.extend(db_centroids)
    bd_centroids = evaluate_centroids(maps[3],bd_misclass,0,)
    combined_results.extend(bd_centroids)
    cb_centroids = evaluate_centroids(maps[4],cb_misclass,2)
    combined_results.extend(cb_centroids)
    bc_centroids = evaluate_centroids(maps[5],bc_misclass,0)
    combined_results.extend(bc_centroids)



    combined_results = sorted(combined_results, key=lambda x: x[1])



    return [(x[0][0],x[0][1],x[2],) for x in combined_results[-1*num:]] #returns p_i,p_j,p_label

def generate_points_batch(y_true,y_pred,num=1,detach=False):
    B = y_true.shape[0]
    points = np.zeros((B,num,2))
    point_labels = np.zeros((B,num,1))

    for i in range(B):
        y_true_input = y_true[i,:,:,:]
        y_true_input = y_true_input[np.newaxis,:,:,:]# need to add pseudo batch dimension to work w generate_points
        y_pred_input = y_pred[i, :, :, :]
        y_pred_input = y_pred_input[np.newaxis, :, :, :]  # need to add pseudo batch dimension to work w generate_points

        gen_points = generate_points(y_true_input,y_pred_input,num,detach=detach)

        for j in range(num):
            try:
                points[i,j,0] = gen_points[j][0]
                points[i,j,1] = gen_points[j][1]
                point_labels[i,j,0] = gen_points[j][2]
            except:
                print(f'gen points error {gen_points}')# points[i,j,0] = gen_points[j][0]
                points[i,j,1] = points[i,j-1,1]
                points[i, j, 0] = points[i, j - 1, 0]
                point_labels[i,j,0] = point_labels[i,j-1,0]

    return points,point_labels



def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)                # (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (512, 512, 3)
    return mask








