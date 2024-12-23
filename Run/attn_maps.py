import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import random


def plot(df,idx,level,layer):
    reg = {'b':0,'d1':6,'d2':12}
    attn_index = 4 + reg[level] + layer
    points = ast.literal_eval(df.iloc[idx,3])[0]
    labels = ast.literal_eval(df.iloc[idx, 4])[0]
    img_path = df.iloc[idx,1]
    img_path = ast.literal_eval(img_path)[0]
    print(f'img path: {img_path}')

    img = cv2.imread(img_path,)
    print(type(img))
    attn_map = np.load(f'{df.iloc[idx,attn_index]}.npy')
    print(attn_map.shape)
    img_feats = attn_map.shape[-1]
    h = w = int(np.sqrt(img_feats))
    num_points = len(points)

    attn_map = np.reshape(attn_map,(1,num_points,h,w))
    print(attn_map.shape)
    print(labels[0])
    #print(attn_map[0,0,int(points[0][0]/16)-5:int(points[0][0]/16)+5,:int(points[0][1]/16)-5:int(points[0][1]/16)+5])

    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    num_points = min(5,num_points)
    cols = num_points
    rows = 2
    plt.figure(figsize=(25, 25))

    for i in range(num_points):
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.flip(img,2))
        circle = plt.Circle((int(points[i][1]), int(points[i][0])), 7, color=color_map[int(labels[i][0])])
        plt.gca().add_patch(circle)
        plt.subplot(rows,cols,i+1+num_points)
        plt.imshow(attn_map[0,i,:,:])
    plt.show()

def progression(df,idx,):

    points = ast.literal_eval(df.iloc[idx,3])[0]
    labels = ast.literal_eval(df.iloc[idx, 4])[0]
    img_path = df.iloc[idx,1]
    img_path = ast.literal_eval(img_path)[0]
    img = cv2.imread(img_path,)


    num_points = len(points)


    color_map = {0: 'red', 1: 'green', 2: 'blue'}
    rand_point = random.randint(0,num_points-1)
    cols = 12
    rows = 1
    plt.figure(figsize=(25, 25))
    plt.subplot(rows,cols,1)
    plt.imshow(np.flip(img,2))
    circle = plt.Circle((int(points[rand_point][1]), int(points[rand_point][0])), 6, color=color_map[int(labels[rand_point][0])])
    plt.gca().add_patch(circle)
    for i in range(3):
        attn_map = np.load(f'{df.iloc[idx,5+i]}.npy')
        img_feats = attn_map.shape[-1]
        h = w = int(np.sqrt(img_feats))
        attn_map = np.reshape(attn_map, (1, num_points, h, w))
        plt.subplot(rows,cols,i+2)
        plt.imshow(attn_map[0,rand_point,:,:])
    for i in range(3):
        attn_map = np.load(f'{df.iloc[idx,11+i]}.npy')
        img_feats = attn_map.shape[-1]
        h = w = int(np.sqrt(img_feats))
        attn_map = np.reshape(attn_map, (1, num_points, h, w))
        plt.subplot(rows,cols,i+5)
        plt.imshow(attn_map[0,rand_point,:,:])
    for i in range(5):
        attn_map = np.load(f'{df.iloc[idx,17+i]}.npy')
        img_feats = attn_map.shape[-1]
        h = w = int(np.sqrt(img_feats))
        attn_map = np.reshape(attn_map, (1, num_points, h, w))
        plt.subplot(rows,cols,i+8)
        plt.imshow(attn_map[0,rand_point,:,:])
    plt.show()

def progression2(df,num_imgs=8,l1=3,l2=3,l3=5):
    cols = 1 + l1 + l2 + l3
    rows = num_imgs
    # Use plt.subplots() to create a figure and an array of axes
    fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
    img_idxs = random.sample(list(range(10)),num_imgs)
    for idx in range(num_imgs):
        img_idx = img_idxs[idx]
        points = ast.literal_eval(df.iloc[img_idx,3])[0]
        labels = ast.literal_eval(df.iloc[img_idx, 4])[0]
        img_path = df.iloc[img_idx,1]
        img_path = ast.literal_eval(img_path)[0]
        img = cv2.imread(img_path,)


        num_points = len(points)


        color_map = {0: 'red', 1: 'green', 2: 'blue'}
        rand_point = random.randint(0,num_points-1)

        # Plot the original image on the first axis
        axs[idx, 0].imshow(np.flip(img,2)) # Use two indices to access the axes
        circle = plt.Circle((int(points[rand_point][1]), int(points[rand_point][0])), 12, color=color_map[int(labels[rand_point][0])])
        axs[idx, 0].add_patch(circle)
        # Loop through the attention maps and plot them on the remaining axes
        for i in range(l1):
            attn_map = np.load(f'{df.iloc[img_idx,5+i]}.npy')
            img_feats = attn_map.shape[-1]
            h = w = int(np.sqrt(img_feats))
            attn_map = np.reshape(attn_map, (1, num_points, h, w))
            axs[idx, i+1].imshow(attn_map[0,rand_point,:,:]) # Use two indices to access the axes
        for i in range(l2):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+i]}.npy')
            img_feats = attn_map.shape[-1]
            h = w = int(np.sqrt(img_feats))
            attn_map = np.reshape(attn_map, (1, num_points, h, w))
            axs[idx, i+1 + l1].imshow(attn_map[0,rand_point,:,:]) # Use two indices to access the axes
        for i in range(l3):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+2*l2+i]}.npy')
            img_feats = attn_map.shape[-1]
            h = w = int(np.sqrt(img_feats))
            attn_map = np.reshape(attn_map, (1, num_points, h, w))
            axs[idx, i+1 + l1 + l2].imshow(attn_map[0,rand_point,:,:]) # Use two indices to access the axes
    plt.show()

"""This plots how image features attend to a random point for each level"""
def progression3(df,idx,num_imgs=8,l1=3,l2=3,l3=5):
    cols = 1 + l1 + l2 + l3
    rows = num_imgs
    # Use plt.subplots() to create a figure and an array of axes
    fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
    img_idxs = random.sample(list(range(10)),num_imgs)
    for idx in range(num_imgs):
        img_idx = img_idxs[idx]
        points = ast.literal_eval(df.iloc[img_idx,3])[0]
        labels = ast.literal_eval(df.iloc[img_idx, 4])[0]
        img_path = df.iloc[img_idx,1]
        img_path = ast.literal_eval(img_path)[0]
        img = cv2.imread(img_path,)


        num_points = len(points)


        color_map = {0: 'red', 1: 'green', 2: 'blue'}
        rand_point = random.randint(0,num_points-1)

        # Plot the original image on the first axis
        axs[idx, 0].imshow(np.flip(img,2)) # Use two indices to access the axes
        circle = plt.Circle((int(points[rand_point][1]), int(points[rand_point][0])), 12, color=color_map[int(labels[rand_point][0])])
        axs[idx, 0].add_patch(circle)
        # Loop through the attention maps and plot them on the remaining axes
        for i in range(l1):
            attn_map = np.load(f'{df.iloc[img_idx,5+l1+i]}.npy')
            img_feats = attn_map.shape[-2]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1, h, w,num_points))
            axs[idx, i+1].imshow(attn_map[0,:,:,rand_point]) # Use two indices to access the axes
        for i in range(l2):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+l2+i]}.npy')

            img_feats = attn_map.shape[-2]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1, h, w,num_points))
            axs[idx, i+1 + l1].imshow(attn_map[0,:,:,rand_point]) # Use two indices to access the axes
        for i in range(l3):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+2*l2+l3+i]}.npy')
            img_feats = attn_map.shape[-2]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1,  h, w,num_points))
            axs[idx, i+1 + l1 + l2].imshow(attn_map[0,:,:,rand_point]) # Use two indices to access the axes
    plt.show()

"""THis function plots how image features attend to every user point for a given image"""
def progression4(df,l1=3,l2=3,l3=5):

    # Use plt.subplots() to create a figure and an array of axes

    img_idx= random.choice(list(range(len(df))))

    points = ast.literal_eval(df.iloc[img_idx,3])[0]
    labels = ast.literal_eval(df.iloc[img_idx, 4])[0]
    img_path = df.iloc[img_idx,1]
    img_path = ast.literal_eval(img_path)[0]
    img = cv2.imread(img_path,)
    print(img.shape)


    num_points = len(points)

    cols = 1 + l1 + l2 + l3
    rows = num_points

    color_map = {0: 'red', 1: 'green', 2: 'blue'}

    fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
    idx = 0
    for rand_point in range(num_points):
        # Plot the original image on the first axis
        axs[idx, 0].imshow(np.flip(img,2)) # Use two indices to access the axes
        circle = plt.Circle((int(points[rand_point][1]), int(points[rand_point][0])), 12, color=color_map[int(labels[rand_point][0])])
        axs[idx, 0].add_patch(circle)
        # Loop through the attention maps and plot them on the remaining axes
        for i in range(l1):
            attn_map = np.load(f'{df.iloc[img_idx,5+l1+i]}.npy')
            img_feats = attn_map.shape[-2]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1, h, w,num_points))
            axs[idx, i+1].imshow(attn_map[0,:,:,rand_point]) # Use two indices to access the axes
        for i in range(l2):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+l2+i]}.npy')

            img_feats = attn_map.shape[-2]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1, h, w,num_points))
            axs[idx, i+1 + l1].imshow(attn_map[0,:,:,rand_point]) # Use two indices to access the axes
        for i in range(l3):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+2*l2+l3+i]}.npy')
            img_feats = attn_map.shape[-2]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1,  h, w,num_points))
            axs[idx, i+1 + l1 + l2].imshow(attn_map[0,:,:,rand_point]) # Use two indices to access the axes
        idx +=1
    plt.show()

def progression5(df,l1=3,l2=3,l3=5):

    # Use plt.subplots() to create a figure and an array of axes

    img_idx= random.choice(list(range(len(df))))

    points = ast.literal_eval(df.iloc[img_idx,3])[0]
    labels = ast.literal_eval(df.iloc[img_idx, 4])[0]
    img_path = df.iloc[img_idx,1]
    img_path = ast.literal_eval(img_path)[0]
    img = cv2.imread(img_path,)


    num_points = len(points)

    cols = 1 + l1 + l2 + l3
    rows = num_points

    color_map = {0: 'red', 1: 'green', 2: 'blue'}

    fig, axs = plt.subplots(rows, cols, figsize=(25, 25))
    idx = 0
    for rand_point in range(num_points):
        # Plot the original image on the first axis
        axs[idx, 0].imshow(np.flip(img,2)) # Use two indices to access the axes
        circle = plt.Circle((int(points[rand_point][1]), int(points[rand_point][0])), 12, color=color_map[int(labels[rand_point][0])])
        axs[idx, 0].add_patch(circle)
        # Loop through the attention maps and plot them on the remaining axes
        for i in range(l1):
            attn_map = np.load(f'{df.iloc[img_idx,5+i]}.npy')
            img_feats = attn_map.shape[-1]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1,num_points,h,w))
            axs[idx, i+1].imshow(attn_map[0,rand_point,:,:]) # Use two indices to access the axes
        for i in range(l2):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+i]}.npy')

            img_feats = attn_map.shape[-1]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1,num_points,h,w))
            axs[idx, i+1 + l1].imshow(attn_map[0,rand_point,:,:]) # Use two indices to access the axes
        for i in range(l3):
            attn_map = np.load(f'{df.iloc[img_idx,5+2*l1+2*l2+i]}.npy')
            img_feats = attn_map.shape[-1]
            h = w = int(np.sqrt(img_feats))
            print(f'h:{h} shape:{attn_map.shape}')
            attn_map = np.reshape(attn_map, (1,  num_points,h,w))
            axs[idx, i+1 + l1 + l2].imshow(attn_map[0,rand_point,:,:]) # Use two indices to access the axes
        idx +=1
    plt.show()


df = pd.read_csv('/home/kebl6872/Desktop/messidor_single_head_attn_maps.csv')
#df = pd.read_csv('/home/kebl6872/Desktop/Messidor_transformattn_maps.csv')
# for level in ['b','d1','d2']:
#     for layer in range(1,4):
progression2(df,6,4,4,4)
progression3(df,15,6,4,4,4)
progression4(df,4,4,4)
#progression5(df,3,3,5)
attn_map = np.load(f'{df.iloc[0,8]}.npy')
print(attn_map.shape)
print(np.sum(attn_map[0,0,:]))
