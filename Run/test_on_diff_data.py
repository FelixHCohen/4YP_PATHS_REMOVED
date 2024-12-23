import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
import monai
from tqdm import tqdm
import time
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_aug.data import train_test_split
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss
from utils import *
import cv2
import glob
import monai
from train_neat import get_data, get_gs1_or_rim_data
device = torch.device('cuda:0')



def make(config,path):
    model = UNet(3, config["classes"], config["base_c"], config["kernels"], config["norm_name"])
    origa,magrabia,messidor,br,g1020 = False,False,False,False,False
    if not os.path.exists(path):
        print("path does not exist")
    checkpoint = torch.load(path,map_location=device)
    model.load_state_dict(checkpoint,strict=False)
    model.eval()
    if config['testset']=='GAMMA':
        train= get_data(train=True,return_path=False,gamma=True,transform=False,)
        test = get_data(train=False,return_path=False,gamma=True,transform=False,)
        if config['dataset']!= 'GAMMA':
            total = torch.utils.data.ConcatDataset([train, test])
        else:
            total = test

    elif config["testset"]== "GS1":
        train = get_data(train=True, return_path=False, gs1=True, transform=False, )
        test = get_data(train=False, return_path=False, gs1=True, transform=False, )
        if config['dataset']!='GS1':
            total = torch.utils.data.ConcatDataset([train, test])
        else:

          total = test

    elif config['testset']=='MESSIDOR':
        total = get_data(train=True,messidor=True,transform=False,return_path=False)
        messidor=True
    elif config['testset']=='G1020':
        total = get_data(train=True,g1020=True,transform=False,return_path=False)
        g1020=True
    elif config['testset'] == 'PAPILA':
        total = get_data(train=True,return_path=False,transform=False,papila=True)
    elif config['testset']=='REFUGE':
        total = get_data(train=False,refuge_test=True, return_path=False, transform=False, )
    elif config['testset'] == 'BR':
        total = get_data(train=True, return_path=False, transform=False, br=True)
        br=True
    elif config['testset'] == 'MAGRABIA':
        total = get_data(train=True, return_path=False, transform=False, magrabia=True)
        magrabia=True
    elif config['testset'] == 'ORIGA':

        total = get_data(train=True, return_path=False, transform=False, origa=True)
        origa=True


    else:
        total = get_data(train=False,refuge_test=True,return_path=False,transform=False,)

    if magrabia or g1020 or br or messidor or origa:
        set_sizes = {'GAMMA': 70, 'GS1': 71, 'MAGRABIA': 66, 'BR': 137, 'MESSIDOR': 322, 'G1020': 552, 'ORIGA': 455}
        #total_set = get_data(train=True,transform=config.transform,origa=origa,messidor=messidor,magrabia=magrabia,br=br,g1020=g1020)
        rng_state = torch.get_rng_state()
        generator = torch.Generator().manual_seed(42)
        train_size = set_sizes[config['testset']]
        val_size = len(total) - train_size
        _,total = torch.utils.data.random_split(total,[train_size,val_size],generator=generator)
        torch.set_rng_state(rng_state)

    train_loader = DataLoader(dataset=total, batch_size=1, shuffle=True, )

    criterion = f1_valid_score
    return model,train_loader,criterion


def plot_output(output,image,label,score,point_tuples,detach=False):
    color_map = {0:'red',1:'green',2:'blue'}

    image_np = image[0,:,:,:].cpu().numpy().transpose(1,2,0)
    image_np = ((image_np * 127.5)+127.5).astype(np.uint16)
    if not detach:
        output = output[0,:,:,:].cpu().numpy().transpose(1,2,0)
    else:
        output = output[0,:,:,:].detach().cpu().numpy().transpose(1,2,0)
    label = label[0,:,:,:].cpu().numpy().transpose(1,2,0)
    label = np.repeat(label,3,2)
    output = np.repeat(output,3,2)
    d = {0:0,1:128,2:255}
    vfunc = np.vectorize(lambda x: d[x])

    # Apply the vectorized function to the array
    label = vfunc(label)
    output = vfunc(output)


    rows = 1
    cols = 3

    # Create a figure with the specified size
    plt.figure(figsize=(15, 15))
    plt.subplot(rows, cols, 1)
    plt.imshow(image_np)
    plt.title("image")
    plt.axis('off')
    # Plot the mask in the even-numbered subplot
    plt.subplot(rows, cols, 2)
    plt.imshow(output)
    plt.title(f"output avg score: {score}")
    for y, x, val in point_tuples: #point tuples stored in index e.g. ij = y,x
        print(f'({x},{y}): {val}')
        circle = plt.Circle((x,y),3,color=color_map[val])
        plt.gca().add_patch(circle)
    plt.axis('off')
    plt.subplot(rows, cols, 3)
    plt.imshow(label,)
    plt.title("ground truth")
    plt.axis('off')




    # Show the plot
    plt.show()

def assd(y_pred,y):
    # assuming y_pred and y are pytorch tensors of shape (1, H, W) with values 0, 1, or 2
    # convert them to one-hot format of shape (1, 3, H, W)
    y_pred_onehot = monai.networks.utils.one_hot(y_pred, num_classes=3)
    y_onehot = monai.networks.utils.one_hot(y, num_classes=3)

    # # create a surface distance metric object
    # assd = monai.metrics.compute_average_surface_distance(y_pred_onehot,y_onehot,
    #     include_background=False,  # exclude the background class
    #     symmetric=True,  # compute the symmetric distance
    #     distance_metric="euclidean" , # use the euclidean distance
    #     spacing='none'
    # )
    # create a surface distance metric object
    metric = monai.metrics.SurfaceDistanceMetric(
        include_background=False,  # exclude the background class
        symmetric=True,  # compute the symmetric distance
        distance_metric="euclidean" , # use the euclidean distance
        reduction='none'
    )



    asd = metric(y_pred_onehot,y_onehot)

    # aggregate the metric over the batch
    # this returns a tensor of shape (3,) with the average surface distance for each class

    #asd = metric.aggregate()

    # print the result
    return asd[0,0].item(), asd[0,1].item()

def test_pipeline(config,model_path):
    model,loader,criterion = make(config,model_path)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        val_score = 0
        test_set_val = 0
        total = 0
        disc_assd_score = 0
        cup_assd_score = 0
        for i,(images,labels,) in enumerate(loader):
            images,labels = images.to(device,dtype=torch.float32),labels.to(device)
            outputs = model(images).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = criterion(outputs,labels)

            val_score += score
            assd_score = assd(outputs,labels)
            disc_assd_score += assd_score[0]
            cup_assd_score += assd_score[1]
            total += labels.size(0)

            if i < 0:
                point_tuples = generate_points(labels,outputs,4)

                #points_df = save_points(point_tuples,im_path,mask_path,points_df)
                plot_output(outputs,images,labels,[assd_score[0],assd_score[1]],point_tuples)



        val_score /= total #currently needs batch size 1
        print(f'total assd cup: {cup_assd_score}, total assd disc: {disc_assd_score}')
        disc_assd_score/= total
        cup_assd_score /= total
        # print(f"cup F1 {val_score[2]}\ndisc F1: {val_score[3]}")
        # print(f'val score avg: {val_score[1]/2 + val_score[2]/2}')
        return val_score[1],val_score[2],val_score[3],disc_assd_score,cup_assd_score





if __name__ == "__main__":
    #model_paths = glob.glob(f'/home/kebl6872/Desktop/weakunet/Checkpoint/seed/**/*lowloss.pth',recursive=True) train only seed:1065,1059 ~~ 1092,1025,1080 ~~ all 2746,2627,2644
    mp1 = [f"/home/kebl6872/Desktop/weakunet/Checkpoint/seed/{seed}/lr_0.0003_bs_8_lowloss.pth" for seed in [1092,1025,1080,]]
    mp2 = [f"/home/kebl6872/Desktop/weakunet/Checkpoint/seed/{seed}/lr_0.0003_bs_16_lowloss.pth" for seed in [1033,1026,]]

    model_paths = mp1 #+ mp2
    testset_models = glob.glob('/home/kebl6872/Documents/baseline_full/**/*.pth',recursive=True)


    for testset in ['BR','G1020','GAMMA','GS1','MAGRABIA','MESSIDOR','ORIGA']:
        print(testset)
        outer_avg = 0
        cup_avg = 0
        disc_avg = 0
        c_assd_avg = 0
        d_assd_avg = 0
        """added this line for my in distribution models"""

       # model_paths = list(filter(lambda s: testset in s, testset_models))

        for model_path in model_paths:

            config = dict(classes=3, base_c=12, kernels=[6, 12, 24, 48], norm_name='batch', batch_size=16, lr=3e-4,
                          seed=336, dataset=testset, testset=testset)
            outer,cup,disc,d_assd,c_assd = test_pipeline(config,model_path)
            outer_avg +=outer
            cup_avg += cup
            disc_avg += disc
            c_assd_avg += c_assd
            d_assd_avg += d_assd

        print(f'FINAL RESULTS FOR {testset}\nouter: {outer_avg/len(model_paths)}\ncup: {cup_avg/len(model_paths)}\ndisc: {disc_avg/len(model_paths)}\navg:{(outer_avg+cup_avg)/(2*len(model_paths))}\n disc assd: {d_assd_avg/len(model_paths)}\ncup assd: {c_assd_avg/len(model_paths)}')



