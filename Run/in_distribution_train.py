import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import time
import os
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_aug.data import train_test_split,GS1_dataset,PAPILA_dataset,RIGA_dataset
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss, DiceFocalLoss
from utils import *
from random import randint
from glob import glob
import math
from train_neat import train_batch, train_log, test, train, get_data,make_loader

"""in distribution unet training scripts"""


def make(config):

    """ defines size of training and validation sets for each dataset - as close to 70:30 split as possible"""
    set_sizes = {'GAMMA': 70, 'GS1': 71, 'MAGRABIA': 66, 'BR': 137, 'MESSIDOR': 322, 'G1020': 552, 'ORIGA': 455}

    if config.dataset=="GS1":
        gs1 = True
    else:
        gs1 = False

    if config.dataset == "GAMMA":
        gamma=True
    else:
        gamma=False

    magrabia = True if config.dataset == 'MAGRABIA' else False
    g1020 = True if config.dataset == 'G1020' else False
    br = True if config.dataset == 'BR' else False
    messidor = True if config.dataset == 'MESSIDOR' else False
    origa = True if config.dataset == 'ORIGA' else False


    if magrabia or g1020 or br or messidor or origa:
        """ gs1 and gamma were manually split into train val sets early into the project, for datasets added later on into the project, I use a random split with seed 42"""
        total_set = get_data(train=True,transform=config.transform,origa=origa,messidor=messidor,magrabia=magrabia,br=br,g1020=g1020)
        rng_state = torch.get_rng_state()
        generator = torch.Generator().manual_seed(42)
        train_size = set_sizes[config.dataset]
        val_size = len(total_set) - train_size
        train,test = torch.utils.data.random_split(total_set,[train_size,val_size],generator=generator)
        torch.set_rng_state(rng_state)

    else:
        train,test = get_data(train=True,transform=config.transform,gs1=gs1,gamma=gamma),get_data(train=False,gs1=gs1,gamma=gamma,)

    eval_criterion = f1_valid_score
    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False)

    criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5,lambda_ce=0.5)

    model = UNet(3,config.classes,config.base_c,config.kernels,config.norm_name)


    return model,train_loader,test_loader,criterion,eval_criterion
def model_pipeline(hyperparameters):
    with wandb.init(project="In_Distribution_UNet_experiment",config=hyperparameters, dir = '/data/engs-mlmi1/kebl6872/wandb'):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion = make(config)
        # print(model)
        model = model.to(device)
        train(model,train_loader,test_loader,criterion,eval_criterion,config)

    return model



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Specify Parameters')

    parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
    parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
    parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
    parser.add_argument('-l1', nargs='+', type=int, metavar='kernels',
                        help='number of channels in unet layers')  # Use like: python test.py -l 1 2 3 4
    parser.add_argument('--base_c', metavar='--base_c', default=12, type=int,
                        help='base_channel which is the first output channel from first conv block')

    parser.add_argument('no_runs', metavar='no_runs', type=int,
                        help='how many random seeds you want to run experiment on')

    args = parser.parse_args()
    lr, batch_size, gpu_index, no_runs = args.lr, args.b_s, args.gpu_index, args.no_runs
    base_c = args.base_c
    kernels = args.l1
    norm_name = 'batch'
    model_name = 'unet'
    print(torch.cuda.is_available())
    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')

    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')
    set_sizes = {'GAMMA':70,'GS1':71,'MAGRABIA':66,'BR':137,'MESSIDOR':322,'G1020':552,'ORIGA':455}
    dataset_names =  ['GAMMA','BR','MAGRABIA','MESSIDOR','GS1','G1020','ORIGA']

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    for a in dataset_names:

        epochs = math.ceil(8000/(math.ceil(set_sizes[a]/batch_size))) #datasets are of different sizes and training time determined by epochs,  this keeps the total number of updates for each in-dist unet roughly the same
        print(f'num epochs: {epochs}')
        for _ in range(no_runs):
            config = dict(epochs=epochs, classes=3, base_c = 12, kernels=kernels, norm_name=norm_name,batch_size=batch_size, learning_rate=lr, dataset=a,architecture=model_name,seed=401,transform='baseline',device=device)
            config["seed"] = randint(2000,100000)
            seeding(config["seed"])


            data_save_path = f'/home/kebl6872/Desktop/new_data/REFUGE/in_distribution_unet/'

            create_dir(data_save_path + f'seed_{config["seed"]}')
            checkpoint_path_lowloss = data_save_path + f'seed_{config["seed"]}/lr_{lr}_bs_{batch_size}_lowloss.pth'
            checkpoint_path_final = data_save_path + f'seed_{config["seed"]}/lr_{lr}_bs_{batch_size}_final.pth'
            create_file(checkpoint_path_lowloss)
            create_file(checkpoint_path_final)
            config['low_loss_path']=checkpoint_path_lowloss
            config['final_path'] = checkpoint_path_final

            model = model_pipeline(config)
