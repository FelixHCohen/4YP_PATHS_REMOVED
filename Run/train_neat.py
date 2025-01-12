import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm
import time
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from data_aug.data import train_test_split,GS1_dataset,RIGA_dataset,PAPILA_dataset
from UNET.UNet_model import UNet
from monai.losses import DiceCELoss, DiceFocalLoss
from utils import *
import argparse
from random import randint

"""Training script for the unet trained using same training and validation sets as the interactive model"""

def train_batch(images,labels,model,optimizer,criterion,config):
    images,labels = images.to(config.device,dtype=torch.float32),labels.to(config.device,dtype=torch.float32)
    outputs = model(images)
    loss = criterion(outputs,labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
def train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct + 1).zfill(5)} batches: {loss:.3f}")

def save_model(path,name):
    artifact = wandb.Artifact(name=name, type="model")
    # Add the model file to the artifact
    artifact.add_file(path)
    # Log the artifact as an output of the run
    wandb.run.log_artifact(artifact)

def test(model,test_loader,criterion,config,best_valid_score,example_ct):
    model.eval()

    with torch.no_grad():
        val_score = 0
        f1_score_record = np.zeros(4)
        total = 0
        for images,labels in test_loader:
            images,labels = images.to(config.device,dtype=torch.float32),labels.to(config.device)
            outputs = model(images).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
            score = criterion(outputs,labels)
            val_score += score[1].item()/2 + score[2].item()/2
            f1_score_record += score
            total += labels.size(0)
        f1_score_record /= len(test_loader) #currently only works with batchsize=1
        val_score /= len(test_loader)
        print(f"model tested on {total} images" +
                  f"val_score: {val_score} f1_scores {f1_score_record}")
        wandb.log({"val_score": val_score, "Validation Background F1":f1_score_record[0],"Validation Outer Ring F1":f1_score_record[1],
                   "Validation Cup F1": f1_score_record[2],"Validation Disk F1": f1_score_record[3]},step=example_ct)


        #     # Save the model in the exchangeable ONNX format
        # torch.onnx.export(model, images, "model.onnx",opset_version=16)
        # wandb.save("model.onnx")

    model.train()

    if val_score > best_valid_score[0]:
        data_str = f"Valid score improved from {best_valid_score[0]:2.8f} to {val_score:2.8f}. Saving checkpoint: {config.low_loss_path}"
        print(data_str)
        best_valid_score[0] = val_score
        torch.save(model.state_dict(), config.low_loss_path)


    return f1_score_record[2],f1_score_record[3]



def train(model, loader,test_loader, criterion, eval_criterion, config):

    wandb.watch(model,criterion,log='all',log_freq=50) #this is freq of gradient recordings

    example_ct = 0
    batch_ct = 0
    optimizer = torch.optim.Adam(model.parameters(), config.learning_rate)
    best_valid_score = [0.0]#in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs),desc='Epochs'):

        avg_epoch_loss = 0.0
        start_time = time.time()
        for _,(images, labels) in enumerate(tqdm(loader,desc=f'Epoch: {epoch+1}')):

            loss = train_batch(images,labels,model,optimizer,criterion,config)
            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct +=1

            if ((batch_ct+1)%4)==0:
                train_log(loss,batch_ct,epoch)



        cup_loss,disk_loss = test(model,test_loader,eval_criterion,config,best_valid_score,batch_ct)
        avg_epoch_loss/=len(loader)
        end_time = time.time()
        iteration_mins,iteration_secs = train_time(start_time,end_time)
        data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s\n'
        data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
        data_str += f'\t Val Cup: {cup_loss:.8f}\n'
        data_str += f'\t Val Disk: {disk_loss:.8f}\n'
        print(data_str)
    torch.save(model.state_dict(),config.final_path)
    save_model(config.final_path,"final_model")




def get_data(train,gs1=False,messidor=False,rim=False,refuge_test=False,transform=False,return_path=False,papila=False,magrabia=False,br=False,origa=False,gamma=False,g1020 = False,):
    if gs1:
        return get_gs1_or_rim_data(train,transform,return_path=return_path)

    if rim:
        return get_gs1_or_rim_data(train,transform,rim=True)

    if refuge_test:
        dataset = "REFUGE2"
        x = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/test/image/*"))
        y = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/test/mask/*"))
        print(f'testing dataset of size: {len(x)}')
    elif papila:
        dataset = "PAPILA"
        x = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/image/*"))
        y1 = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/mask1/*"))
        y2 = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/mask2/*"))
        return PAPILA_dataset(x,y1,y2,transform=transform,return_path=return_path)

    else:
        if train:
            dataset_type = 'train'
        if train == False and gamma==False:
            dataset_type= 'val'
        if train == False and gamma ==True:
            dataset_type='test'

        if gamma == True:
            dataset = 'Gamma'
        elif messidor==True:
            dataset = 'MESSIDOR'
            dataset_type = ''
        elif g1020 == True:
            dataset = 'G1020'
            dataset_type = ''
        elif origa == True:
            dataset = 'ORIGA'
            dataset_type = ''
        elif br == True:
            dataset = 'br'
            dataset_type = ''
        elif magrabia == True:
            dataset = 'magrabia'
            dataset_type = ''
        else:
            dataset = 'REFUGE2'
        x = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/image/*"))
        y = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/mask/*"))
        data_str = f"Training dataset size: {len(x)}"

        print(data_str)


    if messidor or br or magrabia:
        dataset = RIGA_dataset(x,y,transform=transform,return_path=return_path)
    else:
        dataset = train_test_split(x,y,transform=transform,return_path=return_path)


    return dataset

def get_gs1_or_rim_data(train,transform,rim=False,return_path=False):
    if train:
        dataset_type = 'train'
    else:
        dataset_type = 'test'

    if rim:
        dataset = "RIMDL"
    else:
        dataset = "GS1_square"

    gs1_x = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/image/*"))
    gs1_c = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/cup_mask/*"))
    gs1_d = sorted(glob(f"/home/kebl6872/Desktop/new_data/{dataset}/{dataset_type}/disc_mask/*"))
    data_str = f"{dataset_type} dataset size: {len(gs1_x)}"
    print(data_str)

    dataset = GS1_dataset(gs1_x,gs1_c,gs1_d,transform=transform,return_path=return_path,disc_only=False)

    return dataset
def make_loader(dataset,batch_size):
    loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True,)
    return loader
def make(config):

    eval_criterion = f1_valid_score

    train1, train2,= get_data(train=False, transform=config.transform, ), get_data(train=True,transform=config.transform)
    train = torch.utils.data.ConcatDataset([train1, train2,])


    test1, test2 = get_data(train=True, gs1=True), get_data(train=True, gamma=True)
    test = torch.utils.data.ConcatDataset([test1, test2])

    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False)

    criterion = DiceCELoss(include_background=False, softmax=True, to_onehot_y=True, lambda_dice=0.5,lambda_ce=0.5)

    model = UNet(3,config.classes,config.base_c,config.kernels,'batch')


    return model,train_loader,test_loader,criterion,eval_criterion
def model_pipeline(hyperparameters):
    with wandb.init(project="new baseline",config=hyperparameters):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion = make(config)
        # print(model)
        model = model.to(config.device)
        train(model,train_loader,test_loader,criterion,eval_criterion,config)

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify Parameters')

    parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
    parser.add_argument('b_s', metavar='b_s', type=int, help='Specify bach size')
    parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
    parser.add_argument('epochs', metavar='epochs', type=int, help='number of eoochs for training')
    parser.add_argument('-l1', nargs='+', type=int, metavar='kernels',
                        help='number of channels in unet layers')  # Use like: python test.py -l 1 2 3 4
    parser.add_argument('--base_c', metavar='--base_c', default=12, type=int,
                        help='base_channel which is the first output channel from first conv block')

    parser.add_argument('no_runs', metavar='no_runs', type=int,
                        help='how many random seeds you want to run experiment on')

    args = parser.parse_args()
    lr, batch_size, gpu_index, epochs,no_runs = args.lr, args.b_s, args.gpu_index,args.epochs, args.no_runs
    base_c = args.base_c
    kernels = args.l1
    print(torch.cuda.is_available())
    device = torch.device('cuda:0')
    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')

    for a in ['REFUGE']:
        for _ in range(no_runs):
            config = dict(epochs=epochs, classes=3, base_c = base_c, kernels=kernels,
                          batch_size=batch_size, learning_rate=lr, dataset=a,
                          seed=401,transform='baseline',device=device)
            config["seed"] = randint(1501,3000)
            seeding(config["seed"])


            data_save_path = f'/home/kebl6872/Desktop/new_data/{config["dataset"]}/test/lr_{lr}_bs_{batch_size}_fs_{config["base_c"]}_[{"_".join(str(k) for k in config["kernels"])}]/'

            create_dir(data_save_path + f'Checkpoint/seed/{config["seed"]}')
            checkpoint_path_lowloss = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_lowloss.pth'
            checkpoint_path_final = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_final.pth'
            checkpoint_path_lowloss = create_file(checkpoint_path_lowloss)
            checkpoint_path_final = create_file(checkpoint_path_final)
            config['low_loss_path']=checkpoint_path_lowloss
            config['final_path'] = checkpoint_path_final

            model = model_pipeline(config)



