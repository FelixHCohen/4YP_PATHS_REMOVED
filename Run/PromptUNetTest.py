from PromptUNetTrain import *
from PromptUNet.PromptUNet import SymmetricPromptUNet
import os
import pandas as pd
import monai
import math
from tqdm import tqdm

"""testing interactive segmentaiton model on OOD testsets, make sure config params are same as training, can cross-reference using wandb run overview"""
def make(config,path,device):

    model = PromptUNet(device,3,3,config['base_c'],config['kernels'],attention_kernels=config['attention_kernels'],d_model=config['d_model'],num_prompt_heads=4,num_heads=8,dropout=0.1,use_mlp=config['use_mlp'])
    checkpoint = torch.load(path,map_location=device)

    """NOTE: for some  checkpoints you may need to use checkpoint['model_state_dict'], recently changed how I save model weights so optimizer state also saved """
    model.load_state_dict(checkpoint,strict=False)
    #model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    """creates testset dependent on input key"""

    if config['testset'] == 'GAMMA':
        train = get_data(train=True, return_path=False, gamma=True, transform=False, )
        test = get_data(train=False, return_path=False, gamma=True, transform=False, )
        if config['dataset'] != 'GAMMA':
            total = torch.utils.data.ConcatDataset([train, test])
        else:
            total = test
    elif config["testset"]== "MESSIDOR":
        total = get_data(train=True,return_path=False,messidor=True,transform=False)
    elif config["testset"] == "GS1":
        train = get_data(train=True, return_path=False, gs1=True, transform=False)
        test = get_data(train=False, return_path=False, gs1=True, transform=False )
        if config['dataset'] != 'GS1':
            total = torch.utils.data.ConcatDataset([train, test])
        else:
            total = test

    elif config['testset'] == 'G1020':
        total = get_data(train=True,return_path=False,transform=False,g1020=True)
    elif config['testset'] == 'PAPILA':
        total = get_data(train=True,return_path=False,transform=False,papila=True)
    elif config['testset'] == 'ORIGA':
        total = get_data(train=True, return_path=False, transform=False, origa=True)
    elif config['testset'] == 'BR':
        total = get_data(train=True, return_path=False, transform=False, br=True)
    elif config['testset'] == 'MAGRABIA':
        total = get_data(train=True, return_path=False, transform=False, magrabia=True)

    else:
        """else use refuge set"""
        total = get_data(train=False,refuge_test=True, return_path=False, transform=False, )
        #t2 = get_data(train=False,refuge_test=False, return_path=False, transform=False, )
        # total = torch.utils.data.ConcatDataset([t1,t2])

    train_loader = DataLoader(dataset=total, batch_size=1, shuffle=True, )

    criterion = f1_valid_score

    return model, train_loader,criterion

def assd(y_pred,y):
    """assd score function"""
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


def test_like_training(model,loader,criterion,num_points,device):
    """this function isnt used - was to see if inputting points iteratively helped or not as when we train thy are not input iteratively"""
    with torch.no_grad():
        val_scores = np.zeros(num_points)
        f1_score_record = np.zeros((4,num_points))
        total = 0
        for _,(image,y_true) in enumerate(loader):
            prev_output = torch.from_numpy(np.zeros((1, 1, 512, 512))).to(device)  # need to make first input appear to havec come from model so it works w gen_points func
            points, point_labels = generate_points_batch(y_true, prev_output,num=1)  # generate first point on either cup or disc whichever is larger
            point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float), torch.from_numpy(point_labels).to(device, dtype=torch.float)
            image = image.to(device, dtype=torch.float32)
            weak_unet_pred = model(image, point_input,point_label_input).softmax(dim=1).argmax(dim=1).unsqueeze(dim=1) # use initial mistakes from prediction given this point


            y_true_copy = y_true.cpu().numpy().astype(int)
            weak_unet_pred = weak_unet_pred.cpu().numpy().astype(int)
            cup_misclass = np.argwhere(np.logical_and(y_true_copy == 2, weak_unet_pred != 2) == True)[:,2:]  # y_true indices are like [0,0,512,512]
            disc_misclass = np.argwhere(np.logical_and(y_true_copy == 1, weak_unet_pred != 1) == True)[:,2:]  # y_true indices are like [0,0,512,512]
            background_misclass = np.argwhere(np.logical_and(y_true_copy == 0, weak_unet_pred != 0) == True)[:,2:]  # y_true indices are like [0,0,512,512]
            misclass =  [background_misclass,disc_misclass,cup_misclass]
            misclass_val = [0,1,2]

            combined_misclass = gen_components(misclass,misclass_val,val_list=True,min_area=200)


            res = list()

            for i in [5,10,15,20]:

                if i >= len(combined_misclass):
                    class_type = random.randint(0,2)
                    correct_points = np.argwhere(y_true == class_type)[:, 2:]
                    l = list(range(correct_points.shape[0]))
                    rand_point = correct_points[random.choice(l), :]
                    res.append((rand_point[0], rand_point[1], class_type))

                else:
                    res.append(combined_misclass[i])

            points = np.zeros((num_points, 2))
            point_labels = np.zeros((num_points, 1))

            for j in range(len(res)):
                points[j, 0] = res[j][0]
                points[j, 1] = res[j][1]
                point_labels[j, 0] = res[j][2]

            point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float32), torch.from_numpy(point_labels).to(device, dtype=torch.float32)
            y_true = y_true.to(device)

            for i in range(len(res)):

                output = model(image, point_input[:i+1,:].unsqueeze(0), point_label_input[:i+1,:].unsqueeze(0), train_attention=True)
                score = criterion(output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1), y_true)
                val_scores[i] += score[1].item() / 2 + score[2].item() / 2
                f1_score_record[:, i] += score

            total += y_true.size(0)

    f1_score_record /= total
    val_scores /= total
    val_score_str = ', '.join([format(score, '.8f') for score in val_scores])
    disc_scores = ', '.join([format(score, '.8f') for score in f1_score_record[3, :]])
    cup_scores = ', '.join([format(score, '.8f') for score in f1_score_record[2, :]])

    return_str = f"model tested on {total} images\nval_scores: {val_score_str}\ndisc f1 scores {disc_scores}\ncup scores: {cup_scores}"
    print(return_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Specify Parameters')

    parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
    parser.add_argument('d_model', type=int, metavar='d_model', help='dimension of attention layers')
    parser.add_argument('-l1', nargs='+', type=int, metavar='kernels',
                        help='number of channels in unet layers')  # Use like: python test.py -l 1 2 3 4
    parser.add_argument('-l2', nargs='+', type=int, metavar='attention_kernels',
                        help='number of layers in cross attention blocks')
    parser.add_argument('--base_c', metavar='--base_c', default=12, type=int,
                        help='base_channel which is the first output channel from first conv block')

    args = parser.parse_args()
    gpu_index, d_model, = args.gpu_index,args.d_model
    base_c = args.base_c
    kernels = args.l1
    attention_kernels = args.l2
    device = torch.device(f'cuda:{gpu_index}')


#'GS1','BR','MAGRABIA','G1020','MESSIDOR','GAMMA','REFUGE'
    for testset in ['GAMMA','BR','G1020','GS1','MAGRABIA','MESSIDOR','ORIGA','REFUGE']:
        """note - dataset key not really needed, just indicates dataset the interactive model was trained on"""
        config = dict(base_c = base_c,d_model=d_model, kernels=kernels,attention_kernels=attention_kernels,
                          batch_size=1,dataset="REFUGE_train_val",
                          testset= testset,num_points=25,device=device,use_mlp=False)


        model_str = "/data_hd1/students/felix_cohen/Desktop/new_data/GAMMA/test/promptUNET_lr_0.0005_bs_8_fs_12_[6_12_24_48]/Checkpoint/seed/3846/lr_0.0005_bs_8_lowloss.pth"

        model, loader, criterion = make(config,model_str,device)
        num_points = config['num_points']

        """unet model initialisation so you can plot outputs of interactive model and 'baseline unet output' """
        unet_model_path = "/home/kebl6872/Desktop/new_data/REFUGE/test/lr_0.0003_bs_8_fs_12_[6_12_24_48]/Checkpoint/seed/2525/lr_0.0003_bs_8_lowloss.pth"

        unetmodel = UNet(3, 3, 12, [6, 12, 24, 48], 'batch').to(device)
        checkpoint = torch.load(unet_model_path)
        unetmodel.load_state_dict(checkpoint, strict=True)
        unetmodel.eval()



        with torch.no_grad():

            val_scores = np.zeros(num_points)
            f1_score_record = np.zeros((4,num_points))
            assd_score_record = np.zeros((2,num_points))
            total = 0
            error_count = 0
            for _,(images,labels) in enumerate(tqdm(loader)):
                images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                unet_output = unetmodel(images)
                #print(criterion(unet_output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1),labels))
                prev_output = torch.from_numpy(np.zeros((config['batch_size'],1,images.shape[2],images.shape[3]))).to(device)# need to make first input appear to have come from model so it works w gen_points func
                points, point_labels = generate_points_batch(labels,prev_output ,num=1)

                error = False
                plot=False
                for i in range(num_points):

                    point_input, point_label_input = torch.from_numpy(points).to(device,dtype=torch.float32), torch.from_numpy(point_labels).to(device, dtype=torch.float32)
                    outputs = model(images, point_input, point_label_input,train_attention=True)

                    new_points, new_point_labels = generate_points_batch(labels, outputs,1)



                    outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)

                    score = criterion(outputs,labels)
                    assd_score = assd(outputs,labels)


                    val_scores[i] += score[1].item() / 2 + score[2].item() / 2

                    f1_score_record[:,i] += score

                    # if i==0 and score[1].item()/2 + score[2].item()/2 < 0.83:
                    #     plot=True

                    if assd_score[0]==math.inf:
                        assd_score_record[0,i] += np.sqrt(2)*512
                        error = True
                    else:
                        assd_score_record[0, i] += assd_score[0]


                    if assd_score[1]==math.inf:
                        assd_score_record[1,i] += np.sqrt(2)*512
                        error = True
                    else:
                        assd_score_record[1, i] += assd_score[1]





                    plot=False
                    # #
                    # if _ <15 and i%3 == 0 and i < 12:
                    #     plot = True
                    # else:
                    #     plot=False
                    if plot:

                        image_idx = random.randint(0, images.shape[0] - 1)
                        image_idx = 0
                        point_tuples = [(i, j, val[0]) for (i, j), val in
                                        zip(points[image_idx, :, :], point_labels[image_idx, :, :])]


                        plot_output_and_prev(unet_output[image_idx,:,:,:].unsqueeze(dim=0).detach(),outputs, images, labels, score,
                                             point_tuples,
                                             detach=False, image_idx=image_idx)
                    prev_output = outputs
                    point_labels = np.concatenate([point_labels, new_point_labels], axis=1)
                    points = np.concatenate([points, new_points], axis=1)

                total += labels.size(0)
                if error == True:
                    error_count += 1
                if _%5==0:
                    print(f'curent avg: {val_scores[-1]/total}')


        print(f'total assd inf errors: {error_count}')
        f1_score_record /= total
        assd_score_record /= total
        val_scores /= total
        val_score_str = ', '.join([format(score,'.8f') for score in val_scores])
        disc_scores = ', '.join([format(score,'.8f') for score in f1_score_record[3,:]])
        cup_scores = ', '.join([format(score,'.8f') for score in f1_score_record[2, :]])
        outer_ring_scores = ', '.join([format(score,'.8f') for score in f1_score_record[1, :]])
        cup_assd_str = ', '.join([format(score,'.8f') for score in assd_score_record[1, :]])
        disc_assd_str = ', '.join([format(score, '.8f') for score in assd_score_record[0, :]])

        """create dataframe of results and save path for csv"""
        df = pd.DataFrame({f'{testset} disc scores': f1_score_record[3,:].tolist(),f'{testset} cup scores': f1_score_record[2,:].tolist(),f'{testset} outer ring scores':f1_score_record[1,:].tolist(),f'{testset} avg val scores': val_scores})
        df.to_csv(f'/home/kebl6872/Desktop/train2pp/{testset}_actually_trainval.csv')
        assd_df = pd.DataFrame({f'{testset} disc assd': assd_score_record[0,:].tolist(),f'{testset} cup assd':assd_score_record[1,:].tolist()})
        assd_df.to_csv(f'/home/kebl6872/Desktop/train2pp_assd/{testset}_assd_actually_trainval.csv')
        return_str = f"model tested on {total} images\nval_scores: {val_score_str}\ndisc f1 scores {disc_scores}\ncup scores: {cup_scores}\nouter ring scores: {outer_ring_scores}\ndisc assd: {disc_assd_str}\n cup assd: {cup_assd_str}"
        print(return_str)


