import time

import cv2
from train_neat import *
from PromptUNet.PromptUNet import PromptUNet,pointLoss,NormalisedFocalLoss,combine_loss,combine_point_loss,MaskedPromptUNet
from utils import *
import glob
from monai.losses import DiceLoss


"""function that logs training loss to wandb"""
def prompt_train_log(loss,example_ct,epoch):
    wandb.log({"epoch": epoch,"attention training loss":loss},step=example_ct)
    print(f"Loss after {str(example_ct + 1).zfill(5)} batches: {loss:.3f}")


"""function that inputs a batch of images into the model and updates weights"""
def prompt_train_batch(images,labels,points,point_labels,weak_unet_preds,model,optimizer,criterion,config,plot=False):

    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float), torch.from_numpy(point_labels).to(device, dtype=torch.float)


    outputs = model(images, point_input, point_label_input, train_attention=True)


    loss = criterion(outputs, labels,point_input,point_label_input,config.device)


    if plot:
        image_idx = random.randint(0,images.shape[0]-1)
        point_tuples = [(i, j, val[0]) for (i, j), val in zip(points[image_idx, :, :], point_labels[image_idx, :, :])]
        plot_output_and_prev(weak_unet_preds[image_idx],
                             outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1), images, labels, loss, point_tuples,
                             detach=False,image_idx=image_idx)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

"""applies model to validation sets and logs results to wandb"""
def prompt_test(model,test_loader,criterion,config,best_valid_score,example_ct,optimizer,num_points=6,plot=False):
    model.eval()

    with torch.no_grad():

        val_scores = np.zeros(num_points)
        f1_score_record = np.zeros((4,num_points))
        total = 0
        for _,(images,labels) in enumerate(test_loader):
            images, labels = images.to(device, dtype=torch.float32), labels.to(device)
            """generate first point for each image using 'blank prediction' of 0s everywhere"""
            if config.box:
                points,point_labels,_ = gen_boxes(images,labels,model,config.device,weak_unet=False,num_points=0)
            else:
                prev_output = torch.from_numpy(np.zeros((config.batch_size,1,512,512))).to(config.device)# need to make first input appear to havec come from model so it works w gen_points func
                points, point_labels = generate_points_batch(labels,prev_output ,num=1)

            for i in range(num_points):

                point_input, point_label_input = torch.from_numpy(points).to(device,dtype=torch.float32), torch.from_numpy(point_labels).to(device, dtype=torch.float32)

                outputs = model(images, point_input, point_label_input,train_attention=True)

                "we now generate points using the current output of the model"
                new_points, new_point_labels = generate_points_batch(labels, outputs)

                outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
                score = criterion(outputs,labels)


                val_score = score[1].item() / 2 + score[2].item() / 2


                val_scores[i] += val_score


                f1_score_record[:,i] += score

                plot = False
                if plot:

                    image_idx = random.randint(0, images.shape[0] - 1)
                    point_tuples = [(i, j, val[0]) for (i, j), val in
                                    zip(points[image_idx, :, :], point_labels[image_idx, :, :])]
                    print(f'point_tuples: {point_tuples}')
                    plot_output_and_prev(prev_output[image_idx,:,:,:].unsqueeze(dim=0).detach(),
                                         outputs, images, labels, score,
                                         point_tuples,
                                         detach=False, image_idx=image_idx)
                prev_output = outputs

                "concatenate new points to our current set of pointsand re-input into model"
                point_labels = np.concatenate([point_labels, new_point_labels], axis=1)
                points = np.concatenate([points, new_points], axis=1)

            total += labels.size(0)

    f1_score_record /= total
    val_scores /= total
    val_score_str = ', '.join([format(score,'.8f') for score in val_scores])
    disc_scores = ', '.join([format(score,'.8f') for score in f1_score_record[3,:]])
    cup_scores = ', '.join([format(score,'.8f') for score in f1_score_record[2, :]])

    return_str = f"model tested on {total} images\nval_scores: {val_score_str}\ndisc f1 scores {disc_scores}\ncup scores: {cup_scores}"


    data_to_log = {}




    # Loop through the validation scores and add them to the dictionary
    for i, val_score in enumerate(val_scores):
        data_to_log[f"val_score {i + 1} points"] = val_score
        data_to_log[f"Validation Background F1 Score {i + 1}"] = f1_score_record[0][i]
        data_to_log[f"Validation Disc F1 Score {i + 1}"] = f1_score_record[3][i]
        data_to_log[f"Validation Cup F1 Score {i + 1}"] = f1_score_record[2][i]
        data_to_log[f"Validation Outer Ring F1 Score {i + 1}"] = f1_score_record[1][i]

    wandb.log(data_to_log,step=example_ct)
    model.train()

    if val_scores[-1] > best_valid_score[0]:
        data_str = f"Valid score for point {len(val_scores)} improved from {best_valid_score[0]:2.8f} to {val_scores[-1]:2.8f}. Saving checkpoint: {config.low_loss_path}"
        print(data_str)
        best_valid_score[0] = val_scores[-1]
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, config.low_loss_path)

    return return_str


"""training script for interactive segmentation model"""
def prompt_train(model, loader,test_loader, criterion, eval_criterion, config,weak_unet_epochs=4,num_points=10):


    wandb.watch(model,criterion,log='all',log_freq=40) #this is freq of gradient recordings

    example_ct = 0
    batch_ct = 0
    optimizer = torch.optim.Adam(model.parameters(), lr)

    """ learning rate scheduler initialisation """
    def lr_func(step):
        if step < 45000:
            return 1.0
        elif step < 55000:
            return 0.5
        elif step < 65000:
            return 0.25
        elif step < 75000:
            return 0.1
        else:
            return 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)


    best_valid_score = [0.0]#in list so I can alter it in test function
    for epoch in tqdm(range(config.epochs)):

        avg_epoch_loss = 0.0
        start_time = time.time()


        if epoch >= weak_unet_epochs:
            weak_unet = False
        else:
            weak_unet = True

        for _,(images, labels) in enumerate(loader):

            #start = time.perf_counter()
            """randomly generate points for this batch of images"""
            if config.box:
                points, point_labels, weak_unet_preds = gen_boxes(images, labels, model,config.device, weak_unet, 6)
            else:
                points, point_labels, weak_unet_preds = gen_points_from_weak_unet_batch(images, labels, model,config.device,weak_unet,num_points)


            """change to true for debugging purposes"""
            plot = False



            loss = prompt_train_batch(images,labels,points,point_labels,weak_unet_preds,model,optimizer,criterion,config,plot)
            scheduler.step()

            avg_epoch_loss += loss
            example_ct += len(images)
            batch_ct +=1
            #print(f'sigma: {model.pe_layer.sigma.item()}')

            if ((batch_ct+1)%20)==0:
                prompt_train_log(loss,batch_ct,epoch)



        end_time = time.time()
        iteration_mins, iteration_secs = train_time(start_time, end_time)
        print(f'train time: {iteration_mins}m {iteration_secs}s')
        if (epoch)%3 == 0 and epoch >= 3:
            test_results = prompt_test(model,test_loader,eval_criterion,config,best_valid_score,batch_ct,optimizer)
            avg_epoch_loss/=len(loader)
            test_end_time = time.time()

            test_mins,test_secs = train_time(end_time,test_end_time)
            data_str = f'Epoch: {epoch + 1:02} | Iteration Time: {iteration_mins}min {iteration_secs}s Test Time: {test_mins}min {test_secs}s\n'
            data_str += f'\tTrain Loss: {avg_epoch_loss:.8f}\n'
            data_str += test_results
            print(data_str)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, config.final_path)






"""plotting function - was used to determine points were being placed in correct positions"""
def plot_output_and_prev(output,next_output, image, label, score, point_tuples,image_idx=0, detach=False):
        color_map = {0: 'red', 1: 'green', 2: 'blue'}
        if output.shape[1]==3:
            output = output.softmax(dim=1).argmax(dim=1).unsqueeze(1)
        image_np = image[image_idx, :, :, :].cpu().numpy().transpose(1, 2, 0)
        image_np = ((image_np * 127.5) + 127.5).astype(np.uint16)
        if not detach:
            output = output[0, :, :, :].cpu().numpy().transpose(1, 2, 0)
            next_output = next_output[image_idx, :, :, :].cpu().numpy().transpose(1, 2, 0)
        else:
            output = output[0, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
            next_output = next_output[image_idx, :, :, :].detach().cpu().numpy().transpose(1, 2, 0)
        label = label[image_idx, :, :, :].cpu().numpy().transpose(1, 2, 0)
        label = np.repeat(label, 3, 2)
        output = np.repeat(output, 3, 2)
        next_output = np.repeat(next_output, 3, 2)
        d = {0: 0, 1: 128, 2: 255}
        vfunc = np.vectorize(lambda x: d[x])

        # Apply the vectorized function to the array
        label = vfunc(label)
        output = vfunc(output)
        next_output = vfunc(next_output)
        rows = 1
        cols = 4

        # Create a figure with the specified size
        plt.figure(figsize=(25, 25))
        plt.subplot(rows, cols, 1)
        plt.imshow(np.flip(image_np,2))
        plt.title("image")
        plt.axis('off')
        # Plot the mask in the even-numbered subplot
        plt.subplot(rows, cols, 2)
        plt.imshow(output)
        plt.title(f"prev_output")
        for y, x, val in point_tuples:  # point tuples stored in index e.g. ij = y,x
            #print(f'({x},{y}): {val}')
            circle = plt.Circle((x, y), 1, color=color_map[val])
            plt.gca().add_patch(circle)
        plt.axis('off')
        plt.subplot(rows,cols,3)
        plt.imshow(next_output,)
        plt.title(f"current output {score}")
        for y, x, val in point_tuples:  # point tuples stored in index e.g. ij = y,x
            #print(f'({x},{y}): {val}')
            circle = plt.Circle((x, y), 1, color=color_map[val])
            plt.gca().add_patch(circle)
        plt.axis("off")
        plt.subplot(rows, cols, 4)
        plt.imshow(label, )
        plt.title("ground truth")
        for y, x, val in point_tuples:  # point tuples stored in index e.g. ij = y,x
            #print(f'({x},{y}): {val}')
            circle = plt.Circle((x, y), 1, color=color_map[val])
            plt.gca().add_patch(circle)
        plt.axis('off')

        # Show the plot
        plt.show()

"""creates a standard UNet model initialised from 'model_path' weights - used to generate points early on in training"""
def make_weakUNet(model_path):
    model = UNet(3, 3, 12, [6, 12, 24, 48], 'batch')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint,strict=False)
    model.eval()
    return model


"""point generation scheme for a batch of images, num_points determines the maximum amount of points that can be created per image during training, 'weak_unet' boolean determines whether the models outputs are being compared to the ground truth to create points or just a standard unets outputs are"""
def gen_points_from_weak_unet_batch(images,y_true,model,device,weak_unet,num_points):

    num_points = random.randint(1,num_points)
    B = y_true.shape[0]

    points = np.zeros((B,num_points,2))
    point_labels = np.zeros((B,num_points,1))
    weak_unet_preds = list()
    for i in range(B):
        y_true_input = y_true[i, :, :, :]
        y_true_input = y_true_input[np.newaxis, :, :, :]  # need to add pseudo batch dimension to work w generate_points (looking back on this prob could have just vectorized 'gen_points_from_weak_unet')
        image_input = images[i,:,:,:]
        image_input = image_input[np.newaxis,:,:,:]
        gen_points,weak_unet_pred = gen_points_from_weak_unet(y_true_input,image_input,device,model,num_points,weak_unet)
        weak_unet_preds.append(weak_unet_pred)
        for j in range(num_points):

            points[i,j,0] = gen_points[j][0]
            points[i,j,1] = gen_points[j][1]
            point_labels[i,j,0] = gen_points[j][2]

    return points,point_labels,weak_unet_preds

"""point generation scheme for single image"""
def gen_points_from_weak_unet(y_true, image,device,model, num_points,weak_unet=False, detach=False):

    """Point generation scheme, weak unet boolean determines whether points are generated from the model itself (weak_unet=False) or a unet"""
    if not weak_unet:
        prev_output = torch.from_numpy(np.zeros((1, 1, image.shape[2],image.shape[3]))).to(device)  # need to make first input appear to have come from model so it works w gen_points func
        points, point_labels = generate_points_batch(y_true, prev_output, num=1)
        point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float), torch.from_numpy(
            point_labels).to(device, dtype=torch.float)
        image = image.to(device, dtype=torch.float32)
        weak_unet_pred = model(image,point_input,point_label_input)

    else:
        model_paths = glob.glob(f'/home/kebl6872/Desktop/weakunet/Checkpoint/seed/**/*lowloss.pth', recursive=True) #place path to baseline UNETS here
        index = random.randint(0, len(model_paths) - 1)

        weakUNet = make_weakUNet(model_paths[index])
        weakUNet = weakUNet.to(device)
        image = image.to(device,dtype=torch.float32)

        weak_unet_pred = weakUNet(image)

    """turn model outputs into hard predictions"""

    weak_unet_pred = weak_unet_pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    weak_unet_pred_o = torch.clone(weak_unet_pred)
    y_true = y_true.cpu().numpy().astype(int)
    weak_unet_pred = weak_unet_pred.cpu().numpy().astype(int)

    dc_misclass = np.argwhere(np.logical_and(y_true == 1, weak_unet_pred == 2) == True)[:,2:]  # y_true indices are like [0,0,512,512]
    cd_misclass = np.argwhere(np.logical_and(y_true==2,weak_unet_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    db_misclass = np.argwhere(np.logical_and(y_true==1,weak_unet_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    cb_misclass = np.argwhere(np.logical_and(y_true==2,weak_unet_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bd_misclass = np.argwhere(np.logical_and(y_true==0,weak_unet_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bc_misclass = np.argwhere(np.logical_and(y_true==0,weak_unet_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]

    combined_results = list()

    misclass_register = {0:[bd_misclass,bc_misclass],1:[db_misclass,dc_misclass],2:[cb_misclass,cd_misclass]}
    """in order to ensure large error components are attributed more than one point, if cup errors are larger than 80 pixels, we assign more points and 150 for disc or background errors (cup is smaller than disc therefore min area val is smaller)"""
    for i in range(len(misclass_register)):
        if i ==2:
            min_area = 80
        else:
            min_area = 150
        """this function takes a set of misclassification coordinates, connects them into components, and returns points selected from each component sorted by component area"""
        combined_results.append(gen_components(misclass_register[i],i,min_area=min_area))

    type_indexes = [0,0,0]
    res = list()
    for _ in range(num_points):
        class_type = random.randint(0,2)

        """for cases where we have chosen to pick more points from a certain error type than what actually exist"""
        if type_indexes[class_type] >= len(combined_results[class_type]):

            correct_points = np.argwhere(y_true == class_type)[:,2:]
            l = list(range(correct_points.shape[0]))
            """for the very rare case when the transformation is so extreme, the cup/disc class has entirely dissapeared"""
            if not l:
                print(f'ERROR: no pixels of class {class_type}')
                i = random.randint(0, 511)
                j = random.randint(0, 511)
                c_val = y_true[0, 0, i, j].item()
                res.append((i, j, c_val))
            else:
                """if this has not occurred simply pick a pixel location of that class from ground truth"""
                rand_point = correct_points[random.choice(l),:]
                res.append((rand_point[0],rand_point[1],class_type))

        else:
            res.append(combined_results[class_type][type_indexes[class_type]])
            type_indexes[class_type]+=1

    return res,weak_unet_pred_o


"""given a certain misclassification category (e.g. ground truth cup, model predicted disc) this groups the mistakes into components and returns point locations corresponding to each component and sorted by component area"""
def gen_components(indices_list,val,val_list=False,min_area=150):
    components = list()
    """val list functionality is defunct, now always creates a list of whatever number 'val' is"""
    if not val_list:
        val = [val for _ in range(len(indices_list))]


    for val_idx,indices in enumerate(indices_list):
        map = np.zeros((512,512)).astype(np.uint8)
        if len(indices)==0:
            continue
        """create error map for cv2 connected components"""
        for i in range(indices.shape[0]):
            map[indices[i,0],indices[i,1]] = 1

        # generate component map
        (totalLabels, label_map, stats, centroids) = cv2.connectedComponentsWithStats(map, 8, cv2.CV_32S)

        # zip together component stats and their labels
        for (stat, componentLabel) in zip(stats[1:], list(range(1, totalLabels))):

            point_i,point_j = pick_rand(label_map,componentLabel)
            components.append([np.array([point_i,point_j]),stat[cv2.CC_STAT_AREA],val[val_idx]])

            area = stat[cv2.CC_STAT_AREA]
            while area > min_area:
                point_i, point_j = pick_rand(label_map, componentLabel)
                area = area // 2
                components.append([np.array([point_i, point_j]), area, val[val_idx]])


    res = sorted(components,key = lambda x: x[1],reverse=True)
    """return list of tuples of point coordinates and associated point classes"""
    return [(x[0][0],x[0][1],x[2]) for x in res]


"""never got round to implementing this properly - was meant to allow for boxes and points but now I think just boxes would be better"""
def gen_boxes(images,y_true,model,device,weak_unet,num_points=6):

    num_points = random.randint(0,num_points)
    B = y_true.shape[0]

    points = np.zeros((B,num_points+4,2))
    point_labels = np.zeros((B,num_points+4,1))
    weak_unet_preds = list()
    for i in range(B):
        y_true_input = y_true[i, :, :, :]
        disc_box = torch.argwhere(y_true_input == 1)
        cup_box = torch.argwhere(y_true_input == 2)

        (c1,disc_left_i, disc_left_j), (c2,disc_right_i, disc_right_j) = torch.min(disc_box,0)[0],torch.max(disc_box,0)[0]
        (c1,cup_left_i, cup_left_j), (c2,cup_right_i, cup_right_j) =  torch.min(cup_box,0)[0],torch.max(cup_box,0)[0]

        points[i,0,0] = disc_left_i
        points[i,0,1] = disc_left_j
        points[i, 1, 0] = disc_right_i
        points[i, 1, 1] = disc_right_j
        points[i, 2, 0] = cup_left_i
        points[i, 2, 1] = cup_left_j
        points[i, 3, 0] = cup_right_i
        points[i, 3, 1] = cup_right_j

        point_labels[i,0,0] = 3
        point_labels[i, 1, 0] = 4
        point_labels[i, 2, 0] = 5
        point_labels[i, 3, 0] = 6

        y_true_input = y_true_input[np.newaxis, :, :, :]  # need to add pseudo batch dimension to work w generate_points
        image_input = images[i,:,:,:]
        image_input = image_input[np.newaxis,:,:,:]
        gen_points,weak_unet_pred = gen_box_from_weak_unet(y_true_input,image_input,device,model,num_points,points[i,:4,:][np.newaxis,:,:],point_labels[i,:4,:][np.newaxis,:,:],weak_unet,)
        weak_unet_preds.append(weak_unet_pred)
        for j in range(num_points):

            points[i,j+4,0] = gen_points[j][0]
            points[i,j+4,1] = gen_points[j][1]
            point_labels[i,j+4,0] = gen_points[j][2]

    return points,point_labels,weak_unet_preds

def gen_box_from_weak_unet(y_true, image, device, model, num_points,points,point_labels,weak_unet=False, detach=False):

    if not weak_unet:
        point_input, point_label_input = torch.from_numpy(points).to(device, dtype=torch.float), torch.from_numpy(
            point_labels).to(device, dtype=torch.float)
        image = image.to(device, dtype=torch.float32)
        weak_unet_pred = model(image,point_input,point_label_input)

    else:
        model_paths = glob.glob(f'/home/kebl6872/Desktop/weakunet/Checkpoint/seed/**/*lowloss.pth', recursive=True)
        index = random.randint(0, len(model_paths)  - 1)
        weakUNet = make_weakUNet(model_paths[index])
        weakUNet = weakUNet.to(device)
        image = image.to(device,dtype=torch.float32)
        weak_unet_pred = weakUNet(image)

    weak_unet_pred = weak_unet_pred.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
    weak_unet_pred_o = torch.clone(weak_unet_pred)
    y_true = y_true.cpu().numpy().astype(int)
    weak_unet_pred = weak_unet_pred.cpu().numpy().astype(int)

    dc_misclass = np.argwhere(np.logical_and(y_true == 1, weak_unet_pred == 2) == True)[:,2:]  # y_true indices are like [0,0,512,512]
    cd_misclass = np.argwhere(np.logical_and(y_true==2,weak_unet_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    db_misclass = np.argwhere(np.logical_and(y_true==1,weak_unet_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    cb_misclass = np.argwhere(np.logical_and(y_true==2,weak_unet_pred==0)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bd_misclass = np.argwhere(np.logical_and(y_true==0,weak_unet_pred==1)==True)[:,2:] # y_true indices are like [0,0,512,512]
    bc_misclass = np.argwhere(np.logical_and(y_true==0,weak_unet_pred==2)==True)[:,2:] # y_true indices are like [0,0,512,512]

    combined_results = list()

    misclass_register = {0:[bd_misclass,bc_misclass],1:[db_misclass,dc_misclass],2:[cb_misclass,cd_misclass]}
    for i in range(len(misclass_register)):
        combined_results.append(gen_components(misclass_register[i],i))

    type_indexes = [0,0,0]
    res = list()
    for _ in range(num_points):
        class_type = random.randint(0,2)
        if type_indexes[class_type] >= len(combined_results[class_type]):

            correct_points = np.argwhere(y_true == class_type)[:,2:]
            l = list(range(correct_points.shape[0]))
            rand_point = correct_points[random.choice(l),:]
            res.append((rand_point[0],rand_point[1],class_type))

        else:
            res.append(combined_results[class_type][type_indexes[class_type]])
            type_indexes[class_type]+=1

    return res,weak_unet_pred_o


"""function that sets everything up to be input into the training function e.g. loss functions, models, datasets"""
def prompt_make(config):
    """initialise all training and validation sets"""
    train1,train2,train3 = get_data(train=False,transform=config.transform,),get_data(train=True,refuge_test=False,transform=config.transform),get_data(train=False,refuge_test=True,transform=config.transform),
    train = torch.utils.data.ConcatDataset([train1,train2])
    test1,test2 = get_data(train=True,gs1=True),get_data(train=True,gamma=True)
    test = torch.utils.data.ConcatDataset([test1,test2])

    """validation criterion - defined in utils"""
    eval_criterion = f1_valid_score

    train_loader = DataLoader(dataset=train,batch_size=config.batch_size,shuffle=True,)
    test_loader = DataLoader(dataset=test,batch_size=1,shuffle=False)

    """define training loss function - linear combination of Dice Loss, Focal Loss (used to be normalised focal loss but no longer the case) and 'Point Loss' """
    criterion1 = DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
    criterion = NormalisedFocalLoss()
    diceFocal = combine_loss(criterion,criterion1,0.7)
    criterion2 = pointLoss(radius=10)
    pointCriterion = combine_point_loss(criterion2,diceFocal,alpha = 200,beta=1)

    model = PromptUNet(config.device,in_c=3,out_c=3,base_c=config.base_c,kernels=config.kernels,attention_kernels=config.attention_kernels,dropout=0.1,box=config.box,use_mlp=config.use_mlp)

    return model,train_loader,test_loader,pointCriterion,eval_criterion
def prompt_model_pipeline(hyperparameters):
    with wandb.init(project="junk",config=hyperparameters):
        config = wandb.config

        model,train_loader,test_loader,criterion,eval_criterion = prompt_make(config)
        # print(model)
        model = model.to(device)
        prompt_train(model,train_loader,test_loader,criterion,eval_criterion,config,config.weak_unet_epochs,config.no_points)

    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Specify Parameters')

    parser.add_argument('lr', metavar='lr', type=float, help='Specify learning rate')
    parser.add_argument('b_s', metavar='b_s', type=int, help='Specify batch size')
    parser.add_argument('gpu_index', metavar='gpu_index', type=int, help='Specify which gpu to use')
    parser.add_argument('epochs', metavar='epochs',type=int, help='number of eoochs for training')
    parser.add_argument('d_model', type=int, metavar='d_model', help=' embedding dimension of attention layers')
    parser.add_argument('-l1', nargs='+', type=int, metavar='kernels', help='number of channels in unet encoder and decoder layers')# Use like: python test.py -l 1 2 3 4
    parser.add_argument('-l2', nargs='+', type=int, metavar='attention_kernels', help='number of layers in ImP Fusion blocks')
    parser.add_argument('--base_c', metavar='--base_c', default=12, type=int,
                        help='base_channel which is the first output channel from first conv block')



    args = parser.parse_args()
    lr, batch_size, gpu_index,epochs,d_model,= args.lr, args.b_s, args.gpu_index,args.epochs,args.d_model
    base_c = args.base_c
    kernels = args.l1
    attention_kernels = args.l2

    device = torch.device(f'cuda:{gpu_index}' if torch.cuda.is_available() else 'cpu')
    print(device)

    """login to wandb account"""
    wandb.login(key='d40240e5325e84662b34d8e473db0f5508c7d40e')


    """config is passed into most/all functions so they can use its info, the dataset key is redundant, but should be used to keep trakc of which datasets were used for training the model"""

    config = dict(epochs=epochs, base_c = base_c,d_model=d_model, kernels=kernels,attention_kernels=attention_kernels,
                  batch_size=batch_size, learning_rate=lr, dataset="refuge_trainval",
                  seed=401,transform='interactive',weak_unet_epochs=5,no_points=10,device=device,box=False,use_mlp=False,masked=False)
    config["seed"] = randint(1,3900)
    seeding(config["seed"])

    """creates save paths for both the final weights at end of training and the weights that maximised validation score"""

    data_save_path = f'/data_hd1/students/felix_cohen/Desktop/new_data/{config["dataset"]}/test/promptUNET_lr_{lr}_bs_{batch_size}_fs_{config["base_c"]}_[{"_".join(str(k) for k in config["kernels"])}]/'
    create_dir(data_save_path + f'Checkpoint/seed/{config["seed"]}')
    checkpoint_path_lowloss = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_lowloss.pth'
    checkpoint_path_final = data_save_path + f'Checkpoint/seed/{config["seed"]}/lr_{lr}_bs_{batch_size}_final.pth'
    check_point_path_lowloss = create_file(checkpoint_path_lowloss)
    checkpoint_path_final = create_file(checkpoint_path_final)
    config['low_loss_path']=checkpoint_path_lowloss
    config['final_path'] = checkpoint_path_final

    model = prompt_model_pipeline(config)


