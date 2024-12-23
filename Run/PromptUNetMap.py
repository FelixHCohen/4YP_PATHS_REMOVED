from PromptUNetTest import *
from PromptUNet.PromptUNet import PromptUNetAttnMap

def mapMake(config,path,device):

    model = PromptUNetAttnMap(device,3,3,12,[6,12,24,48],[3,4,4,4],64,4,8,)
    checkpoint = torch.load(path,map_location=device)

    model.load_state_dict(checkpoint,strict=False)
    #model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model = model.to(device)
    model.eval()

    if config['testset'] == 'GAMMA':
        train = get_data(train=True, return_path=False, gamma=True, transform=False, )
        test = get_data(train=False, return_path=False, gamma=True, transform=False, )
        if config['dataset'] != 'GAMMA':
            total = torch.utils.data.ConcatDataset([train, test])
        else:
            total = test
    elif config["testset"]== "MESSIDOR":
        total = get_data(train=True,return_path=True,messidor=True,transform=False,)
    elif config["testset"] == "GS1":
        train = get_data(train=True, return_path=True, gs1=True, transform=False, )
        test = get_data(train=False, return_path=True, gs1=True, transform=False, )
        if config['dataset'] != 'GS1':
            #total = train
            total = torch.utils.data.ConcatDataset([train, test])
        else:

            total = test
    elif config['testset'] == 'G1020':
        total = get_data(train=True,return_path=True,transform=False,g1020=True)
    elif config['testset'] == 'PAPILA':
        total = get_data(train=True,return_path=False,transform=False,papila=True)
    elif config['testset'] == 'ORIGA':
        total = get_data(train=True, return_path=False, transform=False, origa=True)
    elif config['testset'] == 'BR':
        total = get_data(train=True, return_path=False, transform=False, br=True)
    elif config['testset'] == 'MAGRABIA':
        total = get_data(train=True, return_path=False, transform=False, magrabia=True)

    else:
        total = get_data(train=False,refuge_test=False, return_path=True, transform=False, )

    train_loader = DataLoader(dataset=total, batch_size=1, shuffle=True, )

    criterion = f1_valid_score

    return model, train_loader,criterion

if __name__ == '__main__':
    device = torch.device('cpu')

    with (wandb.init(project="ARC-TESTING", config={}, dir='/data/engs-mlmi1/kebl6872/wandb')):
        for testset in ['MESSIDOR']:
            print(testset)
            print('-----')
            model_str = "/home/kebl6872/Desktop/new_data/REFUGE/test/lr_0.0003_bs_8_fs_12_[6_12_24_48]/Checkpoint/seed/2525/lr_0.0003_bs_8_lowloss.pth"

            config = dict(epochs=1000, classes=3, base_c = 12, kernels=[6,12,24,48],attention_kernels = [3,4,4,4],d_model=64,
                              batch_size=1, dataset="REFUGE_VAL_TEST",
                              testset=testset,seed=401,transform=True,device=device,batch_norm=False)
            model, loader, criterion = mapMake(config,model_str,device)
            num_points = 10

            attn_maps = {'image path': [], 'mask path': [], 'user points': [], 'user points labels': [],
                         'b prompt to image layer 0': [],
                         'b prompt to image layer 1': [],
                         'b prompt to image layer 2': [],
                         'b prompt to image layer 3': [],

                         'b image to prompt layer 0': [],
                         'b image to prompt layer 1': [],
                         'b image to prompt layer 2': [],
                         'b image to prompt layer 3': [],

                         'd1 prompt to image layer 0': [],
                         'd1 prompt to image layer 1': [],
                         'd1 prompt to image layer 2': [],
                         'd1 prompt to image layer 3': [],

                         'd1 image to prompt layer 0': [],
                         'd1 image to prompt layer 1': [],
                         'd1 image to prompt layer 2': [],
                         'd1 image to prompt layer 3': [],

                         'd2 prompt to image layer 0': [],
                         'd2 prompt to image layer 1': [],
                         'd2 prompt to image layer 2': [],
                         'd2 prompt to image layer 3': [],

                         'd2 image to prompt layer 0': [],
                         'd2 image to prompt layer 1': [],
                         'd2 image to prompt layer 2': [],
                         'd2 image to prompt layer 3': [],
                         'val score': []}

            with torch.no_grad():

                val_scores = np.zeros(num_points)
                f1_score_record = np.zeros((4,num_points))
                total = 0
                for _,(images,labels,image_path,mask_path) in enumerate(loader):
                    images, labels = images.to(device, dtype=torch.float32), labels.to(device)
                    prev_output = torch.from_numpy(np.zeros((1,1,512,512))).to(device)# need to make first input appear to havec come from model so it works w gen_points func
                    points, point_labels = generate_points_batch(labels,prev_output ,num=1)
                    #prev_output = prev_output.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
                    # above line was for when prev_output came out of model
                    retain_point = random.randint(0,9)

                    for i in range(num_points):

                        point_input, point_label_input = torch.from_numpy(points).to(device,dtype=torch.float32), torch.from_numpy(point_labels).to(device, dtype=torch.float32)
                        ##print(point_input.shape)
                        if i == retain_point:
                            attn_maps['image path'].append(image_path)
                            attn_maps['mask path'].append(mask_path)
                            attn_maps['user points'].append(points.tolist())
                            attn_maps['user points labels'].append(point_labels.tolist())
                            outputs = model(images, point_input, point_label_input,attn_maps=attn_maps,train_attention=True)
                        else:
                            outputs = model(images, point_input, point_label_input, attn_maps=False, train_attention=True)


                        new_points, new_point_labels = generate_points_batch(labels, outputs)
                        outputs = outputs.softmax(dim=1).argmax(dim=1).unsqueeze(dim=1)
                        score = criterion(outputs,labels)


                        val_scores[i] += score[1].item() / 2 + score[2].item() / 2

                        if i == retain_point:
                            attn_maps['val score'].append(score[1].item() / 2 + score[2].item() / 2)

                        f1_score_record[:,i] += score
                        plot =False
                        if plot:

                            image_idx = random.randint(0, images.shape[0] - 1)
                            point_tuples = [(i, j, val[0]) for (i, j), val in
                                            zip(points[image_idx, :, :], point_labels[image_idx, :, :])]
                            print(f'point_tuples: {point_tuples}')
                            plot_output_and_prev(prev_output[image_idx,:,:,:].unsqueeze(dim=0).detach(),outputs, images, labels, score,
                                                 point_tuples,
                                                 detach=False, image_idx=image_idx)
                        prev_output = outputs
                        point_labels = np.concatenate([point_labels, new_point_labels], axis=1)
                        points = np.concatenate([points, new_points], axis=1)

                    total += labels.size(0)
                    if total > 10:
                        break


            df = pd.DataFrame(attn_maps)
            df.to_csv('/home/kebl6872/Desktop/messidor_single_head_attn_maps.csv')
            f1_score_record /= total
            val_scores /= total
            val_score_str = ', '.join([format(score,'.8f') for score in val_scores])
            disc_scores = ', '.join([format(score,'.8f') for score in f1_score_record[3,:]])
            cup_scores = ', '.join([format(score,'.8f') for score in f1_score_record[2, :]])

            return_str = f"model tested on {total} images\nval_scores: {val_score_str}\ndisc f1 scores {disc_scores}\ncup scores: {cup_scores}"
            print(return_str)