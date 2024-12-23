
## First steps:

Go to Run/train_neat.py

Change the paths in the get_data() function to where your datasets are stored, currently they are all stored in "/REDACTED/{dataset name}"

Then move to the file Run/PromptUNetTrain.py

The point generation function is initially very slow when the interactive model is producing poor outputs.\
To speed this up you can initially generate points using a standard UNET for the first couple of epochs. Go to the function: \
Gen_points_from_weak_unet and input a path leading to UNET models\
This speeds epoch training time from 15+min -> 3-5 min\
Training without this is possible and time taken per epoch will decrease as the model improves



## To train

in terminal run: 

python3 PromptUNetTrain.py {float: learning rate} {int: batch size} {int: GPU index} {int: epochs} {int: cross attention inner dimension} -l1 {list of ints: feature map multiplicative factor for each image encoder} -l2 {list of ints: number of cross attention layers for each decoder}

### Config setup:

In promptUNetTrain there is a dictionary called config that determines further hyperparameters. This is saved to wandb 


base_c - number of features first image encoder outputs - default 12 \
Learning_rate - learning rate - default 5e-4 \
Dataset - dataset you are training on, determines save_path \
Transform - determines dataset transform type, can either be ‘baseline’ or ‘interactive’ (see transforms in Run/data_aug/data.py file) \
Weak_unet_epochs - how many epochs to generate points with normal unet with - default 5 \
No_points - max number of user points to use during training - default 10 \
Box - whether to use box functionality - default False (haven’t implemented yet) \
Use_mlp - whether to use positional encoding scheme created by Li et al.  - default False \
Masked - whether to use masked cross attention - default False \
Note - not a hyperparameter but if you want to alter the model to take images of a different size it has an ‘img_size’ input set default to (512,512) - currently input image size has to be square and a power of 2


Example:


`Python3 PromptUNetTrain.py 5e-4 8 0 1000 64 -l1 6 12 24 48 -l2 3 4 4 4`

## Testing on OOD datasets
In __main__ portion of script, define the save_path for testing results and the path to the model you want to test. Make sure the config is identical to that used in training (this will have saved to the wandb project)

Config extra hyperparams: 

No_points - max number of user points to test up to \


e.g.

`Python3 PromptUNetTest.py 0 64 -l1 6 12 24 48 -l2 3 4 4 4 `


## Baseline UNet training
Go to Run/train_neat.py and replace data_save_path with where you want to save the model


Very similar set up to PromptUNetTrain:


`python3 train_neat.py {float: learning rate} {int: batch size} {int: GPU index} {int: epochs} -l1 {list of ints: feature map multiplicative factor for each image encoder} {int: number of random seeds you want to run}`

e.g. 


`Python3 train_neat.py 3e-4 8 0 60 -l1 6 12 24 48 1`

## In Distribution UNet Training
To train and validate standard unets on each OOD dataset run this command in terminal: 



`python3 in_distribution_train.py {float: learning rate} {int: batch size} {int: GPU index}  {int: number of random seeds you want to run} -l1 {list of ints: feature map multiplicative factor for each image encoder}`


e.g.

`python3 in_distribution_train.py 3e-4 8 3 -l1 6 12 24 48 `


## Datasets
To resize and pad/crop G1020 and the RIGA datasets (BinRushed,Magrabia,MESSIDOR), inserted a sorted list of image and mask paths alongside a save path into the function in the script: /REDACTED/Run/data_aug/dataset_initialisation.py


GS1 - use /REDACTED/Run/data_aug/GS1_initialisation.py


GAMMA - use /REDACTED/Run/data_aug/gamma_dataset_initialisation.py


ORIGA - use /REDACTED/Run/data_aug/origa_initialisation.py \
If you download the origa dataset from the kaggle link stated in the section below, make sure to 0 pad all numbers in the names e.g. ORIGA-1.png to ORIGA-0001.png otherwise the sorted list of images and masks wont 'match up'

REFUGE - use /REDACTED/Run/data_aug/refuge2_initialisation.py

Note - only REFUGE GAMMA and GS1 were preprocessed on this computer, these scripts can be immediately used as so (only need to change save path names):

access REFUGE_4YP/Run
```
python3 data_aug/gs1_initialisation.py

python3 data_aug/refuge2_initialisation.py

python3 data_aug/gamma_initialisation.py
```


Further notes:

Gamma train-val split: images 1-30 are used for validation

GS1 train-val split: move images 1-43 from testing set into training set, the remaining testing images are used for validation


When training the in distribution UNets, use the training and validation sets as they are named


In contrast, for the training of interactive model and baseline unet, the Gamma train and GS1 Train sets are combined to form a validation set. This validation set was used for all training set configurations (Refuge Train subset, Refuge Train + Val subsets, Entirety of Refuge)





### download instructions

GS1 can be downloaded here: https://www.kaggle.com/datasets/lokeshsaipureddi/drishtigs-retina-dataset-for-onh-segmentation


MESSIDOR,BinRushed and Magrabia images can be downloaded here: https://deepblue.lib.umich.edu/data/concern/data_sets/3b591905z \
Their masks can be downloaded here: https://github.com/mohaEs/RIGA-segmentation-masks \
Use hard segmentations (majority vote)\


REFUGE can be downloaded here: https://refuge.grand-challenge.org/Home2020/

G1020 can be downloaded here: https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets \
Masks can be created using this github repository: https://github.com/mohaEs/G1020-segmentation-mask-generator


GAMMA is tough to download but can be done via: https://aistudio.baidu.com/competition/detail/90/0/introduction


ORIGA is no longer publicly available from official sources but can be downloaded from: \
https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset \
( The ORIGA masks are not binary ( I believe due to this authors method of downsizing) , a function that corrects this is available in data_aug/origa_initialisation, this uses nearest-neighbours type algo to decide what is a positive pixel although in hindsight I should have just set all pixels > 0.5 to 1)

## Attention Maps


Load a PromptUNetMap class using weights from a previously trained interactive model (you have to set strict = False) \
Write save_path for map CSV in PromptUNetMap.py \
Run PromptUNetMap.py \
Write string path to map CSV into attn_maps.py and run to plot output \
Will plot attention map of a single head in each layer \

Note - currently only works with cpu as device for an unknown reason - luckily doesnt take too long to get outputs for ~30 images 

## Extra Notes

The current model_path in PromptUNetTest.py is to a model set up so each ImP Fusion block just accepts the prompt output of the PromptENcoder each time rather than the altered prompt output of the previous ImP Fusion block \
I never determined which method was superior, they give very similar results. Comments in the PromptUNet.py file show how to alter the code to switch between these two methods











