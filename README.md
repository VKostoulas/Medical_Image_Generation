# Hyperparameter-Free Medical Image Generation with nnU-Net and Diffusion Models

### Description
Training and sampling with 2D or 3D image generation models on your dataset
has never been that easy! Simply, create your dataset, and that's it!
Go train your model! Don't worry about GPU memory! (these 3D models
can take a colossal amount of memory) With techniques like mixed precision,
activation checkpointing, and gradient accumulation you will fit your model in
your GPU!

This system is heavily based on nnU-Net. Given a dataset, nnU-Net automatically
defines all the hyperparamaters that should be used for this dataset. We simply
transfer these hyperparameters to the task of training diffusion models
for medical image generation. nnU-Net inferred parameters also work for the image
generation task, and this project provides you with a natural way to enhance
your nnU-Net based segmentation models: generate images (and labels) and use these
to enhance your segmentation model.

### Requirements
- python 3.9.17, cuda 11.8, and at least one GPU with 8GB memory

### Installation

If you want to use this project as a python library:

- Install [pytorch](https://pytorch.org/get-started/locally/) 
- Install [nnunetv2](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
- pip install medimgen

If you want to further develop this project:
1. clone the repository
2. create your virtual environment
3. install [pytorch](https://pytorch.org/get-started/locally/)
4. install [nnunetv2](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md)
3. in the environment run: pip install -e .

[//]: # (- If pip doesn't work:)

[//]: # (  - Clone the repository )

[//]: # (  - You can try installing the requirements.txt, but if this doesn't work:)

[//]: # ()
[//]: # (    - Install pytorch following the official [pytorch )

[//]: # (    installation guide]&#40;https://pytorch.org/get-started/locally/&#41;.)

[//]: # ()
[//]: # (    - Install the following libraries with pip:)

[//]: # (      - pip install pyyaml matplotlib tqdm nibabel scikit-image monai )

[//]: # (      monai-generative nnunet lpips xformers torchinfo)

[//]: # ()
[//]: # (  - &#40;Optional&#41; You can install these libraries also for jupyter notebooks and)

[//]: # (  interactive visualization:)

[//]: # (    - pip install jupyter matplotlib ipywidgets ipympl notebook tornado)

[//]: # (  - run pip install -e . when you are in the main directory)
 

## Usage Instructions

### Environment variables

First, according to nnU-Net, you must set the nnU-Net related environment variables.
Additionally, you will need to set 1 more variable for this project. In total:

```bash
export nnUNet_raw="/path_to_your_folder/nnUNet_raw"
export nnUNet_preprocessed="/path_to_your_folder/nnUNet_preprocessed"
export nnUNet_results="/path_to_your_folder/nnUNet_results"
export medimgen_results="/path_to_your_folder/medimgen_results"
```
For an explanation of the nnU-Net related variables see nnU-Net documentation. 
medimgen_results is the path to the folder where all the image generation results 
will be saved.


### Dataset preparation
You must create a dataset based on nnU-Net conventions. You can start with 
a dataset which follows the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) format, where
all training images are contained in a folder called **imagesTr**, and are compressed 
nifti files (.nii.gz), and convert the dataset to nnUNetv2 format with:

```bash
nnUNetv2_convert_MSD_dataset -i /path_to_original_dataset/Task01_MyDataset
```

This should create a dataset in the nnUNet_raw folder called Dataset001_MyDataset, 
splitting multiple channel images to separate images. For other available dataset 
format options take a look at nnUNet documentation.

### Dataset preprocessing

Given a new dataset, nnU-Net will extract a dataset fingerprint (a set of 
dataset-specific properties such as image sizes, voxel spacings, intensity 
information etc). This information is used to design three U-Net configurations. 
Each of these pipelines operates on its own preprocessed version of the dataset.

The easiest way to run fingerprint extraction, experiment planning and 
preprocessing is to use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

This will create a new subfolder in your nnUNet_preprocessed folder named after the 
dataset. All the images will be cropped to non-zero regions, resampled to the median voxel 
spacing of the dataset, and depending on the image modality other processes should 
be applied (e.g., for MRI images a z-score normalization will be used). For more 
information check nnU-Net documentation.


### Configuration

After preprocessing the dataset with nnUNet, a file called nnUNetPlans.json is
created in the corresponding dataset folder in nnUNet_preprocessed folder. This 
file contains all the hyperparameters derived from nnU-Net, and is used to select
the hyperparameters for image generation. To create a configuration file for
image generation based on nnUNet, run:

```bash
medimgen_plan -d DATASET_ID
```

This will create a file called medimgen_config.yaml in the corresponding dataset
folder in nnUNet_preprocessed folder, containing all the configured parameters that
will be used for image generation. You can also easily modify the configuration file in
case you want to experiment or if you don't like something (e.g., change the 
'val_plot_interval' argument depending on how often you want to get an image plot 
while training). In addition, we have also set some heuristics to derive
the values for some additional hyperparameters that are not involved in nnUNet 
(e.g., loss weights for autoencoder losses).

### Training

#### Denoising Diffusion Probabilistic Model

All the training commands include these arguments:
- DATASET_ID: corresponds to the numeric dataset identifier (e.g., 001 for Brain Tumour
dataset from Medical Segmentation Decathlon)
- SPLITTING: should be one of ['train-val-test', '5-fold'], defining the type of data 
splitting 
- MODEL_TYPE: should be either '2d' or '3d' 
- -p: indicates that we want to see progress bars.

Here is an example to train a Denoising Diffusion Probabilistic Model:

```bash
medimgen_train_ddpm DATASET_ID SPLITTING MODEL_TYPE -p
```

#### Latent Diffusion Model
To train a Latent Diffusion Model, first we need to train an autoencoder:

```bash
medimgen_train_autoencoder DATASET_ID SPLITTING MODEL_TYPE -p
```
After finishing training, we can then train the Latent Diffusion Model:

```bash
medimgen_train_ldm DATASET_ID SPLITTING MODEL_TYPE -p
```

#### Output Files
Running an experiment will create a directory in medimgen_results path with the 
following folders and files: 
- checkpoints folder: contains the checkpoints of the last and the best epoch of 
training (the best one is derived based on the validation reconstruction loss)
- plots folder: in this folder a loss.png file will be saved for every kind of
training, showing the training and validation losses per epoch. Moreover, a gif will 
be saved in this folder (with frequency based on the argument 'val_plot_interval' 
in the config file) in case of 3D training, and a slice in case of 2D. For ddpm 
and ldm, the figures contain generated examples across the z direction, 
while for the autoencoder, an actual image and its reconstruction are visualized.
- loss_dict.pkl: a file containing lists with loss values per epoch
- log.txt (optional): if you have set output_mode to be 'log' then also a log file 
will be created instead of printing everything on screen

#### Tips for Training

- To continue training a model, just pass -c when running the training command.
- The autoencoder shouldn't have more than 2-3 downsampling layers, otherwise it 
won't be able to reconstruct details accurately.
- Only a few convolutional filters for every layer of the autoencoder (e.g., 32), 
can result in good enough reconstruction performance. 
- Loss weights in the training of the autoencoder are really important. Some works
might use relatively small loss weights for the perceptual loss (e.g., 0.01) and the
adversarial loss (e.g., 0.1), but based on experiments a value of 1, and 0.25, 
respectively, gives much better and realistic results.

### Sampling

To sample with your favorite diffusion model run (obviously, after you have trained
the corresponding models):

```bash
medimgen_sample DATASET_ID MODEL_TYPE NUM_IMAGES SAVE_PATH -p
```
This will sample NUM_IMAGES images and save them in SAVE_PATH


## ToDos

1. Pass nnUNet configured parameters to medimgen
   1. create code that reads nnunet file and creates a medimgen config file
   2. adapt autoencoder + diffusion for flexible architectures
2. create code to select the additional hyperparameters not involved in nnUNet
3. Adapt dataset class for 2D and 3D training, nnUNet augmentations, and ideally 
nnUNet patch selection when training (oversampling).

- Add intensity normalization?
- Add option to include labels, so that we can train a model to generate labels 
together with images
- Add efficient implementation of U-Net like in Medical Diffusion
- Add GANs
- Ultimate Goal: like nnU-Net, study and come up with heuristics that can be applied
to multiple datasets and achieve high quality generation. Come up with ways to 
automatically configure every experiment's hyperparameters.

- Experimental: nnUNet works with random cropped patches instead of full images.
Wouldn't that be awesome to do also in image generation? This would reduce 
computational demands and also increase the training dataset size. We can train
the autoencoder to output cropped patches and then do sliding window inference, but
how to perform generation from multiple patches, so multiple latent vectors? IDEA: 
Train the diffusion model to generate latent vectors based on image patches, 
conditioned on latent vectors from image patches around the main patch. On inference,
start generation with the top left patch conditioned on patches of zeros, and then
generate patches sequentially based on previously generated patches.