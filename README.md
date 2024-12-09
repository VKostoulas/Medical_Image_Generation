# 2D and 3D Medical Image Generation with Diffusion models and GANs

## Description
Training and sampling with 2D or 3D image generation models on your dataset
has never been that easy! Simply, create your dataset, and that's it!
Go train your model! Assuming that you have enough GPU memory (these 3D models
can take a colossal amount of memory) and you know exactly your architecture... Or
enjoy hyperparameter tuning!

## Installation

- python 3.9.17, cuda 11.8, and at least one faaat GPU
- pip install medimgen


[//]: # (- If pip doesn't work:)

[//]: # (  - Clone the repository )

[//]: # (  - You can try installing the requirements.txt, but if this doesn't work:)

[//]: # ()
[//]: # (    - Install pytorch following the official [pytorch )

[//]: # (    installation guide]&#40;https://pytorch.org/get-started/locally/&#41;.)

[//]: # ()
[//]: # (    - Install the following libraries with pip:)

[//]: # (      - pip install pyyaml matplotlib tqdm nibabel scikit-image monai )

[//]: # (      monai-generative nnunet lpips)

[//]: # ()
[//]: # (  - &#40;Optional&#41; You can install these libraries also for jupyter notebooks and)

[//]: # (  interactive visualization:)

[//]: # (    - pip install jupyter matplotlib ipywidgets ipympl notebook tornado)

[//]: # (  - run pip install -e . when you are in the main directory)
 

## Usage Instructions

### Environment variables

First, set the following environment variables:

```bash
export DATAPATH=/home/path_to_your_data_folder/
export SAVEPATH=/home/path_to_your_save_folder/
```
DATAPATH is the path to the folder that contains all your dataset folders. SAVEPATH
is the path to the folder where all the results will be saved.


### Dataset preparation
Now create your dataset (if you haven't done yet). Datasets must follow 
the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) format, where
all training images are contained in a folder called **imagesTr**, and are compressed 
nifti files (.nii.gz). All your datasets should be in the same folder for 
flexibility (in the DATAPATH), and the names of the dataset folders give them a 
unique *task* identifier.

### Dataset normalization and histogram equalization

First, you want to make sure that unnecessary background areas are removed and that
all the images have the same voxel spacing. For this run:

```bash
preprocess_dataset --task your_task
```

This will create a new folder called your_task_preprocessed in your data path with 
all the images cropped to non-zero regions, and resampled to the median voxel 
spacing of the dataset. 

If you also want to perform histogram equalization run:

```bash
preprocess_dataset --task your_task -intensity
```

### Configuration
All the global settings of the projects are stored in a configuration file. The 
*config.yaml* file in the configs folder is a basic configuration. DO NOT use this 
as your configuration file as it will not work. It is only there just to show you
all the possible parameters that you can modify. You should create your own file and 
call it when running the main.py (see next section), but keep all the config files 
in the /medimgen/configs folder. It is handy to store different configuration files 
for different experiments (e.g., one file for diffusion model and one for latent 
diffusion model).

### Training

Here is an example to train a Denoising Diffusion Probabilistic Model on the
Brain Tumour dataset from Medical Segmentation Decathlon:

```bash
train_ddpm --task Task01_BrainTumour --config ddpm_config --output_mode log
```
We are using a custom configuration file, and we are saving every output in a 
log file instead of printing on screen (--output_mode is not required).

To train a Latent Diffusion Model, first we need to train a VQ-GAN:

```bash
train_vqgan --task Task01_BrainTumour --config vqgan_config --output_mode log
```
After finishing training, we can then train the Latent Diffusion Model, providing
the path to the VQ-GAN checkpoint:

```bash
train_ldm --task Task01_BrainTumour --config ldm_config --output_mode log \
--vqgan_checkpoint /path_to_vqgan_checkpoint
```

Running an experiment will create a directory in your save_path. Depending 
on your configuration the following folders or files will be created: 
- checkpoints folder: the checkpoints of the last and the best epoch of the training 
will be saved here. For ddpm and ldm models only 2 checkpoints will be saved (the 
last and the best). For vqgan model 4 checkpoints will be saved (2 for generator and
2 for discriminator).
- model.graph: images containing the graphs of the models involved in the training
- plots folder: in this folder a main_loss.png file will be saved for every kind of
training, showing the training and validation loss per epoch. For the vqgan, an 
additional gan_loss.png will be saved showing the progression of generator and 
discriminator. Moreover, a gif will be saved in this folder for every epoch that we
perform a validation step. For ddpm and ldm, the gifs contain slices of a generated 
3D example across the z direction, while for vqgan, an actual image and its
reconstruction are visualized.

### Sampling