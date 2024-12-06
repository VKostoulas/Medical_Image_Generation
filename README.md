# 2D and 3D Medical Image Generation with Diffusion models and GANs

## Description
Training and sampling with 2D or 3D image generation models on your dataset
has never been that easy! Simply, create your dataset, and that's it!
Go train your model! Assuming that you have enough GPU memory (these 3D models
can take a colossal amount of memory) and you know exactly your architecture...

## Requirements

- python 3.9

- Install pytorch following the official [pytorch 
installation guide](https://pytorch.org/get-started/locally/).

- Install the following libraries with pip:
  - pip install pyyaml matplotlib tqdm nibabel scikit-image monai 
  monai-generative nnunet


## Usage Instructions

### Dataset preparation
First create your dataset (if you haven't done yet). Datasets must follow 
the [Medical Segmentation Decathlon](http://medicaldecathlon.com/) format, where
all training images are contained in a folder called **imagesTr**, and are compressed 
nifti files (.nii.gz). All your datasets should be in the same folder for 
flexibility, and the names of the dataset folders gives them a unique *task* 
identifier.

### Configuration
All the global settings of the projects are stored in the config.yaml file. Modify 
the yaml file in your preference, or parse the modifications when running main.py
(see next section).

### Training

To train, simply run *main.py* with required arguments *mode* (train or sample),
*model* (which generative model to use; currently only ddpm), *task* (the task to 
train your model on), *data_path* (the path to the folder that contains all the task 
folders) and *save_path* (path to save results). You can also modify any parameter of
the config file when running main.py

Example:
```bash
python main.py \
--mode train \
--model ddpm \
--task your_task \
--data_path /home/path_to_your_data_folder/ \
--save_path /home/path_to_your_save_folder/ \
--output_mode log
```
In the example we are training with a Denoising Diffusion Probabilistic Model, and 
we are saving every output in a log file instead of printing on screen.

Running an experiment will create a directory in your save_path. Depending 
on your configuration the following folders or files will be created: 
*checkpoints* folder, *model.graph*, *plots* folder.

### Sampling