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

[//]: # (      monai-generative nnunet)

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
flexibility (in the DATAPATH), and the names of the dataset folders gives them a 
unique *task* identifier.

### Dataset normalization and histogram equalization

First, you want to make sure that unnecessary background areas are removed and that
all the images have the same voxel spacing. For this run:

```bash
preprocess_dataset --task your_task
```

This will create a new folder called your_task_preprocessed. If you also want to 
perform histogram equalization run:

```bash
preprocess_dataset --task your_task -intensity
```

### Configuration
All the global settings of the projects are stored in a configuration file. The 
*config.yaml* file in the configs folder is a basic configuration. You can use this 
without defining any configuration file, or create your own file and call it when 
running the main.py (see next section), but keep all the config files in the configs
folder. It is handy to store different configuration files for different experiments 
(e.g., one file for diffusion model and one for latent diffusion model).

### Training

Here is an example to train a Denoising Diffusion Probabilistic Model on the
Brain Tumour dataset from Medical Segmentation Decathlon:
Example:

```bash
train_ddpm --task Task01_BrainTumour --config ddpm_config --output_mode log
```
we are using a custom 
configuration file, and we are saving every output in a log file instead of printing 
on screen.

Running an experiment will create a directory in your save_path. Depending 
on your configuration the following folders or files will be created: 
*checkpoints* folder, *model.graph*, *plots* folder.

### Sampling