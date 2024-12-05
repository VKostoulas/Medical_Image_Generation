# 2D and 3D Medical Image Generation with Diffusion models and GANs

## Description
Training and sampling with 2D or 3D image generation models on your dataset
has never been that easy! Simply, create your dataset based on [Medical 
Segmentation Decathlon](http://medicaldecathlon.com/) format, and that's it!
Go train your model! Assuming that you have enough GPU memory (these 3D models
can take a colossal amount of memory) and you know exactly your architecture...

## Requirements

- python 3.9

It's better to install pytorch on your own following the official [pytorch 
installation guide](https://pytorch.org/get-started/locally/). Also make the 
following installations with pip:

- pip install pyyaml
- pip install matplotlib
- pip install tqdm
- pip install nibabel
- pip install scikit-image
- pip install monai
- pip install monai-generative

## Usage Example

[//]: # (To run the program, use the following command in your terminal:)

```bash
python main.py \
--mode train \
--model ddpm \
--data_path /home/path_to_your_data_folder/ \
--save_path /home/path_to_your_save_folder/ \
--output_mode log
```

- The config.yaml file contains all the global parameters that will be used in 
any experiment you run. If you wish to change the values then either modify 
the config file, or parse the argument with the changed value when you run
main.py.
- Current acceptable modes: train
- Current acceptable models: ddpm