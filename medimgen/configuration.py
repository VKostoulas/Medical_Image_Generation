import json
import os
import ast
import cv2
import glob
import zarr
import pickle
import scipy
import shutil
import sys
import math
import yaml
import argparse
import logging
import matplotlib
import nibabel as nib
import numpy as np
import concurrent.futures
import torch
import gc
import copy

from datetime import datetime
from numcodecs import Blosc
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from skimage.filters import threshold_otsu

from medimgen.data_processing import get_data_loaders
from medimgen.train_autoencoder import AutoEncoder


# def load_config(config_name):
#     """Load default configuration from a YAML file."""
#     if config_name:
#         final_config_name = config_name
#     else:
#         final_config_name = 'config'
#     conf_path = os.path.join(os.getcwd(), 'medimgen', 'configs', final_config_name + '.yaml')
#     with open(conf_path, "r") as file:
#         config_file = yaml.safe_load(file)
#         config_file['config'] = final_config_name
#         return config_file


def add_preprocessing_args(parser):
    parser.add_argument("--intensity", type=lambda x: x.lower() == 'true', help="Enable normalization during dataset processing.")


def add_training_args(parser):
    parser.add_argument("--config", type=str, help="Configuration file name")
    parser.add_argument("--splitting", nargs=2, type=float, help="Split ratios for train, val")
    parser.add_argument("--channels", nargs='+', type=int, help="List of channel indices or None")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs")
    parser.add_argument("--val_interval", type=int, help="Validation interval")
    parser.add_argument("--grad_accumulate_step", type=int, help="Number of steps to accumulate gradients")
    parser.add_argument("--grad_clip_max_norm", type=float, help="Max norm for gradient clipping")
    # Parsing arguments for lr_scheduler
    parser.add_argument("--lr_scheduler", type=str, help="Type of learning rate scheduler")
    parser.add_argument("--start_factor", type=float, help="Start factor for learning rate scheduler")
    parser.add_argument("--end_factor", type=float, help="End factor for learning rate scheduler")
    parser.add_argument("--total_iters", type=int, help="Total iterations for the learning rate scheduler")
    parser.add_argument("--load_model_path", type=str, help="Path to checkpoint of pretrained model")


def add_transformation_args(parser):
    parser.add_argument("--patch_size", nargs=3, type=int, help="Center crop size")
    parser.add_argument("--resize_shape", nargs=3, type=int, help="Resized size")
    parser.add_argument("--elastic", type=lambda x: x.lower() == 'true', help="Enable elastic transformations")
    parser.add_argument("--scaling", type=lambda x: x.lower() == 'true', help="Enable scaling transformations")
    parser.add_argument("--rotation", type=lambda x: x.lower() == 'true', help="Enable rotation transformations")
    parser.add_argument("--gaussian_noise", type=lambda x: x.lower() == 'true', help="Enable Gaussian noise")
    parser.add_argument("--gaussian_blur", type=lambda x: x.lower() == 'true', help="Enable Gaussian blur")
    parser.add_argument("--brightness", type=lambda x: x.lower() == 'true', help="Enable brightness adjustment")
    parser.add_argument("--contrast", type=lambda x: x.lower() == 'true', help="Enable contrast adjustment")
    parser.add_argument("--gamma", type=lambda x: x.lower() == 'true', help="Enable gamma adjustment")
    parser.add_argument("--mirror", type=lambda x: x.lower() == 'true', help="Enable mirroring")
    parser.add_argument("--dummy_2D", type=lambda x: x.lower() == 'true', help="Enable dummy 2D mode")


def add_ddpm_args(parser):
    parser.add_argument("--ddpm_learning_rate", type=float, help="Learning rate")
    parser.add_argument("--time_scheduler_num_train_timesteps", type=int, help="Number of training timesteps")
    # parser.add_argument("--time_scheduler_n_infer_timesteps", type=int, help="Number of inference timesteps")
    parser.add_argument("--time_scheduler_schedule", type=str, help="Time scheduler type")
    parser.add_argument("--time_scheduler_beta_start", type=float, help="Beta start for scheduler")
    parser.add_argument("--time_scheduler_beta_end", type=float, help="Beta end for scheduler")
    parser.add_argument("--time_scheduler_prediction_type", type=str, help="DDPM prediction type ('epsilon' or 'v_prediction')")

    parser.add_argument("--ddpm_spatial_dims", type=int, help="Spatial dimensions")
    parser.add_argument("--ddpm_in_channels", type=int, help="Number of input channels")
    parser.add_argument("--ddpm_out_channels", type=int, help="Number of output channels")
    parser.add_argument("--ddpm_num_channels", nargs='+', type=int, help="List of channel numbers for the model")
    parser.add_argument("--ddpm_attention_levels", nargs='+', type=lambda x: x.lower() == 'true',
                        help="List of attention levels")
    parser.add_argument("--ddpm_num_head_channels", nargs='+', type=int, help="List of head channel numbers")
    parser.add_argument("--ddpm_num_res_blocks", type=int, help="Number of residual blocks")
    parser.add_argument("--ddpm_norm_num_groups", type=int, help="Number of groups for normalization")
    parser.add_argument("--ddpm_use_flash_attention", type=lambda x: x.lower() == 'true',
                        help="Use flash attention for speed and memory efficiency")

def add_vqvae_args(parser):
    parser.add_argument("--vqvae_spatial_dims", type=int, help="Spatial dimensions for model parameters")
    parser.add_argument("--vqvae_in_channels", type=int, help="Number of input channels for the model")
    parser.add_argument("--vqvae_out_channels", type=int, help="Number of output channels for the model")
    parser.add_argument("--vqvae_num_channels", nargs='+', type=int, help="List of channel numbers for the model")
    parser.add_argument("--vqvae_num_res_channels", nargs='+', type=int,
                        help="Number of residual channels in the model")
    parser.add_argument("--vqvae_num_res_layers", type=int, help="Number of residual layers in the model")
    parser.add_argument("--vqvae_downsample_parameters", nargs='+', type=eval,
                        help="Parameters for downsampling in the model")
    parser.add_argument("--vqvae_upsample_parameters", nargs='+', type=eval,
                        help="Parameters for upsampling in the model")
    parser.add_argument("--vqvae_num_embeddings", type=int, help="Number of embeddings for the model")
    parser.add_argument("--vqvae_embedding_dim", type=int, help="Embedding dimension for the model")
    parser.add_argument("--vqvae_use_checkpointing", type=lambda x: x.lower() == 'true',
                        help="Use activation checkpointing")


def add_vae_args(parser):
    parser.add_argument("--vae_spatial_dims", type=int, help="Spatial dimensions for VAE model")
    parser.add_argument("--vae_in_channels", type=int, help="Number of input channels for the VAE model")
    parser.add_argument("--vae_out_channels", type=int, help="Number of output channels for the VAE model")
    parser.add_argument("--vae_num_channels", nargs='+', type=int,
                        help="List of channel numbers for the VAE model")
    parser.add_argument("--vae_latent_channels", type=int, help="Number of latent channels for the VAE model")
    parser.add_argument("--vae_num_res_blocks", type=int, help="Number of residual blocks in the VAE model")
    parser.add_argument("--vae_norm_num_groups", type=int,
                        help="Number of groups for normalization in the VAE model")
    parser.add_argument("--vae_attention_levels", nargs='+', type=lambda x: x.lower() in ['true', 'false'],
                        help="List of attention levels (True/False) for the VAE model")
    parser.add_argument("--vae_with_encoder_nonlocal_attn", type=lambda x: x.lower() == 'true',
                        help="Use non-local attention in the VAE encoder")
    parser.add_argument("--vae_with_decoder_nonlocal_attn", type=lambda x: x.lower() == 'true',
                        help="Use non-local attention in the VAE decoder")
    parser.add_argument("--vae_use_flash_attention", type=lambda x: x.lower() == 'true',
                        help="Use flash attention for VAE")
    parser.add_argument("--vae_use_checkpointing", type=lambda x: x.lower() == 'true',
                        help="Use activation checkpointing for VAE")
    parser.add_argument("--vae_use_convtranspose", type=lambda x: x.lower() == 'true',
                        help="Use ConvTranspose layers in the VAE")
    parser.add_argument("--vae_downsample_parameters", nargs='+', type=eval,
                        help="Parameters for downsampling in the model")
    parser.add_argument("--vae_upsample_parameters", nargs='+', type=eval,
                        help="Parameters for upsampling in the model")


def add_autoencoder_training_args(parser):
    parser.add_argument("--g_learning_rate", type=float, help="Generator learning rate")
    parser.add_argument("--d_learning_rate", type=float, help="Discriminator learning rate")
    parser.add_argument("--autoencoder_warm_up_epochs", type=int, help="Number of epochs to warm up vqvae")
    parser.add_argument("--adv_weight", type=float, help="Adversarial loss weight")
    parser.add_argument("--perc_weight", type=float, help="Perceptual loss weight")
    # Perceptual parameters
    parser.add_argument("--perceptual_spatial_dims", type=int, help="Spatial dimensions for perceptual parameters")
    parser.add_argument("--network_type", type=str, help="Type of perceptual network")
    parser.add_argument("--is_fake_3d", type=lambda x: x.lower() == 'true',
                        help="Flag to indicate if fake 3D is used")
    parser.add_argument("--fake_3d_ratio", type=float, help="Ratio for fake 3D")
    # Discriminator parameters
    parser.add_argument("--discriminator_spatial_dims", type=int,
                        help="Spatial dimensions for discriminator parameters")
    parser.add_argument("--discriminator_in_channels", type=int, help="Number of input channels for discriminator")
    parser.add_argument("--discriminator_out_channels", type=int,
                        help="Number of output channels for discriminator")
    parser.add_argument("--discriminator_num_channels", type=int, help="Number of channels in the discriminator")
    parser.add_argument("--discriminator_num_layers_d", type=int, help="Number of layers of discriminator")


def add_additional_args(parser):
    parser.add_argument("--progress_bar", type=lambda x: x.lower() == 'true', help="Use progress bars")
    parser.add_argument("--output_mode", type=str, help="Output mode")


def parse_arguments(description, args_mode):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--task", required=True, type=str, help="Task identifier")

    # Add mode-specific arguments
    if args_mode == 'preprocess_data':
        add_preprocessing_args(parser)

    elif args_mode in ['train_ddpm', 'train_autoencoder', 'train_ldm']:
        add_training_args(parser)
        add_transformation_args(parser)

        if args_mode in ['train_autoencoder', 'train_ldm']:
            # Latent space type
            parser.add_argument("--latent_space_type", type=str, default="vae", choices=["vae", "vq"],
                                help="Type of latent space to use: 'vae' or 'vq'. Default is 'vae'.")

            # Parse the known arguments so far to determine latent_space_type
            temp_args, _ = parser.parse_known_args()
            latent_space_type = temp_args.latent_space_type if hasattr(temp_args, 'latent_space_type') else 'vae'

            if latent_space_type == 'vq':
                if args_mode == 'train_autoencoder':
                    parser.add_argument("--q_weight", type=float, help="Quantization loss weight")
                    add_vqvae_args(parser)

            elif latent_space_type == 'vae':
                if args_mode == 'train_autoencoder':
                    parser.add_argument("--kl_weight", type=float, help="KL divergence loss weight (used for VAE).")
                    add_vae_args(parser)

        if args_mode == 'train_autoencoder':
            add_autoencoder_training_args(parser)

        if args_mode in ['train_ddpm', 'train_ldm']:
            add_ddpm_args(parser)

        if args_mode == 'train_ldm':
            parser.add_argument("--load_autoencoder_path", type=str, required=True,
                                help="Path to checkpoint of pretrained autoencoder (VQ-VAE or VAE).")

        add_additional_args(parser)

    # Parse and return arguments
    return parser.parse_args()


def update_config_with_args(config, args, args_mode):
    config["task"] = str(args.task)
    config["data_path"] = str(os.getenv('DATAPATH'))
    config["save_path"] = str(os.getenv('SAVEPATH'))

    # Update config only if arguments were provided
    if args_mode in ['train_ddpm', 'train_autoencoder', 'train_ldm']:
        if args.splitting is not None:
            config["splitting"] = args.splitting
        if args.channels is not None:
            config["channels"] = args.channels

        if args.patch_size is not None:
            config["transformations"]["patch_size"] = args.patch_size
        if args.resize_shape is not None:
            config["transformations"]["resize_shape"] = args.resize_shape
        for key in [
            "elastic", "scaling", "rotation", "gaussian_noise",
            "gaussian_blur", "brightness", "contrast", "gamma",
            "mirror", "dummy_2D"]:
            if getattr(args, key, None) is not None:
                config["transformations"][key] = getattr(args, key)

        if args.batch_size is not None:
            config["batch_size"] = args.batch_size
        if args.n_epochs is not None:
            config["n_epochs"] = args.n_epochs
        if args.val_interval is not None:
            config["val_interval"] = args.val_interval
        if args.grad_accumulate_step is not None:
            config["grad_accumulate_step"] = args.grad_accumulate_step
        if args.grad_clip_max_norm is not None:
            config["grad_clip_max_norm"] = args.grad_clip_max_norm

        if args.lr_scheduler is not None:
            config["lr_scheduler"] = args.lr_scheduler
        for key in ["start_factor", "end_factor", "total_iters"]:
            if getattr(args, key, None) is not None:
                config["lr_scheduler_params"][key] = getattr(args, key)

        if args.load_model_path is not None:
            config["load_model_path"] = args.load_model_path

    # ddpm params
    if args_mode in ['train_ddpm', 'train_ldm']:
        if args.ddpm_learning_rate is not None:
            config["ddpm_learning_rate"] = args.ddpm_learning_rate

        for key in ["num_train_timesteps", "schedule", "beta_start", "beta_end", "prediction_type"]:
            if getattr(args, f"time_scheduler_{key}", None) is not None:
                config["time_scheduler_params"][key] = getattr(args, key)
        for key in [
            "spatial_dims", "in_channels", "out_channels", "num_channels",
            "attention_levels", "num_head_channels", "num_res_blocks",
            "norm_num_groups", "use_flash_attention"
        ]:
            if getattr(args, f"ddpm_{key}", None) is not None:
                config["ddpm_params"][key] = getattr(args, f"ddpm_{key}")

    # autoencoder-specific params
    if args_mode in ['train_autoencoder']:
        for key in ["g_learning_rate", "d_learning_rate", "autoencoder_warm_up_epochs", "adv_weight", "perc_weight"]:
            if getattr(args, key, None) is not None:
                config[key] = getattr(args, key)
        for key in ["spatial_dims", "network_type", "is_fake_3d", "fake_3d_ratio"]:
            if getattr(args, f"perceptual_{key}", None) is not None:
                config["perceptual_params"][key] = getattr(args, f"perceptual_{key}")
        for key in ["spatial_dims", "in_channels", "out_channels", "num_channels", "num_layers_d"]:
            if getattr(args, f"discriminator_{key}", None) is not None:
                config["discriminator_params"][key] = getattr(args, f"discriminator_{key}")

    if args_mode in ['train_autoencoder', 'train_ldm']:
        if args.latent_space_type is not None:
            config["latent_space_type"] = args.latent_space_type
        # VQVAE params
        for key in [
            "spatial_dims", "in_channels", "out_channels", "num_channels",
            "num_res_channels", "num_res_layers", "downsample_parameters",
            "upsample_parameters", "num_embeddings", "embedding_dim", "use_checkpointing"
        ]:
            if getattr(args, f"vqvae_{key}", None) is not None:
                config["vqvae_params"][key] = getattr(args, f"vqvae_{key}")

        # VAE params
        for key in [
            "spatial_dims", "in_channels", "out_channels", "num_channels",
            "latent_channels", "num_res_blocks", "norm_num_groups", "attention_levels",
            "with_encoder_nonlocal_attn", "with_decoder_nonlocal_attn", "use_flash_attention",
            "use_checkpointing", "use_convtranspose", "downsample_parameters", "upsample_parameters"
        ]:
            if getattr(args, f"vae_{key}", None) is not None:
                config["vae_params"][key] = getattr(args, f"vae_{key}")

    # Additional arguments
    for key in ["progress_bar", "output_mode"]:
        if getattr(args, key, None) is not None:
            config[key] = getattr(args, key)

    if args_mode == 'train_ldm':
        if args.load_autoencoder_path is not None:
            config["load_autoencoder_path"] = args.load_autoencoder_path

    return config


def filter_config_by_mode(config, args_mode):
    """
    Filters the configuration object by removing unnecessary arguments based on args_mode.

    Args:
        config (dict): The configuration dictionary.
        args_mode (str): The current mode of operation.

    Returns:
        dict: Filtered configuration dictionary.
    """
    if args_mode == 'train_ddpm':
        config.pop('latent_space_type', None)
        config.pop("vae_params", None)
        config.pop("kl_weight", None)
        config.pop("vqvae_params", None)
        config.pop("q_weight", None)
        config.pop("load_autoencoder_path", None)

    if args_mode == "train_autoencoder":
        # Remove DDPM-related parameters
        for key in ["ddpm_params", "time_scheduler_params", "ddpm_learning_rate", "load_autoencoder_path"]:
            config.pop(key, None)

    if args_mode in ["train_ddpm", "train_ldm"]:
        # Remove autoencoder-specific parameters
        for key in [
            "g_learning_rate", "d_learning_rate", "q_weight", "kl_weight",
            "adv_weight", "perc_weight", "autoencoder_warm_up_epochs",
            "perceptual_params", "discriminator_params"
        ]:
            config.pop(key, None)

    if args_mode in ["train_autoencoder", "train_ldm"]:
        # Handle latent space-specific filtering
        latent_space_type = config.get("latent_space_type", "vae").lower()
        if latent_space_type == "vq":
            # Remove VAE-specific parameters
            config.pop("vae_params", None)
            config.pop("kl_weight", None)
        elif latent_space_type == "vae":
            # Remove VQVAE-specific parameters
            config.pop("vqvae_params", None)
            config.pop("q_weight", None)

    return config


def create_save_path_dict(config):
    """
    Create a dictionary with folder names and save paths for various outputs (e.g., checkpoints, graphs, etc.).
    Args:
        config (dict): The sanitized configuration dictionary.
    Returns:
        dict: A dictionary where keys are folder names and values are paths or False (if saving is disabled).
    """
    # Generate a timestamped save directory
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = os.path.join(config['save_path'], timestamp)
    os.mkdir(save_path)

    # Setup logging only if mode is 'log'
    if config["output_mode"] == "log":
        log_file_path = os.path.join(save_path, 'log_file.txt')
        setup_logging(log_file_path)

    # save the config parameters in a file
    with open(os.path.join(save_path, 'config.yaml'), 'w') as file:
        yaml.dump(config, file, default_flow_style=False,  sort_keys=False)

    save_dict = {'checkpoints': os.path.join(save_path, 'checkpoints'), 'plots': os.path.join(save_path, 'plots')}

    return save_dict, save_path


def print_configuration(config, save_path, mode, space_from_start=40, model=None):
    """
    Print the mode, model, and configuration parameters in a perfectly aligned format.
    Args:
        config (dict): Configuration dictionary.
        mode (str): Mode of operation (e.g., "train").
        model (str): Model type (e.g., "ddpm").
        save_path (str):
        space_from_start (int): Column where values should start.
    """
    def flatten_dict(d, parent_key="", sep="."):
        """Flatten nested dictionaries for better display."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_config = flatten_dict(config)

    # Print header with mode and model
    header = f"{'Configuration Summary'.center(space_from_start * 3)}\n"
    print(header + "=" * (space_from_start * 3))

    data_path = os.path.join(config['data_path'], config['task'], 'imagesTr')

    print(f"Mode{' ' * (space_from_start - len('Mode'))}{mode}")
    if model:
        print(f"Model{' ' * (space_from_start - len('Model'))}{model}")
    print(f"Task{' ' * (space_from_start - len('Task'))}{config['task']}")
    if model:
        print(f"Configuration File{' ' * (space_from_start - len('Configuration File'))}{config['config']}")
    print(f"Data Path{' ' * (space_from_start - len('Data Path'))}{data_path}")
    print(f"Save Path{' ' * (space_from_start - len('Save Path'))}{save_path}")
    if model:
        print("\nParameters:\n" + "-" * (space_from_start * 3))

        # Print each parameter with aligned values
        for i, (key, value) in enumerate(flat_config.items()):
            if key not in ["task", "config", "data_path", "save_path"]:  # Skip already printed keys
                spaces = " " * (space_from_start - len(key))  # Calculate spaces for alignment
                if i == len(flat_config) - 5:
                    print(f"{key}{spaces}{value}\n{'=' * (space_from_start * 3)}")
                else:
                    print(f"{key}{spaces}{value}")
    else:
        print(f"{'=' * (space_from_start * 3)}")


def suppress_console_handlers():
    """
    Suppress console (StreamHandler) output from all loggers except the root logger.
    """
    root_logger = logging.getLogger()
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger):
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    logger.removeHandler(handler)  # Remove StreamHandler
            logger.propagate = True  # Ensure logs flow to root logger


def setup_logging(log_file_path=None):
    """
    Setup logging to capture all logs, suppress console outputs, and redirect print statements.
    Args:
        log_file_path (str): Path to the log file.
    """
    if not log_file_path:
        raise ValueError("log_file_path must be provided when logging is enabled.")

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # Capture all levels, DEBUG and higher
    logger.handlers = []  # Clear existing handlers

    # File handler to capture all logs
    file_handler = logging.FileHandler(log_file_path, mode='w')
    file_handler.setLevel(logging.INFO)  # Log everything to the file
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Redirect stdout and stderr to logger
    sys.stdout = LoggerWriter(logger, logging.INFO)
    sys.stderr = LoggerWriter(logger, logging.ERROR)

    # Suppress third-party console handlers
    suppress_console_handlers()

    # Suppress unnecessary debug logs from matplotlib
    matplotlib.set_loglevel("warning")


class LoggerWriter:
    """
    Redirects stdout and stderr to the logging module.
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, message):
        if message.strip():  # Avoid empty messages
            self.logger.log(self.level, message.strip())

    def flush(self):
        pass


def validate_channels(value):
    try:
        channels = ast.literal_eval(value)
        if isinstance(channels, list) and all(isinstance(x, int) for x in channels):
            return channels
    except (ValueError, SyntaxError):
        pass
    raise argparse.ArgumentTypeError("Channels must be a list of integers.")


# def create_autoencoder_dict(nnunet_config_dict, input_channels, spatial_dims):
#
#     # features_per_stage = nnunet_config_dict['architecture']['arch_kwargs']['features_per_stage']
#     kernel_sizes = nnunet_config_dict['architecture']['arch_kwargs']['kernel_sizes']
#     strides = nnunet_config_dict['architecture']['arch_kwargs']['strides']
#
#     median_image_size = nnunet_config_dict['median_image_size_in_voxels']
#
#     # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
#     valid_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
#     patch_size_3d = [min(valid_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
#     patch_size = nnunet_config_dict['patch_size'] if spatial_dims == 2 else patch_size_3d
#
#     base_autoencoder_channels = [32, 64, 128, 128]
#
#     vae_dict = {'spatial_dims': spatial_dims,
#                 'in_channels': len(input_channels),
#                 'out_channels': len(input_channels),
#                 'latent_channels': 8,
#                 'num_res_blocks': 2,
#                 'with_encoder_nonlocal_attn': False,
#                 'with_decoder_nonlocal_attn': False,
#                 'use_flash_attention': False,
#                 'use_checkpointing': False,
#                 'use_convtranspose': False
#                }
#
#     # use maximum of 3 autoencoder downsampling layers
#     # for max image size 512, 3 ae layers --> latent size 64 --> good
#     # for max image size 400, 3 ae layers --> latent size 50 --> good
#     # for max image size 320, 3 ae layers --> latent size 40 --> good
#     # when 3 layers are more than needed? when latent size after 2 downsamplings is <= 64 --> patch_size <= 256
#     # for max image size 256, 2 ae layers --> latent size 64 --> good
#     # for max image size 200, 2 ae layers --> latent size 50 --> good
#     # for max image size 160, 2 ae layers --> latent size 40 --> good
#     # for max image size 128, 2 ae layers --> latent size 32 --> good
#     # when 2 layers are more than needed? when latent size after 1 downsamplings is <= 32 --> patch_size <= 64
#     # for max image size 100, 2 ae layers --> latent size 25 --> good
#     # for max image size 64, 1 ae layer --> latent size 32 --> good
#     if np.max(patch_size) <= 64:
#         vae_n_layers = 1
#     elif np.max(patch_size) <= 256:
#         vae_n_layers = 2
#     else:
#         vae_n_layers = 3
#
#     # vae_dict['num_channels'] = features_per_stage[:vae_n_layers+1]
#     # vae_dict['attention_levels'] = [False] * (vae_n_layers+1)
#     # vae_dict['norm_num_groups'] = vae_dict['num_channels'][0]
#     vae_dict['num_channels'] = base_autoencoder_channels[:vae_n_layers+1]
#     vae_dict['attention_levels'] = [False] * (vae_n_layers+1)
#     vae_dict['norm_num_groups'] = 16
#
#     # nnunet gives you the parameters of the first conv block and then all the downsample parameters
#     # For the autoencoder we pass these directly but for the ddpm things are a bit different (see create_ddpm_dict)
#     downsample_parameters = [[item1, item2] for item1, item2 in zip(strides[:vae_n_layers+1], kernel_sizes[:vae_n_layers+1])]
#     paddings = [[1 if k == 3 else 0 for k in layer] for layer in kernel_sizes[:vae_n_layers+1]]
#     downsample_parameters = [item1 + [item2] for item1, item2 in zip(downsample_parameters, paddings)]
#     vae_dict['downsample_parameters'] = downsample_parameters
#     vae_dict['upsample_parameters'] = list(reversed(downsample_parameters))[:-1]
#     return vae_dict
#
#
# def create_ddpm_dict(nnunet_config_dict, spatial_dims):
#
#     # features_per_stage = nnunet_config_dict['architecture']['arch_kwargs']['features_per_stage']
#     kernel_sizes = nnunet_config_dict['architecture']['arch_kwargs']['kernel_sizes']
#     strides = nnunet_config_dict['architecture']['arch_kwargs']['strides']
#
#     median_image_size = nnunet_config_dict['median_image_size_in_voxels']
#
#     # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
#     valid_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
#     patch_size_3d = [min(valid_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
#     patch_size = nnunet_config_dict['patch_size'] if spatial_dims == 2 else patch_size_3d
#
#     ddpm_dict = {'spatial_dims': spatial_dims,
#                  'in_channels': 8,
#                  'out_channels': 8,
#                  'num_res_blocks': 2,
#                  'use_flash_attention': False,
#                 }
#
#     # check create_autoencoder_dict
#     if np.max(patch_size) <= 64:
#         vae_n_layers = 1
#     elif np.max(patch_size) <= 256:
#         vae_n_layers = 2
#     else:
#         vae_n_layers = 3
#
#     # ddpm_dict['num_channels'] = features_per_stage[vae_n_layers:]
#     # if len(ddpm_dict['num_channels']) < 2:
#     #     raise ValueError("The number of stages must be at least 2.")
#     # # First 2 stages without attention, then attention for the rest
#     # ddpm_dict['attention_levels'] = [False, False] + [True] * (len(ddpm_dict['num_channels']) - 2)
#     ddpm_dict['num_channels'] = [256, 512, 768]
#     ddpm_dict['attention_levels'] = [False, True, True]
#     ddpm_dict['num_head_channels'] = [0, 512, 768]
#
#     # if len(ddpm_dict['num_channels']) != len(ddpm_dict['attention_levels']):
#     #     raise ValueError("num_channels and attention_levels must be of the same length.")
#     # ddpm_dict['num_head_channels'] = [channel if use_attention else 0
#     #                                      for channel, use_attention in zip(ddpm_dict['num_channels'], ddpm_dict['attention_levels'])]
#
#     # Now the remaining conv parameters from nnunet do not involve the first conv block of the ddpm unet
#     # For the first layer of the ddpm unet we always keep the strides at 1, but we take the kernel sizes from the
#     # corresponding layer of nnunet. Then we use all the corresponding nnunet layers for the rest of diffusion layers
#     ddpm_dict['strides'] = [[1] * spatial_dims] + strides[vae_n_layers+1:vae_n_layers+3]
#     ddpm_dict['kernel_sizes'] = [kernel_sizes[vae_n_layers+1]] + kernel_sizes[vae_n_layers+1:vae_n_layers+3]
#     ddpm_dict['paddings'] = [[1 if k == 3 else 0 for k in layer] for layer in ddpm_dict['kernel_sizes']]
#
#     return ddpm_dict


# def create_config_dict(nnunet_config_dict, input_channels, n_epochs_multiplier, autoencoder_dict, ddpm_dict):
#
#     # features_per_stage = nnunet_config_dict['architecture']['arch_kwargs']['features_per_stage']
#     median_image_size = nnunet_config_dict['median_image_size_in_voxels']
#
#     # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
#     valid_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
#     patch_size_3d = [min(valid_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
#     patch_size = nnunet_config_dict['patch_size'] if autoencoder_dict['spatial_dims'] == 2 else patch_size_3d
#
#     print(f"Patch size {autoencoder_dict['spatial_dims']}D: {patch_size}")
#
#     ae_transformations = {
#         "patch_size": patch_size,
#         "scaling": True,
#         "rotation": True,
#         "gaussian_noise": False,
#         "gaussian_blur": False,
#         "low_resolution": False,
#         "brightness": True,
#         "contrast": True,
#         "gamma": True,
#         "mirror": True,
#         "dummy_2d": False
#     }
#     # not using augmentations for ddpm training
#     ddpm_transformations = {
#         "patch_size": patch_size,
#         "scaling": False,
#         "rotation": False,
#         "gaussian_noise": False,
#         "gaussian_blur": False,
#         "low_resolution": False,
#         "brightness": False,
#         "contrast": False,
#         "gamma": False,
#         "mirror": False,
#         "dummy_2d": False
#     }
#
#     if autoencoder_dict['spatial_dims'] == 2:
#         perceptual_params = {'spatial_dims': 2, 'network_type': "vgg"}
#     else:
#         perceptual_params = {'spatial_dims': 3, 'network_type': "vgg", 'is_fake_3d': True, 'fake_3d_ratio': 0.2}
#
#     discriminator_params = {'spatial_dims': autoencoder_dict['spatial_dims'], 'in_channels': autoencoder_dict['in_channels'],
#                             'out_channels': 1, 'num_channels': 64, 'num_layers_d': 3}
#
#     # adjust the number of epochs based on the training model (2D/3D) and number of training data
#     n_epochs = 300 if autoencoder_dict['spatial_dims'] == 3 else 200
#     n_epochs = n_epochs * n_epochs_multiplier
#
#     # adjust the batch size and gradient accumulation
#     if autoencoder_dict['spatial_dims'] == 2:
#         # for 2d use 75% of batch size for both ae and ddpm
#         ae_batch_size = int(nnunet_config_dict['batch_size'] * 0.75)
#         ddpm_batch_size = int(nnunet_config_dict['batch_size'] * 0.75)
#         grad_accumulate_step = 1
#     else:
#         ae_batch_size = 2
#         ddpm_batch_size = ae_batch_size * 2
#         grad_accumulate_step = 1
#
#     # if batch size and patch size get large, use gradient accumulation
#     if math.prod(patch_size + [ae_batch_size]) > 2e+6:
#         ae_batch_size //= 2
#         ddpm_batch_size //= 2
#         grad_accumulate_step *= 2
#         print(f"We will use 2 gradient accumulation steps while training in {autoencoder_dict['spatial_dims']}D.")
#
#     config = {
#         'input_channels': input_channels,
#         'ae_transformations': ae_transformations,
#         'ddpm_transformations': ddpm_transformations,
#         'ae_batch_size': ae_batch_size,
#         'ddpm_batch_size': ddpm_batch_size,
#         'n_epochs': n_epochs,
#         'val_plot_interval': 10,
#         'grad_clip_max_norm': 1,
#         'grad_accumulate_step': grad_accumulate_step,
#         'oversample_ratio': 0.33,
#         'num_workers': 8,
#         # 'lr_scheduler': "LinearLR",
#         # 'lr_scheduler_params': {'start_factor': 1.0, 'end_factor': 0., 'total_iters': n_epochs},
#         # 'lr_scheduler_params': {'start_factor': 1.0, 'end_factor': 0.0001, 'total_iters': int(n_epochs*0.9)},
#         'lr_scheduler': None, # "PolynomialLR",
#         'lr_scheduler_params': {'total_iters': n_epochs, 'power': 0.9},
#         'time_scheduler_params': {'num_train_timesteps': 1000, 'schedule': "scaled_linear_beta", 'beta_start': 0.0015,
#                                   'beta_end': 0.0205, 'prediction_type': "epsilon"},
#         'ae_learning_rate': 5e-5,
#         # 'weight_decay': 3e-5,
#         'd_learning_rate': 5e-5,
#         'autoencoder_warm_up_epochs': 5,
#         'adv_weight': 0.05,
#         'perc_weight': 0.5 if autoencoder_dict['spatial_dims'] == 2 else 0.125,
#         'vae_params': autoencoder_dict,
#         'perceptual_params': perceptual_params,
#         'discriminator_params': discriminator_params,
#         'ddpm_learning_rate': 2e-5,
#         'ddpm_params': ddpm_dict
#     }
#
#     # missing: grad_accumulate_step, q_weight, kl_weight, adv_weight, perc_weight, autoencoder_warm_up_epochs,
#     #          latent_space_type: "vae"
#
#     return config


def compute_downsample_parameters(input_size, num_layers):
    """
    Generalized to handle 1D, 2D, or 3D input sizes.

    Args:
        input_size: list of ints [D, H, W] or [H, W] or [W]
        num_layers: int, number of layers including the first one

    Returns:
        List of lists: [[[stride], [kernel], [padding]], ...] for each layer
    """
    ndim = len(input_size)
    current_size = list(input_size)
    parameters = []

    for i in range(num_layers):
        stride = [1] * ndim
        kernel = [3] * ndim
        padding = [1] * ndim

        if i == 0:
            # First layer: adjust based on dimension disparity
            for d in range(ndim):
                other_dims = [current_size[j] for j in range(ndim) if j != d]
                if current_size[d] <= 0.5 * max(other_dims, default=current_size[d]):
                    kernel[d] = 1
                    padding[d] = 0
        else:
            # Downsampling layers
            for d in range(ndim):
                other_dims = [current_size[j] for j in range(ndim) if j != d]
                if current_size[d] <= 0.5 * max(other_dims, default=current_size[d]):
                    stride[d] = 1
                    kernel[d] = 1
                    padding[d] = 0
                else:
                    stride[d] = 2
                    kernel[d] = 3
                    padding[d] = 1

            # Update size after downsampling
            for d in range(ndim):
                current_size[d] = (current_size[d] + 2 * padding[d] - kernel[d]) // stride[d] + 1

        parameters.append([stride, kernel, padding])

    return parameters


def compute_output_size(input_size, downsample_parameters):
    """
    Args:
        input_size: list of ints (e.g. [D, H, W] or [H, W])
        downsample_parameters: list of [[stride], [kernel], [padding]] per layer

    Returns:
        output_size: list of ints representing final size after all layers
    """
    output_size = list(input_size)

    for layer in downsample_parameters:
        stride, kernel, padding = layer
        for d in range(len(output_size)):
            output_size[d] = (
                (output_size[d] + 2 * padding[d] - kernel[d]) // stride[d]
            ) + 1

    return output_size


def create_autoencoder_dict(dataset_config, input_channels, spatial_dims):
    median_image_size = dataset_config['median_shape']
    max_image_size = dataset_config['max_shape']
    # For 2D, for each axis, use as size the closest multiple of 2, 3, 5 or 7 by powers of 2, to the corresponding size of max patch size
    valid_2d_sizes = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    patch_size_2d = [min(valid_2d_sizes, key=lambda x: abs(x - size)) for size in max_image_size]
    # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_3d_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_3d_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = patch_size_2d[1:] if spatial_dims == 2 else patch_size_3d

    base_autoencoder_channels = [64, 128, 256, 256] if spatial_dims == 2 else [32, 64, 128, 128]

    vae_dict = {'spatial_dims': spatial_dims,
                'in_channels': len(input_channels),
                'out_channels': len(input_channels),
                'latent_channels': 8,
                'num_res_blocks': 2,
                'with_encoder_nonlocal_attn': False,
                'with_decoder_nonlocal_attn': False,
                'use_flash_attention': False,
                'use_checkpointing': False,
                'use_convtranspose': False
               }

    # use maximum of 3 autoencoder downsampling layers
    # we want the latent dims to be less than 100 to be managable (say less than 96)
    if np.max(patch_size) <= 96:
        vae_n_layers = 1
    elif np.max(patch_size) <= 384:
        vae_n_layers = 2
    else:
        vae_n_layers = 3

    vae_dict['num_channels'] = base_autoencoder_channels[:vae_n_layers+1]
    vae_dict['attention_levels'] = [False] * (vae_n_layers+1)
    vae_dict['norm_num_groups'] = 16

    downsample_parameters = compute_downsample_parameters(patch_size, vae_n_layers + 1)
    vae_dict['downsample_parameters'] = downsample_parameters
    vae_dict['upsample_parameters'] = list(reversed(downsample_parameters))[:-1]
    return vae_dict


def create_ddpm_dict(dataset_config, spatial_dims):
    median_image_size = dataset_config['median_shape']
    max_image_size = dataset_config['max_shape']
    # For 2D, for each axis, use as size the closest multiple of 2, 3, 5 or 7 by powers of 2, to the corresponding size of max patch size
    valid_2d_sizes = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    patch_size_2d = [min(valid_2d_sizes, key=lambda x: abs(x - size)) for size in max_image_size]
    # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_3d_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_3d_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = patch_size_2d[1:] if spatial_dims == 2 else patch_size_3d

    ddpm_dict = {'spatial_dims': spatial_dims,
                 'in_channels': 8,
                 'out_channels': 8,
                 'num_res_blocks': 2,
                 'use_flash_attention': False,
                }

    # use maximum of 3 autoencoder downsampling layers
    # we want the latent dims to be less than 100 to be managable (say less than 96)
    if np.max(patch_size) <= 96:
        vae_n_layers = 1
    elif np.max(patch_size) <= 384:
        vae_n_layers = 2
    else:
        vae_n_layers = 3

    ddpm_dict['num_channels'] = [256, 512, 768]
    ddpm_dict['attention_levels'] = [False, True, True]
    ddpm_dict['num_head_channels'] = [0, 512, 768]

    vae_down_params = compute_downsample_parameters(patch_size, vae_n_layers + 1)
    latent_size = compute_output_size(patch_size, vae_down_params)
    ddpm_down_params = compute_downsample_parameters(latent_size, 3)

    ddpm_dict['strides'] = [item[0] for item in ddpm_down_params]
    ddpm_dict['kernel_sizes'] = [item[1] for item in ddpm_down_params]
    ddpm_dict['paddings'] = [item[2] for item in ddpm_down_params]

    return ddpm_dict


def create_config_dict(dataset_config, input_channels, n_epochs_multiplier, autoencoder_dict, ddpm_dict):
    median_image_size = dataset_config['median_shape']
    max_image_size = dataset_config['max_shape']
    # For 2D, for each axis, use as size the closest multiple of 2, 3, 5 or 7 by powers of 2, to the corresponding size of max patch size
    valid_2d_sizes = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    patch_size_2d = [min(valid_2d_sizes, key=lambda x: abs(x - size)) for size in max_image_size]
    # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_3d_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_3d_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = patch_size_2d[1:] if autoencoder_dict['spatial_dims'] == 2 else patch_size_3d

    if autoencoder_dict['spatial_dims'] == 2:
        # if 400 < np.max(patch_size):
        #     batch_size = 16
        # elif 300 < np.max(patch_size) < 400:
        #     batch_size = 32
        # elif 200 < np.max(patch_size) < 300:
        #     batch_size = 64
        # else:
        #     batch_size = 128
        batch_size = 24
    else:
        batch_size = 2

    print(f"Patch size {autoencoder_dict['spatial_dims']}D: {patch_size}")

    ae_transformations = {
        "patch_size": patch_size,
        "scaling": True,
        "rotation": True,
        "gaussian_noise": False,
        "gaussian_blur": False,
        "low_resolution": False,
        "brightness": True,
        "contrast": True,
        "gamma": True,
        "mirror": True,
        "dummy_2d": False
    }
    # not using augmentations for ddpm training
    ddpm_transformations = {
        "patch_size": patch_size,
        "scaling": True,
        "rotation": False,
        "gaussian_noise": False,
        "gaussian_blur": False,
        "low_resolution": False,
        "brightness": True,
        "contrast": True,
        "gamma": True,
        "mirror": True,
        "dummy_2d": False
    }

    if autoencoder_dict['spatial_dims'] == 2:
        perceptual_params = {'spatial_dims': 2, 'network_type': "vgg"}
    else:
        perceptual_params = {'spatial_dims': 3, 'network_type': "vgg", 'is_fake_3d': True, 'fake_3d_ratio': 0.2}

    discriminator_params = {'spatial_dims': autoencoder_dict['spatial_dims'], 'in_channels': autoencoder_dict['in_channels'],
                            'out_channels': 1, 'num_channels': 64, 'num_layers_d': 3}

    # adjust the number of epochs based on the training model (2D/3D) and number of training data
    n_epochs = 300 if autoencoder_dict['spatial_dims'] == 3 else 200
    n_epochs = n_epochs * n_epochs_multiplier

    ae_batch_size = batch_size
    ddpm_batch_size = ae_batch_size * 2
    grad_accumulate_step = 1

    # # adjust the batch size and gradient accumulation
    # if autoencoder_dict['spatial_dims'] == 2:
    #     # for 2d use 75% of batch size for both ae and ddpm
    #     ae_batch_size = int(batch_size * 0.75)
    #     ddpm_batch_size = int(batch_size * 0.75)
    #     grad_accumulate_step = 1
    # else:
    #     ae_batch_size = 2
    #     ddpm_batch_size = ae_batch_size * 2
    #     grad_accumulate_step = 1
    #
    # # if batch size and patch size get large, use gradient accumulation
    # if math.prod(patch_size + [ae_batch_size]) > 2e+6:
    #     ae_batch_size //= 2
    #     ddpm_batch_size //= 2
    #     grad_accumulate_step *= 2
    #     print(f"We will use 2 gradient accumulation steps while training in {autoencoder_dict['spatial_dims']}D.")

    config = {
        'input_channels': input_channels,
        'ae_transformations': ae_transformations,
        'ddpm_transformations': ddpm_transformations,
        'ae_batch_size': ae_batch_size,
        'ddpm_batch_size': ddpm_batch_size,
        'n_epochs': n_epochs,
        'val_plot_interval': 10,
        'grad_clip_max_norm': 1,
        'grad_accumulate_step': grad_accumulate_step,
        'oversample_ratio': 0.3,
        'num_workers': 8,
        # 'lr_scheduler': "LinearLR",
        # 'lr_scheduler_params': {'start_factor': 1.0, 'end_factor': 0., 'total_iters': n_epochs},
        # 'lr_scheduler_params': {'start_factor': 1.0, 'end_factor': 0.0001, 'total_iters': int(n_epochs*0.9)},
        'lr_scheduler': None, # "PolynomialLR",
        'lr_scheduler_params': {'total_iters': n_epochs, 'power': 0.9},
        'time_scheduler_params': {'num_train_timesteps': 1000, 'schedule': "scaled_linear_beta", 'beta_start': 0.0015,
                                  'beta_end': 0.0205, 'prediction_type': "epsilon"},
        'ae_learning_rate': 5e-5,
        # 'weight_decay': 3e-5,
        'd_learning_rate': 5e-5,
        'autoencoder_warm_up_epochs': 5,
        'adv_weight': 0.01,
        'perc_weight': 0.5 if autoencoder_dict['spatial_dims'] == 2 else 0.125,
        'kl_weight': 1e-6 if autoencoder_dict['spatial_dims'] == 2 else 1e-7,
        'vae_params': autoencoder_dict,
        'perceptual_params': perceptual_params,
        'discriminator_params': discriminator_params,
        'ddpm_learning_rate': 2e-5,
        'ddpm_params': ddpm_dict
    }
    return config


def save_properties(data_path, patient_id, properties):
    output_path = os.path.join(data_path, f"{patient_id}.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(properties, f)


def extract_spacing(path):
    img = nib.load(path)
    spacing = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))  # voxel spacing from affine
    return spacing


def calculate_median_spacing(image_paths):
    with ProcessPoolExecutor() as executor:
        spacings = list(executor.map(extract_spacing, image_paths))
    return tuple(np.median(spacings, axis=0))


def crop_image_label(image, label=None):
    image_data = image.get_fdata()
    if label is not None:
        label_data = label.get_fdata()
    nonzero_mask = image_data != 0
    nonzero_coords = np.array(np.where(nonzero_mask))
    min_coords = nonzero_coords.min(axis=1)
    max_coords = nonzero_coords.max(axis=1)
    cropped_image = image_data[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ]
    if label is not None:
        cropped_label = label_data[
            min_coords[0]:max_coords[0]+1,
            min_coords[1]:max_coords[1]+1,
            min_coords[2]:max_coords[2]+1
        ]
    log_lines = [f"    Original size: {image_data.shape} - Cropped size: {cropped_image.shape}"]
    if label is not None:
        return cropped_image, cropped_label, log_lines
    else:
        return cropped_image, log_lines


# def resample_image_label(image, target_spacing, label=None):
#     image_data = image.get_fdata()
#     if label is not None:
#         label_data = label.get_fdata()
#     original_spacing = np.sqrt(np.sum(image.affine[:3, :3] ** 2, axis=0))
#     if tuple(original_spacing) != tuple(target_spacing):
#         log_lines = ["    Difference with target spacing. Resampling image...",
#                     f"        Original spacing: {original_spacing} - Final spacing: {target_spacing}"]
#         zoom_factors = original_spacing / target_spacing
#         resampled_image = scipy.ndimage.zoom(image_data, zoom_factors, order=3)  # Trilinear interpolation
#         # clip resampling artifacts
#         resampled_image = np.clip(resampled_image, 0, None)
#         resampled_image = nib.Nifti1Image(resampled_image, image.affine, image.header)
#         if label is not None:
#             resampled_label = scipy.ndimage.zoom(label_data, zoom_factors, order=0)  # Nearest-neighbor
#             resampled_label = nib.Nifti1Image(resampled_label, label.affine, label.header)
#             return resampled_image, resampled_label, log_lines
#         else:
#             return resampled_image, log_lines
#     else:
#         log_lines = ["    No resampling needed"]
#         if label is not None:
#             return image, label, log_lines
#         else:
#             return image, log_lines


def is_anisotropic(spacing, threshold=3.0):
    return (np.max(spacing) / np.min(spacing)) > threshold


def resample_image_label(image, target_spacing, label=None):
    image_data = image.get_fdata()
    if label is not None:
        label_data = label.get_fdata()

    original_spacing = np.sqrt(np.sum(image.affine[:3, :3] ** 2, axis=0))
    zoom_factors = original_spacing / target_spacing
    anisotropic = is_anisotropic(original_spacing)

    log_lines = []
    if not np.allclose(original_spacing, target_spacing):
        log_lines.append("    Difference with target spacing. Resampling image...")
        log_lines.append(f"        Original spacing: {original_spacing} - Final spacing: {target_spacing}")

        if anisotropic:
            lowres_axis = np.argmax(original_spacing)
            order = [3 if i != lowres_axis else 0 for i in range(3)]
        else:
            order = [3, 3, 3]

        resampled_image = image_data
        for axis in range(3):
            if zoom_factors[axis] != 1:
                resampled_image = scipy.ndimage.zoom(resampled_image, zoom=[zoom_factors[axis] if i == axis else 1 for i in range(3)],
                                                     order=order[axis])
        # resampled_image = np.clip(resampled_image, 0, None)
        resampled_image = nib.Nifti1Image(resampled_image, image.affine, image.header)

        if label is not None:
            unique_labels = np.unique(label_data)
            unique_labels = unique_labels[unique_labels != 0]  # exclude background
            one_hot = np.stack([label_data == cls for cls in unique_labels], axis=0)

            if anisotropic:
                interp_orders = [1 if i != np.argmax(original_spacing) else 0 for i in range(3)]
            else:
                interp_orders = [1, 1, 1]

            resampled_channels = []
            for c in range(one_hot.shape[0]):
                channel = one_hot[c].astype(np.float32)
                for axis in range(3):
                    if zoom_factors[axis] != 1:
                        channel = scipy.ndimage.zoom(channel, zoom=[zoom_factors[axis] if i == axis else 1 for i in range(3)],
                                                     order=interp_orders[axis])
                resampled_channels.append(channel)

            argmax_output = np.argmax(np.stack(resampled_channels, axis=0), axis=0)
            resampled_label = np.zeros_like(argmax_output, dtype=np.uint8)
            for idx, cls in enumerate(unique_labels):
                resampled_label[argmax_output == idx] = cls

            resampled_label = nib.Nifti1Image(resampled_label, label.affine, label.header)

            return resampled_image, resampled_label, log_lines
        else:
            return resampled_image, log_lines
    else:
        log_lines.append("    No resampling needed")
        if label is not None:
            return image, label, log_lines
        else:
            return image, log_lines


def normalize_foreground_percentiles(image, lower_p=0., upper_p=99.5):
    """
    Normalize a multi-channel image using percentile-based clipping and scaling.
    Background (value==0) is preserved. Returns normalized image and per-channel min/max.

    Args:
        image: np.ndarray, shape (C, D, H, W) or (C, H, W)
        lower_p: lower percentile (default: 5)
        upper_p: upper percentile (default: 95)

    Returns:
        normalized: np.ndarray, same shape as input
        min_max_per_channel: list of (vmin, vmax) for each channel
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    min_max_per_channel = []

    for c in range(image.shape[0]):
        channel_data = image[c]
        foreground_mask = channel_data > 0

        fg_vals = channel_data[foreground_mask]
        vmin = np.percentile(fg_vals, lower_p)
        vmax = np.percentile(fg_vals, upper_p)

        clipped = np.clip(channel_data, vmin, vmax)
        scaled = (clipped - vmin) / (vmax - vmin)

        normalized[c] = np.where(foreground_mask, scaled, 0.0)
        min_max_per_channel.append((vmin, vmax))

    return normalized, min_max_per_channel


def normalize_zscore_then_minmax(image):
    normalized = np.zeros_like(image, dtype=np.float32)
    min_max_per_channel = []

    for c in range(image.shape[0]):
        channel_data = image[c]

        vmin = np.min(channel_data)
        vmax = np.max(channel_data)

        z_image = (channel_data - np.mean(channel_data)) / np.std(channel_data)
        z_min = np.min(z_image)
        z_max = np.max(z_image)
        normalized[c] = (z_image - z_min) / (z_max - z_min)

        min_max_per_channel.append((vmin, vmax))

    return normalized, min_max_per_channel


def normalize_zscore_then_clip_then_minmax(image):
    normalized = np.zeros_like(image, dtype=np.float32)
    min_max_per_channel = []

    for c in range(image.shape[0]):
        channel_data = image[c]

        vmin = np.min(channel_data)
        vmax = np.max(channel_data)

        z_image = (channel_data - np.mean(channel_data)) / np.std(channel_data)

        z_min = np.percentile(z_image, 0.5)
        z_max = np.percentile(z_image, 99.5)
        clipped = np.clip(z_image, z_min, z_max)

        normalized[c] = (clipped - z_min) / (z_max - z_min)

        min_max_per_channel.append((vmin, vmax))

    return normalized, min_max_per_channel


def compute_laplacian_variance(slice_2d):
    norm_slice = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX)
    norm_slice_uint8 = norm_slice.astype(np.uint8)
    lap = cv2.Laplacian(norm_slice_uint8, cv2.CV_64F)
    return lap.var()


def get_cropped_resampled_shape_channel_min_max_and_quality(path, median_spacing, input_channels):
    img = nib.load(path)
    resampled_image, *_ = resample_image_label(img, target_spacing=median_spacing)
    cropped_image, *_ = crop_image_label(resampled_image)
    if cropped_image.ndim == 3:
        cropped_image = np.expand_dims(cropped_image, axis=-1)
    cropped_image = np.transpose(cropped_image, (3, 2, 1, 0))
    current_input_channels = input_channels if input_channels is not None else [i for i in range(cropped_image.shape[0])]

    high_quality_dict = {'pass': True}
    for c in range(cropped_image.shape[0]):
        if c in current_input_channels:
            # sampled_slices = cropped_image[c, ::5, :, :]  # sample every 5th slice to save time
            # laplacian_variances = [compute_laplacian_variance(sampled_slices[i, :, :])
            #                        for i in range(sampled_slices.shape[0])]
            laplacian_variances = [compute_laplacian_variance(cropped_image[c, i, ...])
                                   for i in range(cropped_image[c].shape[0])]
            avg_laplacian_var = np.mean(laplacian_variances)
            high_quality_dict[f'Channel {c}'] = avg_laplacian_var

    _, min_max_per_channel = normalize_zscore_then_minmax(cropped_image)

    return cropped_image.shape, min_max_per_channel, high_quality_dict


def calculate_dataset_shapes_channel_min_max_and_qualities(image_paths, median_spacing, input_channels, lq_threshold):
    # Prepare function with fixed spacing
    fn = partial(get_cropped_resampled_shape_channel_min_max_and_quality, input_channels=input_channels,
                 median_spacing=median_spacing)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(fn, image_paths))

    shapes, min_max_per_channel, high_quality_dicts = zip(*results)
    median_shape = tuple(np.median(np.array(shapes), axis=0).astype(int))
    min_shape = tuple(np.min(np.array(shapes), axis=0).astype(int))
    max_shape = tuple(np.max(np.array(shapes), axis=0).astype(int))

    min_max_per_channel = np.array(min_max_per_channel)  # Shape: (num_images, num_channels, 2)
    global_channel_min = min_max_per_channel[..., 0].min(axis=0)
    global_channel_max = min_max_per_channel[..., 1].max(axis=0)

    current_input_channels = input_channels if input_channels is not None else [i for i in range(median_shape[0])]

    for c in current_input_channels:

        if lq_threshold is not None:
            current_channel_lp_vars = np.array([item[f'Channel {c}'] for item in high_quality_dicts])
            if lq_threshold == 'otsu':
                print('\nUsing otsu thresholding for laplacian variances to detect low quality images')
                threshold = threshold_otsu(current_channel_lp_vars)
            elif lq_threshold == 'percentile':
                print('\nUsing 5% percentile thresholding for laplacian variances to detect low quality images')
                threshold = np.percentile(current_channel_lp_vars, 5)
            elif isinstance(lq_threshold, int):
                print('\nUsing manual thresholding for laplacian variances to detect low quality images')
                threshold = lq_threshold
            else:
                raise ValueError(
                    "Argument 'lq_threshold' should be one of: None, 'otsu', 'percentile' or an integer value")

            print(f'Threshold: {threshold}')
            for item in high_quality_dicts:
                if item[f'Channel {c}'] < threshold:
                    item['pass'] = False

    return median_shape, min_shape, max_shape, global_channel_min.tolist(), global_channel_max.tolist(), high_quality_dicts


# def get_cropped_and_resampled_image_shape_and_channel_min_max(path, median_spacing):
#     img = nib.load(path)
#     resampled_image, *_ = resample_image_label(img, target_spacing=median_spacing)
#     cropped_image, *_ = crop_image_label(resampled_image)
#     if cropped_image.ndim == 3:
#         cropped_image = np.expand_dims(cropped_image, axis=-1)
#     cropped_image = np.transpose(cropped_image, (3, 2, 1, 0))
#     _, min_max_per_channel = normalize_zscore_then_clip_then_minmax(cropped_image)
#     return cropped_image.shape, min_max_per_channel
#
#
# def calculate_dataset_shapes_and_channel_min_max(image_paths, median_spacing):
#     # Prepare function with fixed spacing
#     fn = partial(get_cropped_and_resampled_image_shape_and_channel_min_max, median_spacing=median_spacing)
#
#     with ProcessPoolExecutor() as executor:
#         results = list(executor.map(fn, image_paths))
#
#     shapes, min_max_per_channel = zip(*results)
#     median_shape = tuple(np.median(np.array(shapes), axis=0).astype(int))
#     min_shape = tuple(np.min(np.array(shapes), axis=0).astype(int))
#     max_shape = tuple(np.max(np.array(shapes), axis=0).astype(int))
#
#     min_max_per_channel = np.array(min_max_per_channel)  # Shape: (num_images, num_channels, 2)
#     global_channel_min = min_max_per_channel[..., 0].min(axis=0)
#     global_channel_max = min_max_per_channel[..., 1].max(axis=0)
#     return median_shape, min_shape, max_shape, global_channel_min.tolist(), global_channel_max.tolist()


def get_sampled_class_locations(label_array, samples_per_slice=50):
    class_locations = {}
    unique_labels = np.unique(label_array)

    for lbl in unique_labels:
        if lbl == 0:
            continue  # skip background

        coords = []
        for z in range(label_array.shape[0]):
            slice_mask = label_array[z] == lbl
            slice_coords = np.argwhere(slice_mask)

            if slice_coords.shape[0] == 0:
                continue  # no voxels for this label in this slice

            if slice_coords.shape[0] > samples_per_slice:
                indices = np.random.choice(slice_coords.shape[0], samples_per_slice, replace=False)
                sampled = slice_coords[indices]
            else:
                sampled = slice_coords

            # Add Z back as the first coordinate
            sampled = [(z, y, x) for y, x in sampled]
            coords.extend(sampled)

        class_locations[int(lbl)] = coords

    return class_locations


def process_patient(patient_id, images_path, labels_path, images_save_path, labels_save_path, median_spacing, median_shape):
    log_lines = [f"Processing {patient_id}..."]

    image_path = os.path.join(images_path, patient_id + '.nii.gz')
    label_path = os.path.join(labels_path, patient_id + '.nii.gz')
    image_save_path = os.path.join(images_save_path, patient_id + '.zarr')
    label_save_path = os.path.join(labels_save_path, patient_id + '.zarr')

    image = nib.load(image_path)
    label = nib.load(label_path)

    resampled_image, resampled_label, resample_log_lines = resample_image_label(image, median_spacing, label)
    cropped_image, cropped_label, crop_log_lines = crop_image_label(resampled_image, resampled_label)
    if cropped_image.ndim == 3:
        cropped_image = np.expand_dims(cropped_image, axis=-1)
    cropped_image = np.transpose(cropped_image, (3, 2, 1, 0))
    cropped_label = np.transpose(cropped_label, (2, 1, 0))
    log_lines.extend(resample_log_lines), log_lines.extend(crop_log_lines)

    normalized_image_data, min_max = normalize_zscore_then_minmax(cropped_image)

    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    # Choose chunk size based on data shape and access pattern
    # This example assumes (C, Z, Y, X) and center cropping in Y, X
    image_chunks = (1, 1) + tuple(median_shape[-2:])
    label_chunks = (1,) + tuple(median_shape[-2:])
    z_image = zarr.open(image_save_path, mode='w')
    z_image.create_dataset(name='image', data=normalized_image_data.astype(np.float32), chunks=image_chunks, compressor=compressor, overwrite=True)
    z_label = zarr.open(label_save_path, mode='w')
    z_label.create_dataset(name='label', data=cropped_label.astype(np.uint8), chunks=label_chunks, compressor=compressor, overwrite=True)

    # np.savez_compressed(image_save_path, data=normalized_image_data.astype(np.float32))
    # np.savez_compressed(label_save_path, data=resampled_label.astype(np.uint8))
    log_lines.append(f"    Saved processed image to {image_save_path}")
    log_lines.append(f"    Saved processed label to {label_save_path}")

    unique_labels = np.unique(cropped_label).tolist()
    class_locations = get_sampled_class_locations(cropped_label, samples_per_slice=50)

    properties = {'class_locations': class_locations, 'min_max': min_max}
    save_properties(images_save_path, patient_id, properties)

    return {
        "patient_id": patient_id,
        "shape": normalized_image_data.shape,
        "labels": [item for item in unique_labels if item != 0],
        "log": "\n".join(log_lines)
    }


def validate_lq_threshold(value):
    if value.lower() == 'auto':
        return 'auto'
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("lq_threshold must be 'auto', an integer, or not set (None).")


def process_patient_wrapper(args):
    return process_patient(*args)


def auto_select_hyperparams(dataset_id, model_fn, config, model_type='2d', init_batch_size=48, init_grad_accum=1):

    assert model_type in ['2d', '3d'], "model_type must be either '2d' or '3d'"

    batch_size = init_batch_size
    grad_accum = init_grad_accum

    min_batch_size = 6 if model_type == '2d' else 1

    preprocessed_dataset_path = glob.glob(os.getenv('medimgen_preprocessed') + f'/Task{dataset_id}*/')[0]
    dataset_folder_name = preprocessed_dataset_path.split('/')[-2]
    results_path = os.path.join(os.getenv('medimgen_results'), dataset_folder_name, model_type, 'autoencoder')

    def try_run(batch_size, grad_accum):
        try:
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

            # Update config with new batch size and grad accum
            test_config = copy.deepcopy(config)
            test_config['ae_batch_size'] = batch_size
            test_config['grad_accumulate_step'] = grad_accum
            test_config['n_epochs'] = 1

            test_config['progress_bar'] = False
            test_config['output_mode'] = 'verbose'
            test_config['results_path'] = results_path
            test_config['load_model_path'] = None

            # Rebuild model and data loaders
            model = model_fn(config=test_config, latent_space_type='vae', print_summary=False)
            transformations = test_config['ae_transformations']
            train_loader, val_loader = get_data_loaders(test_config, dataset_id, 'train-val-test', batch_size, model_type, transformations)

            # Try training for a short time (1 epoch)
            model.train(train_loader=train_loader, val_loader=val_loader)
            print(f"We will use batch size = {batch_size} and grad_accumulate_step = {grad_accum} while training in {model_type}.")
            if os.path.exists(os.path.join(os.getenv('medimgen_results'), dataset_folder_name)):
                shutil.rmtree(os.path.join(os.getenv('medimgen_results'), dataset_folder_name))
            return True

        except RuntimeError as e:
            if os.path.exists(os.path.join(os.getenv('medimgen_results'), dataset_folder_name)):
                shutil.rmtree(os.path.join(os.getenv('medimgen_results'), dataset_folder_name))
            if any([item in str(e) for item in ["CUDA out of memory", "Failed to run torchinfo"]]):
                print(f"[OOM] BatchSize: {batch_size}, GradAccumSteps: {grad_accum}")
                del model
                torch.cuda.empty_cache()
                gc.collect()
                return False
            else:
                raise e

    # Try initial setting
    if try_run(batch_size, grad_accum):
        return batch_size, grad_accum

    if model_type == '2d':
        grad_accum = 2
        while batch_size > min_batch_size:
            batch_size //= 2
            if try_run(batch_size, grad_accum):
                return batch_size, grad_accum

        if try_run(min_batch_size, grad_accum):
            return min_batch_size, grad_accum
        else:
            print(f"Warning! 2D model cannot fit even with batch_size = {batch_size} and grad_accumulate_step = {grad_accum}. You need a bigger GPU!")
            return batch_size, grad_accum

    elif model_type == '3d':
        batch_size //= 2
        grad_accum = 2
        if try_run(batch_size, grad_accum):
            return batch_size, grad_accum
        else:
            print(f"Warning! 3D model cannot fit even with batch_size = {batch_size} and grad_accumulate_step = {grad_accum}. You need a bigger GPU!")
            return batch_size, grad_accum


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset and create configuration file.")
    parser.add_argument("dataset_path", type=str, help="Path to dataset folder")
    parser.add_argument("-c", "--input_channels", required=False, type=validate_channels, default=None,
                        help="List of integers specifying input channel indexes to use. If not specified, all available channels will be used.")
    parser.add_argument("-lqt", "--lq_threshold", required=False, type=validate_lq_threshold, default=None,
                        help="Threshold to separate high/low quality images based on Laplacian variance. "
                             "Accepts 'otsu', 'percentile', an integer value or None (default: None).")

    args = parser.parse_args()
    dataset_path = args.dataset_path
    input_channels = args.input_channels
    lq_threshold = args.lq_threshold

    # given dataset must be in the form TaskXXX_DatasetName and have an 'imagesTr' and 'labelsTr' folder with .nii.gz files
    images_path = os.path.join(dataset_path, 'imagesTr')
    labels_path = os.path.join(dataset_path, 'labelsTr')

    basename = os.path.basename(dataset_path)
    dataset_id = basename.split('_')[0][4:]
    # format the task number to 3 digits with leading zeros
    formatted_task_number = f"{int(dataset_id):03d}"
    # standardized folder name
    standardized_folder_name = f"Task{formatted_task_number}_" + "_".join(basename.split('_')[1:])
    dataset_save_path = os.path.join(os.getenv('medimgen_preprocessed'), standardized_folder_name)

    if os.path.exists(dataset_save_path):
        raise FileExistsError(f"Dataset {os.path.basename(dataset_path)} already exists.")

    images_save_path = os.path.join(dataset_save_path, 'imagesTr')
    labels_save_path = os.path.join(dataset_save_path, 'labelsTr')

    os.makedirs(images_save_path, exist_ok=True)
    os.makedirs(labels_save_path, exist_ok=True)

    image_paths = glob.glob(images_path + "/*.nii.gz")
    patient_ids = sorted([os.path.basename(path).replace('.nii.gz', '') for path in image_paths])

    print(f"\nNumber of patients: {len(patient_ids)}")
    print("\nCalculating median voxel spacing of the whole dataset...")
    median_spacing = calculate_median_spacing(image_paths)
    print(
        "Calculating dataset min and max values, median, min, and max shape after cropping and resampling, and low quality images...")
    dataset_results = calculate_dataset_shapes_channel_min_max_and_qualities(image_paths, median_spacing,
                                                                             input_channels, lq_threshold)
    median_shape, min_shape, max_shape, global_channel_min, global_channel_max, high_quality_dicts = dataset_results
    print(f"\nMedian voxel spacing: {median_spacing}")
    print(f"Median Shape: {median_shape}")
    print(f"Min Shape: {min_shape}")
    print(f"Max Shape: {max_shape}")
    print(f"Min per channel: {global_channel_min}")
    print(f"Max per channel: {global_channel_max}")

    if lq_threshold is not None:
        print(f"\nNumber of low quality images: {np.sum([True for item in high_quality_dicts if not item['pass']])}")
        image_paths = [item for i, item in enumerate(image_paths) if high_quality_dicts[i]['pass']]
        patient_ids = sorted([os.path.basename(path).replace('.nii.gz', '') for path in image_paths])
        print(f"Number of final patients: {len(patient_ids)}\n")

    median_shape_w_channel = median_shape
    median_shape, min_shape, max_shape = median_shape[1:], min_shape[1:], max_shape[1:]

    results = []
    args_list = [(pid, images_path, labels_path, images_save_path, labels_save_path, median_spacing, median_shape)
                 for pid in patient_ids]

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_patient_wrapper, args_list):
            print(result["log"])
            results.append(result)

    all_labels = [lbl for r in results for lbl in r["labels"]]
    unique_labels = sorted(set(all_labels))
    n_patients = len(results)
    n_channels = median_shape_w_channel[0] if len(median_shape_w_channel) == 4 else 1

    # save median image shape, median voxel spacing and n_labels in a dataset.json file
    dataset_config = {
        'median_shape': tuple(int(x) for x in median_shape),
        'min_shape': tuple(int(x) for x in min_shape),
        'max_shape': tuple(int(x) for x in max_shape),
        'median_spacing': [float(x) for x in median_spacing],
        'channel_mins': [float(x) for x in global_channel_min],
        'channel_maxs': [float(x) for x in global_channel_max],
        'n_classes': int(len(unique_labels)),
        'class_labels': [int(c) for c in unique_labels],
        'n_channels': int(n_channels),
        'n_patients': int(n_patients)
    }
    with open(os.path.join(dataset_save_path, 'dataset.json'), 'w') as f:
        json.dump(dataset_config, f, indent=4)

    print(f"\nDataset configuration file saved in {os.path.join(dataset_save_path, 'dataset.json')}")

    print(f"\nConfiguring image generation parameters for Dataset ID: {formatted_task_number}")

    input_channels = input_channels if input_channels is not None \
        else [i for i in range(dataset_config['n_channels'])]
    print(f"Input channels: {input_channels if input_channels is not None else 'all'}")

    if 0.7 * dataset_config['n_patients'] < 100:
        n_epochs_multiplier = 1
    elif 100 < 0.7 * dataset_config['n_patients'] < 500:
        n_epochs_multiplier = 2
    else:
        n_epochs_multiplier = 3

    vae_dict_2d = create_autoencoder_dict(dataset_config, input_channels, spatial_dims=2)
    vae_dict_3d = create_autoencoder_dict(dataset_config, input_channels, spatial_dims=3)

    ddpm_dict_2d = create_ddpm_dict(dataset_config, spatial_dims=2)
    ddpm_dict_3d = create_ddpm_dict(dataset_config, spatial_dims=3)

    config_2d = create_config_dict(dataset_config, input_channels, n_epochs_multiplier, vae_dict_2d, ddpm_dict_2d)
    config_3d = create_config_dict(dataset_config, input_channels, n_epochs_multiplier, vae_dict_3d, ddpm_dict_3d)

    print('\nConfiguring batch size and gradient accumulation steps based on GPU capacity...')
    batch_size_2d, grad_accumulate_step_2d = auto_select_hyperparams(formatted_task_number, AutoEncoder, config_2d, model_type='2d', init_batch_size=24, init_grad_accum=1)
    batch_size_3d, grad_accumulate_step_3d = auto_select_hyperparams(formatted_task_number, AutoEncoder, config_3d, model_type='3d', init_batch_size=2, init_grad_accum=1)

    config_2d['ae_batch_size'] = batch_size_2d
    config_2d['ddpm_batch_size'] = batch_size_2d
    config_2d['grad_accumulate_step'] = grad_accumulate_step_2d

    config_3d['ae_batch_size'] = batch_size_3d
    config_3d['ddpm_batch_size'] = batch_size_3d * 2
    config_3d['grad_accumulate_step'] = grad_accumulate_step_3d

    config = {'2D': config_2d, '3D': config_3d}

    config_save_path = os.path.join(dataset_save_path, 'medimgen_config.yaml')

    # Custom Dumper to avoid anchors and enforce list formatting
    class CustomDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True  # Removes YAML anchors (&id001)

    # Ensure lists stay in flow style
    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    CustomDumper.add_representer(list, represent_list)

    # Save to YAML with all fixes
    with open(config_save_path, "w") as file:
        yaml.dump(config, file, sort_keys=False, Dumper=CustomDumper)

    print(f"Experiment configuration file saved at {config_save_path}")








# def main():
#     parser = argparse.ArgumentParser(description="Automatically configure the parameters of image generation model "
#                                                  "with nnU-Net for dataset ID.")
#     parser.add_argument("-d", "--dataset_id", required=True, type=str, help="Dataset ID")
#     parser.add_argument("-c", "--input_channels", required=False, type=validate_channels, default=None,
#                         help="List of integers specifying input channel indexes to use. If not specified, all available channels will be used.")
#
#     args = parser.parse_args()
#     dataset_id = args.dataset_id
#     input_channels = args.input_channels
#
#     print(f"Configuring image generation parameters for Dataset ID: {dataset_id}")
#     print(f"Input channels: {input_channels if input_channels is not None else 'all'}")
#
#     # configure image generation with nnunet files
#     preprocessed_dataset_path = glob.glob(os.getenv('nnUNet_preprocessed') + f'/Dataset{dataset_id}*/')[0]
#     nnunet_plan_path = os.path.join(preprocessed_dataset_path, 'nnUNetPlans.json')
#     nnunet_data_json_path = os.path.join(preprocessed_dataset_path, 'dataset.json')
#
#     with open(nnunet_plan_path, "r") as file:
#         nnunet_plan = json.load(file)
#
#     with open(nnunet_data_json_path, "r") as file:
#         nnunet_data_json = json.load(file)
#
#     input_channels = input_channels if input_channels is not None \
#         else [i for i in range(len(nnunet_data_json['channel_names']))]
#
#     if 0.7 * nnunet_data_json['numTraining'] < 100:
#         n_epochs_multiplier = 1
#     elif 100 < 0.7 * nnunet_data_json['numTraining'] < 500:
#         n_epochs_multiplier = 2
#     else:
#         n_epochs_multiplier = 3
#
#     configuration_2d = nnunet_plan['configurations']['2d']
#     configuration_3d = nnunet_plan['configurations']['3d_fullres']
#
#     # for item in configuration_2d:
#     #     print(item, configuration_2d[item])
#
#     vae_dict_2d = create_autoencoder_dict(configuration_2d, input_channels, spatial_dims=2)
#     vae_dict_3d = create_autoencoder_dict(configuration_3d, input_channels, spatial_dims=3)
#
#     ddpm_dict_2d = create_ddpm_dict(configuration_2d, spatial_dims=2)
#     ddpm_dict_3d = create_ddpm_dict(configuration_3d, spatial_dims=3)
#
#     config_2d = create_config_dict(configuration_2d, input_channels, n_epochs_multiplier, vae_dict_2d, ddpm_dict_2d)
#     config_3d = create_config_dict(configuration_3d, input_channels, n_epochs_multiplier, vae_dict_3d, ddpm_dict_3d)
#
#     config = {'2D': config_2d, '3D': config_3d}
#
#     # TODO: define gradient accumulation and activation checkpointing based on the required gpu memory usage
#     # TODO: in the training code, define learning rates, learning rate schedulers, optimizer, time scheduler
#
#     # all networks have group norm implemented. To convert group norm to instance norm just use n_groups = n_channels
#
#     # TODO: adapt networks with given activations, normalizations, and convolution sizes
#     # TODO: define all the loss weights and autoencoder warm up epochs
#
#     # for item in config['2D']:
#     #     print(item)
#     #     print(config['2D'][item])
#
#     config_save_path = os.path.join(preprocessed_dataset_path, 'medimgen_config.yaml')
#
#     # Custom Dumper to avoid anchors and enforce list formatting
#     class CustomDumper(yaml.SafeDumper):
#         def ignore_aliases(self, data):
#             return True  # Removes YAML anchors (&id001)
#
#     # Ensure lists stay in flow style
#     def represent_list(dumper, data):
#         return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
#
#     CustomDumper.add_representer(list, represent_list)
#
#     # Save to YAML with all fixes
#     with open(config_save_path, "w") as file:
#         yaml.dump(config, file, sort_keys=False, Dumper=CustomDumper)
#
#     print(f"Configuration file for Dataset {dataset_id} saved at {config_save_path}")
#
#     print("Calculating min-max per patient...")
#     nnunet_2d_path = os.path.join(preprocessed_dataset_path, 'nnUNetPlans_2d')
#     nnunet_3d_path = os.path.join(preprocessed_dataset_path, 'nnUNetPlans_3d_fullres')
#     nnunet_raw_path = glob.glob(os.getenv('nnUNet_raw') + f'/Dataset{dataset_id}*/')[0]
#     nnunet_raw_path = os.path.join(nnunet_raw_path, 'imagesTr')
#
#     file_paths = glob.glob(os.path.join(nnunet_2d_path, "*.npz"))
#     patient_ids = [os.path.basename(fp).replace('.npz', '') for fp in file_paths]
#     if not patient_ids:
#         # we got .b2nd files
#         file_paths = glob.glob(os.path.join(nnunet_2d_path, "*.b2nd"))
#         patient_ids = [os.path.basename(fp).replace('.b2nd', '') for fp in file_paths if '_seg' not in fp]
#     print(f"Found {len(patient_ids)} patients.")
#
#     max_workers = 8
#     total_errors = 0
#     patients_with_errors = []
#
#     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#         func = partial(process_patient, nnunet_raw_path=nnunet_raw_path,
#                                         nnunet_2d_path=nnunet_2d_path,
#                                         nnunet_3d_path=nnunet_3d_path)
#
#         futures = {executor.submit(func, pid): pid for pid in patient_ids}
#
#         for future in concurrent.futures.as_completed(futures):
#             pid = futures[future]
#             try:
#                 result = future.result()
#                 errors = result["errors"]
#                 if errors:
#                     total_errors += errors
#                     patients_with_errors.append((pid, errors))
#                     print(f"⚠️  {pid} had {errors} error(s):")
#                     for key in ["raw_error", "2d_error", "3d_error"]:
#                         if key in result:
#                             print(f"    {key.replace('_', ' ').upper()}: {result[key]}")
#                 else:
#                     print(f"✅ {pid} processed successfully.")
#             except Exception as e:
#                 total_errors += 3  # Assume all 3 failed if outer fails
#                 patients_with_errors.append((pid, 3))
#                 print(f"❌ {pid} failed completely: {e}")
#
#     # Final summary
#     print("\n=== Summary ===")
#     print(f"Total patients processed: {len(patient_ids)}")
#     print(f"Patients with errors: {len(patients_with_errors)}")
#     print(f"Total individual errors: {total_errors}")
#     if patients_with_errors:
#         print("Patients with errors:")
#         for pid, count in patients_with_errors:
#             print(f"  - {pid}: {count} error(s)")



