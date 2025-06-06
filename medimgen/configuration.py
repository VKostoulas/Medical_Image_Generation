import json
import os
import ast
import glob
import blosc2
import pickle
import sys
import math
import yaml
import argparse
import logging
import matplotlib
import nibabel as nib
import numpy as np
import concurrent.futures

from datetime import datetime
from functools import partial

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

def load_config(config_path):
    with open(config_path, "r") as file:
        config_file = yaml.safe_load(file)
        return config_file


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


def create_autoencoder_dict(nnunet_config_dict, input_channels, spatial_dims):

    # features_per_stage = nnunet_config_dict['architecture']['arch_kwargs']['features_per_stage']
    kernel_sizes = nnunet_config_dict['architecture']['arch_kwargs']['kernel_sizes']
    strides = nnunet_config_dict['architecture']['arch_kwargs']['strides']

    median_image_size = nnunet_config_dict['median_image_size_in_voxels']

    # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = nnunet_config_dict['patch_size'] if spatial_dims == 2 else patch_size_3d

    base_autoencoder_channels = [32, 64, 128, 128]

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
    # for max image size 512, 3 ae layers --> latent size 64 --> good
    # for max image size 400, 3 ae layers --> latent size 50 --> good
    # for max image size 320, 3 ae layers --> latent size 40 --> good
    # when 3 layers are more than needed? when latent size after 2 downsamplings is <= 64 --> patch_size <= 256
    # for max image size 256, 2 ae layers --> latent size 64 --> good
    # for max image size 200, 2 ae layers --> latent size 50 --> good
    # for max image size 160, 2 ae layers --> latent size 40 --> good
    # for max image size 128, 2 ae layers --> latent size 32 --> good
    # when 2 layers are more than needed? when latent size after 1 downsamplings is <= 32 --> patch_size <= 64
    # for max image size 100, 2 ae layers --> latent size 25 --> good
    # for max image size 64, 1 ae layer --> latent size 32 --> good
    if np.max(patch_size) <= 64:
        vae_n_layers = 1
    elif np.max(patch_size) <= 256:
        vae_n_layers = 2
    else:
        vae_n_layers = 3

    # vae_dict['num_channels'] = features_per_stage[:vae_n_layers+1]
    # vae_dict['attention_levels'] = [False] * (vae_n_layers+1)
    # vae_dict['norm_num_groups'] = vae_dict['num_channels'][0]
    vae_dict['num_channels'] = base_autoencoder_channels[:vae_n_layers+1]
    vae_dict['attention_levels'] = [False] * (vae_n_layers+1)
    vae_dict['norm_num_groups'] = 16

    # nnunet gives you the parameters of the first conv block and then all the downsample parameters
    # For the autoencoder we pass these directly but for the ddpm things are a bit different (see create_ddpm_dict)
    downsample_parameters = [[item1, item2] for item1, item2 in zip(strides[:vae_n_layers+1], kernel_sizes[:vae_n_layers+1])]
    paddings = [[1 if k == 3 else 0 for k in layer] for layer in kernel_sizes[:vae_n_layers+1]]
    downsample_parameters = [item1 + [item2] for item1, item2 in zip(downsample_parameters, paddings)]
    vae_dict['downsample_parameters'] = downsample_parameters
    vae_dict['upsample_parameters'] = list(reversed(downsample_parameters))[:-1]
    return vae_dict


def create_ddpm_dict(nnunet_config_dict, spatial_dims):

    # features_per_stage = nnunet_config_dict['architecture']['arch_kwargs']['features_per_stage']
    kernel_sizes = nnunet_config_dict['architecture']['arch_kwargs']['kernel_sizes']
    strides = nnunet_config_dict['architecture']['arch_kwargs']['strides']

    median_image_size = nnunet_config_dict['median_image_size_in_voxels']

    # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = nnunet_config_dict['patch_size'] if spatial_dims == 2 else patch_size_3d

    ddpm_dict = {'spatial_dims': spatial_dims,
                 'in_channels': 8,
                 'out_channels': 8,
                 'num_res_blocks': 2,
                 'use_flash_attention': False,
                }

    # check create_autoencoder_dict
    if np.max(patch_size) <= 64:
        vae_n_layers = 1
    elif np.max(patch_size) <= 256:
        vae_n_layers = 2
    else:
        vae_n_layers = 3

    # ddpm_dict['num_channels'] = features_per_stage[vae_n_layers:]
    # if len(ddpm_dict['num_channels']) < 2:
    #     raise ValueError("The number of stages must be at least 2.")
    # # First 2 stages without attention, then attention for the rest
    # ddpm_dict['attention_levels'] = [False, False] + [True] * (len(ddpm_dict['num_channels']) - 2)
    ddpm_dict['num_channels'] = [256, 512, 768]
    ddpm_dict['attention_levels'] = [False, True, True]
    ddpm_dict['num_head_channels'] = [0, 512, 768]

    # if len(ddpm_dict['num_channels']) != len(ddpm_dict['attention_levels']):
    #     raise ValueError("num_channels and attention_levels must be of the same length.")
    # ddpm_dict['num_head_channels'] = [channel if use_attention else 0
    #                                      for channel, use_attention in zip(ddpm_dict['num_channels'], ddpm_dict['attention_levels'])]

    # Now the remaining conv parameters from nnunet do not involve the first conv block of the ddpm unet
    # For the first layer of the ddpm unet we always keep the strides at 1, but we take the kernel sizes from the
    # corresponding layer of nnunet. Then we use all the corresponding nnunet layers for the rest of diffusion layers
    ddpm_dict['strides'] = [[1] * spatial_dims] + strides[vae_n_layers+1:vae_n_layers+3]
    ddpm_dict['kernel_sizes'] = [kernel_sizes[vae_n_layers+1]] + kernel_sizes[vae_n_layers+1:vae_n_layers+3]
    ddpm_dict['paddings'] = [[1 if k == 3 else 0 for k in layer] for layer in ddpm_dict['kernel_sizes']]

    return ddpm_dict


def create_config_dict(nnunet_config_dict, input_channels, n_epochs_multiplier, autoencoder_dict, ddpm_dict):

    # features_per_stage = nnunet_config_dict['architecture']['arch_kwargs']['features_per_stage']
    median_image_size = nnunet_config_dict['median_image_size_in_voxels']

    # For 3D, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = nnunet_config_dict['patch_size'] if autoencoder_dict['spatial_dims'] == 2 else patch_size_3d

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
        "scaling": False,
        "rotation": False,
        "gaussian_noise": False,
        "gaussian_blur": False,
        "low_resolution": False,
        "brightness": False,
        "contrast": False,
        "gamma": False,
        "mirror": False,
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

    # adjust the batch size and gradient accumulation
    if autoencoder_dict['spatial_dims'] == 2:
        # for 2d use 75% of batch size for both ae and ddpm
        ae_batch_size = int(nnunet_config_dict['batch_size'] * 0.75)
        ddpm_batch_size = int(nnunet_config_dict['batch_size'] * 0.75)
        grad_accumulate_step = 1
    else:
        ae_batch_size = 2
        ddpm_batch_size = ae_batch_size * 2
        grad_accumulate_step = 1

    # if batch size and patch size get large, use gradient accumulation
    if math.prod(patch_size + [ae_batch_size]) > 2e+6:
        ae_batch_size //= 2
        ddpm_batch_size //= 2
        grad_accumulate_step *= 2
        print(f"We will use 2 gradient accumulation steps while training in {autoencoder_dict['spatial_dims']}D.")

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
        'oversample_ratio': 0.33,
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
        'adv_weight': 0.05,
        'perc_weight': 0.5 if autoencoder_dict['spatial_dims'] == 2 else 0.125,
        'vae_params': autoencoder_dict,
        'perceptual_params': perceptual_params,
        'discriminator_params': discriminator_params,
        'ddpm_learning_rate': 2e-5,
        'ddpm_params': ddpm_dict
    }

    # missing: grad_accumulate_step, q_weight, kl_weight, adv_weight, perc_weight, autoencoder_warm_up_epochs,
    #          latent_space_type: "vae"

    return config


def load_raw_patient_images(raw_path, patient_id):
    # Load all channels for the patient from .nii.gz files
    channel_files = sorted(glob.glob(os.path.join(raw_path, f"{patient_id}_*.nii.gz")))
    if not channel_files:
        raise FileNotFoundError(f"No .nii.gz files found for patient {patient_id} in {raw_path}")

    channels = []
    for ch_file in channel_files:
        nii = nib.load(ch_file)
        arr = nii.get_fdata(dtype=np.float32)  # get_fdata ensures float32 for consistency
        channels.append(arr)

    return np.stack(channels, axis=0)  # Shape: (C, D, H, W) or (C, H, W)


def load_image(data_path, name):
    dparams = {'nthreads': 1}

    image_path_npy = os.path.join(data_path, name + '.npy')
    image_path_npz = os.path.join(data_path, name + '.npz')
    data_b2nd_file = os.path.join(data_path, name + '.b2nd')
    pkl_file = os.path.join(data_path, name + '.pkl')

    if os.path.isfile(image_path_npy):
        image = np.load(image_path_npy, mmap_mode='r')
    elif os.path.isfile(image_path_npz):
        image = np.load(image_path_npz)['data']
    elif os.path.isfile(data_b2nd_file):
        image = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')
    else:
        raise FileNotFoundError(f"No image file found for {name} in {data_path}")

    with open(pkl_file, 'rb') as f:
        properties = pickle.load(f)

    return image, properties


def calculate_min_max_per_channel(image):
    return [(float(np.min(image[i])), float(np.max(image[i]))) for i in range(image.shape[0])]


def save_min_max(data_path, patient_id, min_max):
    output_path = os.path.join(data_path, f"{patient_id}_min_max.pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(min_max, f)


def process_patient(patient_id, nnunet_raw_path, nnunet_2d_path, nnunet_3d_path):
    result = {"patient_id": patient_id, "errors": 0}

    try:
        raw_image = load_raw_patient_images(nnunet_raw_path, patient_id)
        raw_min_max = calculate_min_max_per_channel(raw_image)
        save_min_max(nnunet_raw_path, patient_id, raw_min_max)
        result["raw"] = raw_min_max
    except Exception as e:
        result["raw_error"] = f"{type(e).__name__}: {e}"
        result["errors"] += 1

    try:
        image_2d, _ = load_image(nnunet_2d_path, patient_id)
        min_max_2d = calculate_min_max_per_channel(image_2d)
        save_min_max(nnunet_2d_path, patient_id, min_max_2d)
        result["2d"] = min_max_2d
    except Exception as e:
        result["2d_error"] = f"{type(e).__name__}: {e}"
        result["errors"] += 1

    try:
        image_3d, _ = load_image(nnunet_3d_path, patient_id)
        min_max_3d = calculate_min_max_per_channel(image_3d)
        save_min_max(nnunet_3d_path, patient_id, min_max_3d)
        result["3d"] = min_max_3d
    except Exception as e:
        result["3d_error"] = f"{type(e).__name__}: {e}"
        result["errors"] += 1

    return result


def main():
    parser = argparse.ArgumentParser(description="Automatically configure the parameters of image generation model "
                                                 "with nnU-Net for dataset ID.")
    parser.add_argument("-d", "--dataset_id", required=True, type=str, help="Dataset ID")
    parser.add_argument("-c", "--input_channels", required=False, type=validate_channels, default=None,
                        help="List of integers specifying input channel indexes to use. If not specified, all available channels will be used.")

    args = parser.parse_args()
    dataset_id = args.dataset_id
    input_channels = args.input_channels

    print(f"Configuring image generation parameters for Dataset ID: {dataset_id}")
    print(f"Input channels: {input_channels if input_channels is not None else 'all'}")

    # configure image generation with nnunet files
    preprocessed_dataset_path = glob.glob(os.getenv('nnUNet_preprocessed') + f'/Dataset{dataset_id}*/')[0]
    nnunet_plan_path = os.path.join(preprocessed_dataset_path, 'nnUNetPlans.json')
    nnunet_data_json_path = os.path.join(preprocessed_dataset_path, 'dataset.json')

    with open(nnunet_plan_path, "r") as file:
        nnunet_plan = json.load(file)

    with open(nnunet_data_json_path, "r") as file:
        nnunet_data_json = json.load(file)

    input_channels = input_channels if input_channels is not None \
        else [i for i in range(len(nnunet_data_json['channel_names']))]

    if 0.7 * nnunet_data_json['numTraining'] < 100:
        n_epochs_multiplier = 1
    elif 100 < 0.7 * nnunet_data_json['numTraining'] < 500:
        n_epochs_multiplier = 2
    else:
        n_epochs_multiplier = 3

    configuration_2d = nnunet_plan['configurations']['2d']
    configuration_3d = nnunet_plan['configurations']['3d_fullres']

    # for item in configuration_2d:
    #     print(item, configuration_2d[item])

    vae_dict_2d = create_autoencoder_dict(configuration_2d, input_channels, spatial_dims=2)
    vae_dict_3d = create_autoencoder_dict(configuration_3d, input_channels, spatial_dims=3)

    ddpm_dict_2d = create_ddpm_dict(configuration_2d, spatial_dims=2)
    ddpm_dict_3d = create_ddpm_dict(configuration_3d, spatial_dims=3)

    config_2d = create_config_dict(configuration_2d, input_channels, n_epochs_multiplier, vae_dict_2d, ddpm_dict_2d)
    config_3d = create_config_dict(configuration_3d, input_channels, n_epochs_multiplier, vae_dict_3d, ddpm_dict_3d)

    config = {'2D': config_2d, '3D': config_3d}

    # TODO: define gradient accumulation and activation checkpointing based on the required gpu memory usage
    # TODO: in the training code, define learning rates, learning rate schedulers, optimizer, time scheduler

    # all networks have group norm implemented. To convert group norm to instance norm just use n_groups = n_channels

    # TODO: adapt networks with given activations, normalizations, and convolution sizes
    # TODO: define all the loss weights and autoencoder warm up epochs

    # for item in config['2D']:
    #     print(item)
    #     print(config['2D'][item])

    config_save_path = os.path.join(preprocessed_dataset_path, 'medimgen_config.yaml')

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

    print(f"Configuration file for Dataset {dataset_id} saved at {config_save_path}")

    print("Calculating min-max per patient...")
    nnunet_2d_path = os.path.join(preprocessed_dataset_path, 'nnUNetPlans_2d')
    nnunet_3d_path = os.path.join(preprocessed_dataset_path, 'nnUNetPlans_3d_fullres')
    nnunet_raw_path = glob.glob(os.getenv('nnUNet_raw') + f'/Dataset{dataset_id}*/')[0]
    nnunet_raw_path = os.path.join(nnunet_raw_path, 'imagesTr')

    file_paths = glob.glob(os.path.join(nnunet_2d_path, "*.npz"))
    patient_ids = [os.path.basename(fp).replace('.npz', '') for fp in file_paths]
    if not patient_ids:
        # we got .b2nd files
        file_paths = glob.glob(os.path.join(nnunet_2d_path, "*.b2nd"))
        patient_ids = [os.path.basename(fp).replace('.b2nd', '') for fp in file_paths if '_seg' not in fp]
    print(f"Found {len(patient_ids)} patients.")

    max_workers = 8
    total_errors = 0
    patients_with_errors = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        func = partial(process_patient, nnunet_raw_path=nnunet_raw_path,
                                        nnunet_2d_path=nnunet_2d_path,
                                        nnunet_3d_path=nnunet_3d_path)

        futures = {executor.submit(func, pid): pid for pid in patient_ids}

        for future in concurrent.futures.as_completed(futures):
            pid = futures[future]
            try:
                result = future.result()
                errors = result["errors"]
                if errors:
                    total_errors += errors
                    patients_with_errors.append((pid, errors))
                    print(f"⚠️  {pid} had {errors} error(s):")
                    for key in ["raw_error", "2d_error", "3d_error"]:
                        if key in result:
                            print(f"    {key.replace('_', ' ').upper()}: {result[key]}")
                else:
                    print(f"✅ {pid} processed successfully.")
            except Exception as e:
                total_errors += 3  # Assume all 3 failed if outer fails
                patients_with_errors.append((pid, 3))
                print(f"❌ {pid} failed completely: {e}")

    # Final summary
    print("\n=== Summary ===")
    print(f"Total patients processed: {len(patient_ids)}")
    print(f"Patients with errors: {len(patients_with_errors)}")
    print(f"Total individual errors: {total_errors}")
    if patients_with_errors:
        print("Patients with errors:")
        for pid, count in patients_with_errors:
            print(f"  - {pid}: {count} error(s)")



