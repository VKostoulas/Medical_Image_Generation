import argparse
import logging
import matplotlib
import os
import sys
import yaml

from datetime import datetime


def load_config(config_name):
    """Load default configuration from a YAML file."""
    if config_name:
        final_config_name = config_name
    else:
        final_config_name = 'config'
    conf_path = os.path.join(os.getcwd(), 'medimgen', 'configs', final_config_name + '.yaml')
    with open(conf_path, "r") as file:
        config_file = yaml.safe_load(file)
        config_file['config'] = final_config_name
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
    parser.add_argument("--save_model", type=lambda x: x.lower() == 'true', help="Whether to save the model")
    parser.add_argument("--save_graph", type=lambda x: x.lower() == 'true',
                        help="Whether to save the computation graph")
    parser.add_argument("--save_plots", type=lambda x: x.lower() == 'true', help="Whether to save plots")
    parser.add_argument("--save_profile", type=lambda x: x.lower() == 'true', help="Whether to save the profile")


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
            "use_checkpointing", "use_convtranspose"
        ]:
            if getattr(args, f"vae_{key}", None) is not None:
                config["vae_params"][key] = getattr(args, f"vae_{key}")

    # Additional arguments
    for key in ["progress_bar", "output_mode", "save_model", "save_graph", "save_plots", "save_profile"]:
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
    save_dict = {
        'checkpoints': config['save_model'],
        'graph': config['save_graph'],
        'plots': config['save_plots'],
        'profile': config['save_profile']
    }

    # If no saving is enabled, return the dictionary without creating directories
    if not any(save_dict.values()):
        return save_dict, ''

    # Generate a timestamped save directory
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = os.path.join(config['save_path'], timestamp)

    # Setup logging only if mode is 'log'
    if config["output_mode"] == "log":
        log_file_path = os.path.join(save_path, 'log_file.txt')
        setup_logging(log_file_path)

    # Create subdirectories for enabled save types
    for dir_name, should_save in save_dict.items():
        if should_save:
            temp_save_path = os.path.join(save_path, dir_name)
            save_dict[dir_name] = temp_save_path
        else:
            save_dict[dir_name] = False

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


