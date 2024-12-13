import argparse
import logging
import matplotlib
import os
import sys
import yaml

from datetime import datetime


def load_config(config_name):
    """Load default configuration from a YAML file."""
    conf_path = os.path.join(os.getcwd(), 'medimgen', 'configs', config_name + '.yaml') \
        if config_name else "./medimgen/configs/config.yaml"
    with open(conf_path, "r") as file:
        return yaml.safe_load(file)


def parse_arguments(description, args_mode):
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument("--task", required=True, type=str, help="Task identifier")

    if args_mode == 'preprocess_data':
        parser.add_argument("-intensity", action="store_true",
                            help="Enable normalization during the dataset processing.")
    if args_mode != 'preprocess_data':
        parser.add_argument("--config", type=str, help="Configuration file name")
        parser.add_argument("--splitting", nargs=2, type=float, help="Split ratios for train, val")
        parser.add_argument("--channels", nargs='+', type=int, help="List of channel indices or None")

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

        parser.add_argument("--batch_size", type=int, help="Batch size")
        parser.add_argument("--n_epochs", type=int, help="Number of epochs")
        parser.add_argument("--val_interval", type=int, help="Validation interval")

        # Parsing arguments for lr_scheduler
        parser.add_argument("--lr_scheduler", type=str, help="Type of learning rate scheduler")
        parser.add_argument("--start_factor", type=float, help="Start factor for learning rate scheduler")
        parser.add_argument("--end_factor", type=float, help="End factor for learning rate scheduler")
        parser.add_argument("--total_iters", type=int, help="Total iterations for the learning rate scheduler")

    if args_mode == 'train_ddpm':
        parser.add_argument("--load_model_path", type=str, help="Path to checkpoint of pretrained ddpm_model")

        parser.add_argument("--n_train_timesteps", type=int, help="Number of training timesteps")
        parser.add_argument("--n_infer_timesteps", type=int, help="Number of inference timesteps")
        parser.add_argument("--time_scheduler", type=str, help="Time scheduler type")
        parser.add_argument("--learning_rate", type=float, help="Learning rate")
        parser.add_argument("--spatial_dims", type=int, help="Spatial dimensions")
        parser.add_argument("--in_channels", type=int, help="Number of input channels")
        parser.add_argument("--out_channels", type=int, help="Number of output channels")
        parser.add_argument("--num_channels", nargs='+', type=int, help="List of channel numbers for the model")
        parser.add_argument("--attention_levels", nargs='+', type=lambda x: x.lower() == 'true', help="List of attention levels")
        parser.add_argument("--num_head_channels", nargs='+', type=int, help="List of head channel numbers")
        parser.add_argument("--num_res_blocks", type=int, help="Number of residual blocks")
        parser.add_argument("--use_flash_attention", type=lambda x: x.lower() == 'true', help="Use flash attention for speed and memory efficiency")

    if args_mode == 'train_vqgan':
        parser.add_argument("--g_learning_rate", type=float, help="Generator learning rate")
        parser.add_argument("--d_learning_rate", type=float, help="Discriminator learning rate")
        parser.add_argument("--adv_weight", type=float, help="Adversarial loss weight")
        parser.add_argument("--perc_weight", type=float, help="Perceptual loss weight")
        parser.add_argument("--q_weight", type=float, help="Quantization loss weight")

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

        # Model parameters
        parser.add_argument("--model_spatial_dims", type=int, help="Spatial dimensions for model parameters")
        parser.add_argument("--model_in_channels", type=int, help="Number of input channels for the model")
        parser.add_argument("--model_out_channels", type=int, help="Number of output channels for the model")
        parser.add_argument("--num_channels", nargs='+', type=int, help="List of channel numbers for the model")
        parser.add_argument("--num_res_channels", nargs='+', type=int, help="Number of residual channels in the model")
        parser.add_argument("--num_res_layers", type=int, help="Number of residual layers in the model")
        parser.add_argument("--downsample_parameters", nargs='+', type=eval,
                            help="Parameters for downsampling in the model")
        parser.add_argument("--upsample_parameters", nargs='+', type=eval,
                            help="Parameters for upsampling in the model")
        parser.add_argument("--num_embeddings", type=int, help="Number of embeddings for the model")
        parser.add_argument("--embedding_dim", type=int, help="Embedding dimension for the model")
        parser.add_argument("--use_checkpointing", type=lambda x: x.lower() == 'true', help="Use activation checkpointing")

    if args_mode != 'preprocess_data':
        parser.add_argument("--progress_bar", type=lambda x: x.lower() == 'true', help="Use progress bars")
        parser.add_argument("--output_mode", type=str, help="Output mode")
        parser.add_argument("--save_model", type=lambda x: x.lower() == 'true', help="Whether to save the model")
        parser.add_argument("--save_graph", type=lambda x: x.lower() == 'true', help="Whether to save the computation graph")
        parser.add_argument("--save_plots", type=lambda x: x.lower() == 'true', help="Whether to save plots")
        parser.add_argument("--save_profile", type=lambda x: x.lower() == 'true', help="Whether to save the profile")

    return parser.parse_args()


def update_config_with_args(config, args, args_mode):
    config["task"] = str(args.task)
    config["data_path"] = str(os.getenv('DATAPATH'))
    config["save_path"] = str(os.getenv('SAVEPATH'))
    config["config"] = str(args.config)

    # Update config only if arguments were provided
    if args.splitting is not None:
        config["splitting"] = args.splitting
    if args.channels is not None:
        config["channels"] = args.channels

    if args.patch_size is not None:
        config["transformations"]["patch_size"] = args.patch_size
    if args.resize_shape is not None:
        config["transformations"]["resize_shape"] = args.resize_shape
    if args.elastic is not None:
        config["transformations"]["elastic"] = args.elastic
    if args.scaling is not None:
        config["transformations"]["scaling"] = args.scaling
    if args.rotation is not None:
        config["transformations"]["rotation"] = args.rotation
    if args.gaussian_noise is not None:
        config["transformations"]["gaussian_noise"] = args.gaussian_noise
    if args.gaussian_blur is not None:
        config["transformations"]["gaussian_blur"] = args.gaussian_blur
    if args.brightness is not None:
        config["transformations"]["brightness"] = args.brightness
    if args.contrast is not None:
        config["transformations"]["contrast"] = args.contrast
    if args.gamma is not None:
        config["transformations"]["gamma"] = args.gamma
    if args.mirror is not None:
        config["transformations"]["mirror"] = args.mirror
    if args.dummy_2D is not None:
        config["transformations"]["dummy_2D"] = args.dummy_2D

    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.n_epochs is not None:
        config["n_epochs"] = args.n_epochs
    if args.val_interval is not None:
        config["val_interval"] = args.val_interval

    if args.lr_scheduler is not None:
        config["lr_scheduler"] = args.lr_scheduler
    if args.start_factor is not None:
        config["lr_scheduler_params"]["start_factor"] = args.start_factor
    if args.end_factor is not None:
        config["lr_scheduler_params"]["end_factor"] = args.end_factor
    if args.total_iters is not None:
        config["lr_scheduler_params"]["total_iters"] = args.total_iters

    if args_mode == 'train_ddpm':
        if args.load_model_path is not None:
            config["load_model_path"] = args.load_model_path

        if args.n_train_timesteps is not None:
            config["n_train_timesteps"] = args.n_train_timesteps
        if args.n_infer_timesteps is not None:
            config["n_infer_timesteps"] = args.n_infer_timesteps
        if args.time_scheduler is not None:
            config["time_scheduler"] = args.time_scheduler
        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate
        if args.spatial_dims is not None:
            config["model_params"]["spatial_dims"] = args.spatial_dims
        if args.in_channels is not None:
            config["model_params"]["in_channels"] = args.in_channels
        if args.out_channels is not None:
            config["model_params"]["out_channels"] = args.out_channels
        if args.num_channels is not None:
            config["model_params"]["num_channels"] = args.num_channels
        if args.attention_levels is not None:
            config["model_params"]["attention_levels"] = args.attention_levels
        if args.num_head_channels is not None:
            config["model_params"]["num_head_channels"] = args.num_head_channels
        if args.num_res_blocks is not None:
            config["model_params"]["num_res_blocks"] = args.num_res_blocks
        if args.use_flash_attention is not None:
            config["model_params"]["use_flash_attention"] = args.use_flash_attention

    if args_mode == 'train_vqgan':
        if args.g_learning_rate is not None:
            config["g_learning_rate"] = args.g_learning_rate
        if args.d_learning_rate is not None:
            config["d_learning_rate"] = args.d_learning_rate
        if args.adv_weight is not None:
            config["adv_weight"] = args.adv_weight
        if args.perc_weight is not None:
            config["perc_weight"] = args.perc_weight
        if args.q_weight is not None:
            config["q_weight"] = args.q_weight
        # Update perceptual parameters
        if args.perceptual_spatial_dims is not None:
            config["perceptual_params"]["spatial_dims"] = args.spatial_dims
        if args.network_type is not None:
            config["perceptual_params"]["network_type"] = args.network_type
        if args.is_fake_3d is not None:
            config["perceptual_params"]["is_fake_3d"] = args.is_fake_3d
        if args.fake_3d_ratio is not None:
            config["perceptual_params"]["fake_3d_ratio"] = args.fake_3d_ratio
        # Update discriminator parameters
        if args.discriminator_spatial_dims is not None:
            config["discriminator_params"]["spatial_dims"] = args.discriminator_spatial_dims
        if args.discriminator_in_channels is not None:
            config["discriminator_params"]["in_channels"] = args.discriminator_in_channels
        if args.discriminator_out_channels is not None:
            config["discriminator_params"]["out_channels"] = args.discriminator_out_channels
        if args.discriminator_num_channels is not None:
            config["discriminator_params"]["num_channels"] = args.discriminator_num_channels
        if args.discriminator_num_layers_d is not None:
            config["discriminator_params"]["num_layers_d"] = args.discriminator_num_layers_d
        # Update model parameters
        if args.model_spatial_dims is not None:
            config["model_params"]["spatial_dims"] = args.model_spatial_dims
        if args.model_in_channels is not None:
            config["model_params"]["in_channels"] = args.model_in_channels
        if args.model_out_channels is not None:
            config["model_params"]["out_channels"] = args.model_out_channels
        if args.num_channels is not None:
            config["model_params"]["num_channels"] = args.num_channels
        if args.num_res_channels is not None:
            config["model_params"]["num_res_channels"] = args.num_res_channels
        if args.num_res_layers is not None:
            config["model_params"]["num_res_layers"] = args.num_res_layers
        if args.downsample_parameters is not None:
            config["model_params"]["downsample_parameters"] = args.downsample_parameters
        if args.upsample_parameters is not None:
            config["model_params"]["upsample_parameters"] = args.upsample_parameters
        if args.num_embeddings is not None:
            config["model_params"]["num_embeddings"] = args.num_embeddings
        if args.embedding_dim is not None:
            config["model_params"]["embedding_dim"] = args.embedding_dim
        if args.use_checkpointing is not None:
            config["model_params"]["use_checkpointing"] = args.use_checkpointing

    if args.progress_bar is not None:
        config["progress_bar"] = args.progress_bar
    if args.output_mode is not None:
        config["output_mode"] = args.output_mode
    if args.save_model is not None:
        config["save_model"] = args.save_model
    if args.save_graph is not None:
        config["save_graph"] = args.save_graph
    if args.save_plots is not None:
        config["save_plots"] = args.save_plots
    if args.save_profile is not None:
        config["save_profile"] = args.save_profile

    return config


def validate_and_cast_config(config, args_mode):
    """
    Validate and cast configuration values to their correct data types.

    Raises:
        ValueError: If any configuration value does not meet the expected criteria.
    """
    # Ensure the splitting ratios sum to 1
    config["splitting"] = tuple(config["splitting"])
    if sum(config["splitting"]) != 1.0:
        raise ValueError("Splitting ratios must sum to 1.")

    if config["channels"] is not None:
        if not isinstance(config["channels"], list) or not all(
                isinstance(ch, int) and ch >= 0 for ch in config["channels"]):
            raise ValueError("channels must be a list of non-negative integers or None.")

    # Cast flat parameters and validate
    config["transformations"]["patch_size"] = tuple(config["transformations"]["patch_size"])
    if len(config["transformations"]["patch_size"]) != 3 or not all(isinstance(x, int) for x in config["transformations"]["patch_size"]):
        raise ValueError("patch_size must be a tuple of 3 integers.")

    if config["transformations"]["resize_shape"]:
        config["transformations"]["resize_shape"] = tuple(config["transformations"]["resize_shape"])
        if len(config["transformations"]["resize_shape"]) != 3 or not all(isinstance(x, int) for x in config["transformations"]["resize_shape"]):
            raise ValueError("resize_shape must be a tuple of 3 integers.")

    for key in [
        "elastic", "scaling", "rotation", "gaussian_noise",
        "gaussian_blur", "brightness", "contrast", "gamma",
        "mirror", "dummy_2D"
    ]:
        config["transformations"][key] = bool(config["transformations"][key])

    config["batch_size"] = int(config["batch_size"])
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer.")

    config["n_epochs"] = int(config["n_epochs"])
    if config["n_epochs"] <= 0:
        raise ValueError("n_epochs must be a positive integer.")

    config["val_interval"] = int(config["val_interval"])
    if config["val_interval"] <= 0:
        raise ValueError("val_interval must be a positive integer.")

    lr_scheduler_params = config["lr_scheduler_params"]
    # Validate and cast start_factor
    lr_scheduler_params["start_factor"] = float(lr_scheduler_params["start_factor"])
    if not (0.0 < lr_scheduler_params["start_factor"] <= 1.0):
        raise ValueError("start_factor must be a float between 0 (exclusive) and 1 (inclusive).")

    lr_scheduler_params["end_factor"] = float(lr_scheduler_params["end_factor"])
    if not (0.0 <= lr_scheduler_params["end_factor"] <= 1.0):
        raise ValueError("end_factor must be a float between 0 and 1 (inclusive).")

    lr_scheduler_params["total_iters"] = int(lr_scheduler_params["total_iters"])
    if lr_scheduler_params["total_iters"] <= 0:
        raise ValueError("total_iters must be a positive integer.")

    if args_mode == 'train_ddpm':
        config["n_train_timesteps"] = int(config["n_train_timesteps"])
        if config["n_train_timesteps"] <= 0:
            raise ValueError("n_train_timesteps must be a positive integer.")

        config["n_infer_timesteps"] = int(config["n_infer_timesteps"])
        if config["n_infer_timesteps"] <= 0:
            raise ValueError("n_infer_timesteps must be a positive integer.")

        config["time_scheduler"] = str(config["time_scheduler"])

        config["learning_rate"] = float(config["learning_rate"])
        if config["learning_rate"] <= 0:
            raise ValueError("learning_rate must be a positive number.")

        # Validate and cast nested model parameters
        params = config["model_params"]
        params["spatial_dims"] = int(params["spatial_dims"])
        if params["spatial_dims"] not in [2, 3]:
            raise ValueError("spatial_dims must be 2 or 3.")

        params["in_channels"] = int(params["in_channels"])
        if params["in_channels"] <= 0:
            raise ValueError("in_channels must be a positive integer.")

        params["out_channels"] = int(params["out_channels"])
        if params["out_channels"] <= 0:
            raise ValueError("out_channels must be a positive integer.")

        params["num_channels"] = [int(x) for x in params["num_channels"]]
        if not all(x > 0 for x in params["num_channels"]):
            raise ValueError("All values in num_channels must be positive integers.")

        params["attention_levels"] = [bool(x) for x in params["attention_levels"]]
        if len(params["attention_levels"]) != len(params["num_channels"]):
            raise ValueError("attention_levels must have the same length as num_channels.")

        params["num_head_channels"] = [int(x) for x in params["num_head_channels"]]
        if not all(x >= 0 for x in params["num_head_channels"]):
            raise ValueError("All values in num_head_channels must be non-negative integers.")

        params["num_res_blocks"] = int(params["num_res_blocks"])
        if params["num_res_blocks"] <= 0:
            raise ValueError("num_res_blocks must be a positive integer.")

        params["use_flash_attention"] = bool(params["use_flash_attention"])

    if args_mode == 'train_vqgan':
        # Validate and cast global learning parameters
        config["g_learning_rate"] = float(config["g_learning_rate"])
        if config["g_learning_rate"] <= 0:
            raise ValueError("g_learning_rate must be a positive number.")

        config["d_learning_rate"] = float(config["d_learning_rate"])
        if config["d_learning_rate"] <= 0:
            raise ValueError("d_learning_rate must be a positive number.")

        config["adv_weight"] = float(config["adv_weight"])
        if config["adv_weight"] < 0:
            raise ValueError("adv_weight must be a non-negative number.")

        config["perc_weight"] = float(config["perc_weight"])
        if config["perc_weight"] < 0:
            raise ValueError("perc_weight must be a non-negative number.")

        config["q_weight"] = float(config["q_weight"])
        if config["q_weight"] < 0:
            raise ValueError("q_weight must be a non-negative number.")

        # Validate and cast perceptual parameters
        perceptual_params = config["perceptual_params"]
        perceptual_params["spatial_dims"] = int(perceptual_params["spatial_dims"])
        if perceptual_params["spatial_dims"] not in [2, 3]:
            raise ValueError("spatial_dims in perceptual_params must be 2 or 3.")

        valid_nets = ["alex", "vgg", "squeeze", "radimagenet_resnet50", "medicalnet_resnet10_23datasets",
                      "medicalnet_resnet50_23datasets", "resnet50"]
        perceptual_params["network_type"] = str(perceptual_params["network_type"])
        if perceptual_params["network_type"] not in valid_nets:
            raise ValueError(f"network_type must be one of {valid_nets}.")

        perceptual_params["is_fake_3d"] = bool(perceptual_params["is_fake_3d"])

        perceptual_params["fake_3d_ratio"] = float(perceptual_params["fake_3d_ratio"])
        if not (0 <= perceptual_params["fake_3d_ratio"] <= 1):
            raise ValueError("fake_3d_ratio must be a float between 0 and 1.")

        # Validate and cast discriminator parameters
        discriminator_params = config["discriminator_params"]
        discriminator_params["spatial_dims"] = int(discriminator_params["spatial_dims"])
        if discriminator_params["spatial_dims"] not in [2, 3]:
            raise ValueError("spatial_dims in discriminator_params must be 2 or 3.")

        discriminator_params["in_channels"] = int(discriminator_params["in_channels"])
        if discriminator_params["in_channels"] <= 0:
            raise ValueError("in_channels in discriminator_params must be a positive integer.")

        discriminator_params["out_channels"] = int(discriminator_params["out_channels"])
        if discriminator_params["out_channels"] <= 0:
            raise ValueError("out_channels in discriminator_params must be a positive integer.")

        discriminator_params["num_channels"] = int(discriminator_params["num_channels"])
        if discriminator_params["num_channels"] <= 0:
            raise ValueError("num_channels in discriminator_params must be a positive integer.")

        discriminator_params["num_layers_d"] = int(discriminator_params["num_layers_d"])
        if discriminator_params["num_channels"] <= 0:
            raise ValueError("num_layers_d in discriminator_params must be a positive integer.")

        # Validate and cast model parameters
        model_params = config["model_params"]
        model_params["spatial_dims"] = int(model_params["spatial_dims"])
        if model_params["spatial_dims"] not in [2, 3]:
            raise ValueError("spatial_dims in model_params must be 2 or 3.")

        model_params["in_channels"] = int(model_params["in_channels"])
        if model_params["in_channels"] <= 0:
            raise ValueError("in_channels in model_params must be a positive integer.")

        model_params["out_channels"] = int(model_params["out_channels"])
        if model_params["out_channels"] <= 0:
            raise ValueError("out_channels in model_params must be a positive integer.")

        model_params["num_channels"] = [int(x) for x in model_params["num_channels"]]
        if not all(x > 0 for x in model_params["num_channels"]):
            raise ValueError("All values in num_channels must be positive integers.")

        model_params["num_res_channels"] = [int(x) for x in model_params["num_res_channels"]]
        if not all(x > 0 for x in model_params["num_res_channels"]):
            raise ValueError("All values in num_res_channels must be positive integers.")

        model_params["num_res_layers"] = int(model_params["num_res_layers"])
        if model_params["num_res_layers"] <= 0:
            raise ValueError("num_res_layers must be a positive integer.")

        model_params["downsample_parameters"] = [x if isinstance(x, tuple) else tuple(x) for x in model_params["downsample_parameters"]]
        if not all(len(t) == 4 for t in model_params["downsample_parameters"]):
            raise ValueError("Each tuple in downsample_parameters must have exactly 4 elements.")

        model_params["upsample_parameters"] = [tuple(map(int, x)) for x in model_params["upsample_parameters"]]
        if not all(len(t) == 5 for t in model_params["upsample_parameters"]):
            raise ValueError("Each tuple in upsample_parameters must have exactly 5 elements.")

        model_params["num_embeddings"] = int(model_params["num_embeddings"])
        if model_params["num_embeddings"] <= 0:
            raise ValueError("num_embeddings must be a positive integer.")

        model_params["embedding_dim"] = int(model_params["embedding_dim"])
        if model_params["embedding_dim"] <= 0:
            raise ValueError("embedding_dim must be a positive integer.")

        model_params["use_checkpointing"] = bool(model_params["use_checkpointing"])

    config["progress_bar"] = bool(config["progress_bar"])
    config["output_mode"] = str(config["output_mode"])
    if config["output_mode"] not in ["log", "verbose"]:
        raise ValueError("output_mode must be 'log' or 'verbose'.")
    config["save_model"] = bool(config["save_model"])
    config["save_graph"] = bool(config["save_graph"])
    config["save_plots"] = bool(config["save_plots"])
    config["save_profile"] = bool(config["save_profile"])

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


