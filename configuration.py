import argparse
import logging
import matplotlib
import os
import sys
import yaml

from datetime import datetime


def load_config(config_path="config.yaml"):
    """Load default configuration from a YAML file."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Python Project Configuration")
    # Required arguments
    parser.add_argument("--mode", required=True, type=str, help="Mode of operation (e.g., train, infer)")
    parser.add_argument("--model", required=True, type=str, help="Path to the model file")
    parser.add_argument("--data_path", required=True, type=str, help="Path to the data")
    parser.add_argument("--save_path", required=True, type=str, help="Path to save outputs")
    # Optional arguments (no defaults here)
    parser.add_argument("--center_crop_size", nargs=3, type=int, help="Center crop size")
    parser.add_argument("--resized_size", nargs=3, type=int, help="Resized size")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--n_train_timesteps", type=int, help="Number of training timesteps")
    parser.add_argument("--n_infer_timesteps", type=int, help="Number of inference timesteps")
    parser.add_argument("--time_scheduler", type=str, help="Time scheduler type")
    parser.add_argument("--n_epochs", type=int, help="Number of epochs")
    parser.add_argument("--val_interval", type=int, help="Validation interval")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--spatial_dims", type=int, help="Spatial dimensions")
    parser.add_argument("--in_channels", type=int, help="Number of input channels")
    parser.add_argument("--out_channels", type=int, help="Number of output channels")
    parser.add_argument("--num_channels", nargs='+', type=int, help="List of channel numbers for the model")
    parser.add_argument("--attention_levels", nargs='+', type=lambda x: x.lower() == 'true',
                        help="List of attention levels")
    parser.add_argument("--num_head_channels", nargs='+', type=int, help="List of head channel numbers")
    parser.add_argument("--num_res_blocks", type=int, help="Number of residual blocks")
    parser.add_argument("--progress_bar", type=lambda x: x.lower() == 'true', help="Use progress bars")
    parser.add_argument("--output_mode", type=str, help="Output mode")
    parser.add_argument("--save_model", type=lambda x: x.lower() == 'true', help="Whether to save the model")
    parser.add_argument("--save_graph", type=lambda x: x.lower() == 'true',
                        help="Whether to save the computation graph")
    parser.add_argument("--save_plots", type=lambda x: x.lower() == 'true', help="Whether to save plots")
    parser.add_argument("--save_profile", type=lambda x: x.lower() == 'true', help="Whether to save the profile")

    return parser.parse_args()


def update_config_with_args(config, args):
    config["data_path"] = str(args.data_path)
    config["save_path"] = str(args.save_path)

    # Update config only if arguments were provided
    if args.center_crop_size is not None:
        config["center_crop_size"] = args.center_crop_size
    if args.resized_size is not None:
        config["resized_size"] = args.resized_size
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.n_train_timesteps is not None:
        config["n_train_timesteps"] = args.n_train_timesteps
    if args.n_infer_timesteps is not None:
        config["n_infer_timesteps"] = args.n_infer_timesteps
    if args.time_scheduler is not None:
        config["time_scheduler"] = args.time_scheduler
    if args.n_epochs is not None:
        config["n_epochs"] = args.n_epochs
    if args.val_interval is not None:
        config["val_interval"] = args.val_interval
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


def validate_and_cast_config(config):
    """
    Validate and cast configuration values to their correct data types.

    Raises:
        ValueError: If any configuration value does not meet the expected criteria.
    """
    # Cast flat parameters and validate
    config["center_crop_size"] = tuple(config["center_crop_size"])
    if len(config["center_crop_size"]) != 3 or not all(isinstance(x, int) for x in config["center_crop_size"]):
        raise ValueError("center_crop_size must be a tuple of 3 integers.")

    config["resized_size"] = tuple(config["resized_size"])
    if len(config["resized_size"]) != 3 or not all(isinstance(x, int) for x in config["resized_size"]):
        raise ValueError("resized_size must be a tuple of 3 integers.")

    config["batch_size"] = int(config["batch_size"])
    if config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer.")

    config["n_train_timesteps"] = int(config["n_train_timesteps"])
    if config["n_train_timesteps"] <= 0:
        raise ValueError("n_train_timesteps must be a positive integer.")

    config["n_infer_timesteps"] = int(config["n_infer_timesteps"])
    if config["n_infer_timesteps"] <= 0:
        raise ValueError("n_infer_timesteps must be a positive integer.")

    config["time_scheduler"] = str(config["time_scheduler"])

    config["n_epochs"] = int(config["n_epochs"])
    if config["n_epochs"] <= 0:
        raise ValueError("n_epochs must be a positive integer.")

    config["val_interval"] = int(config["val_interval"])
    if config["val_interval"] <= 0:
        raise ValueError("val_interval must be a positive integer.")

    config["learning_rate"] = float(config["learning_rate"])
    if config["learning_rate"] <= 0:
        raise ValueError("learning_rate must be a positive number.")

    config["progress_bar"] = bool(config["progress_bar"])

    config["output_mode"] = str(config["output_mode"])
    if config["output_mode"] not in ["log", "verbose"]:
        raise ValueError("output_mode must be 'log' or 'verbose'.")

    config["save_model"] = bool(config["save_model"])
    config["save_graph"] = bool(config["save_graph"])
    config["save_plots"] = bool(config["save_plots"])
    config["save_profile"] = bool(config["save_profile"])

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

    return config


def print_configuration(config, mode, model, space_from_start=40):
    """
    Print the mode, model, and configuration parameters in a perfectly aligned format.
    Args:
        config (dict): Configuration dictionary.
        mode (str): Mode of operation (e.g., "train").
        model (str): Model type (e.g., "ddpm").
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

    # Print the Mode, Model, and Data Path
    print(f"Mode{' ' * (space_from_start - len('Mode'))}{mode}")
    print(f"Model{' ' * (space_from_start - len('Model'))}{model}")
    print(f"Data Path{' ' * (space_from_start - len('Data Path'))}{config['data_path']}")
    print(f"Save Path{' ' * (space_from_start - len('Save Path'))}{config['save_path']}")
    print("\nParameters:\n" + "-" * (space_from_start * 3))

    # Print each parameter with aligned values
    for i, (key, value) in enumerate(sorted(flat_config.items())):
        if key not in ["data_path", "save_path"]:  # Skip already printed keys
            spaces = " " * (space_from_start - len(key))  # Calculate spaces for alignment
            if i == len(flat_config) - 1:
                print(f"{key}{spaces}{value}\n{'=' * (space_from_start * 3)}")
            else:
                print(f"{key}{spaces}{value}")

    # print("=" * (space_from_start * 3))


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
        return save_dict

    # Generate a timestamped save directory
    timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    save_path = os.path.join(config['save_path'], timestamp)
    os.makedirs(save_path, exist_ok=True)

    # Setup logging only if mode is 'log'
    if config["output_mode"] == "log":
        log_file_path = os.path.join(save_path, 'log_file.txt')
        setup_logging(log_file_path)

    # Create subdirectories for enabled save types
    for dir_name, should_save in save_dict.items():
        if should_save:
            temp_save_path = os.path.join(save_path, dir_name)
            os.makedirs(temp_save_path, exist_ok=True)
            save_dict[dir_name] = temp_save_path
        else:
            save_dict[dir_name] = False

    return save_dict


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


# Example:
# python main.py --mode train --model ddpm --data_path /path/to/dataset \
#     --center_crop_size 160 200 155 \
#     --resized_size 64 64 64 \
#     --model_params.num_channels 128 256 512 \
#     --model_params.attention_levels true false true

