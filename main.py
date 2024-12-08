import warnings
warnings.filterwarnings("ignore")

from configuration import (load_config, parse_arguments, update_config_with_args, validate_and_cast_config,
                           print_configuration, create_save_path_dict)
from training import train_model


def main():
    args = parse_arguments()
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    config = validate_and_cast_config(config)
    mode = args.mode
    model = args.model
    save_dict, save_path = create_save_path_dict(config)
    print_configuration(config, mode, model, save_path)

    # Step 6: Use the mode, model, and configuration in your script logic
    if mode == "train":
        train_model(model, config, save_dict)
    elif mode == "infer":
        pass
        # Add your inference code here
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'infer'.")


if __name__ == "__main__":
    main()