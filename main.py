import warnings
warnings.filterwarnings("ignore")

from configuration import (load_config, parse_arguments, update_config_with_args, validate_and_cast_config,
                           print_configuration, create_save_path_dict)
from training import train_model


def main():
    config = load_config()
    args = parse_arguments()
    config = update_config_with_args(config, args)
    config = validate_and_cast_config(config)
    mode = args.mode
    model = args.model
    save_dict = create_save_path_dict(config)
    print_configuration(config, mode, model)

    # Step 6: Use the mode, model, and configuration in your script logic
    if mode == "train":
        print(f"\nStarting training {model} model...")
        train_model(model, config, save_dict)
    elif mode == "infer":
        print("Starting inference...")
        # Add your inference code here
    else:
        raise ValueError("Invalid mode. Choose 'train' or 'infer'.")


if __name__ == "__main__":
    main()