import matplotlib.pyplot as plt


def create_gif_from_images(images, output_path, duration=100):
    """
    Creates a GIF from a list of PIL.Image objects.

    Args:
        images (list): List of PIL.Image objects.
        output_path (str): The path to save the output GIF.
        duration (int): The duration of each frame in milliseconds (default: 100ms).

    Returns:
        None
    """
    if not images:
        raise ValueError("No images provided to create GIF.")

    # Save the images as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully at {output_path}")


def save_main_losses(epoch_loss_list, val_epoch_loss_list, validation_interval, save_path):
    """
    Saves a plot of training and validation loss per epoch, handling cases where validation loss is logged at intervals.

    Args:
        epoch_loss_list (list): List of training loss values per epoch.
        val_epoch_loss_list (list): List of validation loss values collected at intervals.
        validation_interval (int): Interval at which validation losses are logged (e.g., every 20 epochs).
        save_path (str): Path to save the plot.
    """
    epochs = range(len(epoch_loss_list))  # Epochs for training loss
    val_epochs = list(range(0, len(epoch_loss_list), validation_interval))  # Epochs for validation loss

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, epoch_loss_list, label="Training Loss", marker='o', linestyle='-')
    plt.plot(val_epochs, val_epoch_loss_list, label="Validation Loss", marker='o', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Loss plot saved at {save_path}")


def save_all_losses(gen_loss, disc_loss, train_recon_loss, val_recon_loss, regularization_loss,
                    perceptual_loss, save_path, validation_interval):
    """
    Saves a plot of generator, discriminator, training reconstruction, validation reconstruction,
    and perceptual losses per epoch.

    Args:
        gen_loss (list): Generator loss values per epoch.
        disc_loss (list): Discriminator loss values per epoch.
        train_recon_loss (list): Training reconstruction loss values per epoch.
        val_recon_loss (list): Validation reconstruction loss values per epoch.
        perceptual_loss (list): Perceptual loss values per epoch.
        save_path (str): Path to save the plot.
    """
    epochs = range(len(train_recon_loss))  # Epoch indices
    val_epochs = list(range(0, len(train_recon_loss), validation_interval))

    plt.figure(figsize=(10, 8))
    plt.plot(epochs, gen_loss, label="Generator Loss", marker='o', linestyle='-')
    plt.plot(epochs, disc_loss, label="Discriminator Loss", marker='o', linestyle='-')
    plt.plot(epochs, train_recon_loss, label="Train Reconstruction Loss", marker='o', linestyle='-')
    plt.plot(val_epochs, val_recon_loss, label="Val Reconstruction Loss", marker='o', linestyle='-')
    plt.plot(epochs, regularization_loss, label="Regularization Loss", marker='o', linestyle='-')
    plt.plot(epochs, perceptual_loss, label="Perceptual Loss", marker='o', linestyle='-')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Losses per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Loss plot saved at {save_path}")
