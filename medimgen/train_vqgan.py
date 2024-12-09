import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.nn import L1Loss
from torchinfo import summary
from torch.cuda.amp import GradScaler, autocast
from monai.utils import set_determinism
from generative.networks.nets import VQVAE, PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, validate_and_cast_config,
                                    print_configuration, create_save_path_dict)
from medimgen.utils import create_gif_from_folder


def train_vqgan(config, train_loader, val_loader, device, save_dict):
    model = VQVAE(**config['model_params'])
    model.to(device)

    img_shape = config['transformations']['resize_shape'] if config['transformations']['resize_shape'] \
        else config['transformations']['patch_size']
    input_shape = (1, config['model_params']['in_channels'], *img_shape)
    summary(model, input_shape, batch_dim=None, depth=3)

    discriminator = PatchDiscriminator(**config['discriminator_params'])
    discriminator.to(device)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(**config['perceptual_params'])
    perceptual_loss.to(device)

    optimizer_g = torch.optim.Adam(params=model.parameters(), lr=config['g_learning_rate'])
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=config['d_learning_rate'])

    scaler_g = GradScaler()
    scaler_d = GradScaler()

    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_epoch_loss_list = []

    disable_prog_bar = config['output_mode'] == 'log' or not config['progress_bar']
    total_start = time.time()
    for epoch in range(config['n_epochs']):
        start = time.time()
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch["image"].to(device)

                # Generator part
                optimizer_g.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    reconstructions, quantization_loss = model(images=images)
                    logits_fake = discriminator(reconstructions.contiguous().float())[-1]

                    recons_loss = l1_loss(reconstructions.float(), images.float())
                    p_loss = perceptual_loss(reconstructions.float(), images.float())
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g = recons_loss + quantization_loss + config['perc_weight'] * p_loss + config['adv_weight'] * generator_loss

                scaler_g.scale(loss_g).backward()
                scaler_g.step(optimizer_g)
                scaler_g.update()

                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    logits_fake = discriminator(reconstructions.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                    loss_d = config['adv_weight'] * discriminator_loss

                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()

                epoch_loss += recons_loss.item()
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

                progress_bar.set_postfix(
                    {
                        "recons_loss": epoch_loss / (step + 1),
                        "gen_loss": gen_epoch_loss / (step + 1),
                        "disc_loss": disc_epoch_loss / (step + 1),
                    }
                )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))

        if disable_prog_bar:
            end = time.time() - start
            print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Train Loss: {epoch_loss / len(train_loader):.4f}")

        if epoch % config['val_interval'] == 0:
            start = time.time()
            model.eval()
            val_epoch_loss = 0
            with tqdm(enumerate(val_loader), total=len(val_loader), ncols=70,
                      disable=disable_prog_bar, file=sys.stdout) as val_progress_bar:
                for step, batch in val_progress_bar:
                    images = batch["image"].to(device)
                    with torch.no_grad():
                        with autocast(enabled=True):
                            reconstructions, _ = model(images=images)
                            recons_loss = l1_loss(reconstructions.float(), images.float())

                    val_epoch_loss += recons_loss.item()
                    val_progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Log validation loss
            if disable_prog_bar:
                end = time.time() - start
                print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                      f"Validation Loss: {val_epoch_loss / len(val_loader):.4f}")

            if config['save_plots']:
                # Create a directory for the current epoch
                epoch_dir = os.path.join(save_dict['plots'], f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)

                image = images[0]
                image = image.unsqueeze(0) if len(image.shape) < 5 else image
                reconstruction = reconstructions[0]
                reconstruction = reconstruction.unsqueeze(0) if len(reconstruction.shape) < 5 else reconstruction
                # Get the number of slices along the desired axis (e.g., the 4th dimension)
                num_slices = image.shape[2]  # Assuming the image is [batch, channel, x, y, z]

                # Normalize the original image volume to 0-1
                image_min, image_max = image.cpu().min(), image.cpu().max()
                normalized_image = (image.cpu() - image_min) / (image_max - image_min)

                # Normalize the reconstruction volume to 0-1
                recon_min, recon_max = reconstruction.cpu().min(), reconstruction.cpu().max()
                normalized_reconstruction = (reconstruction.cpu() - recon_min) / (recon_max - recon_min)

                for slice_idx in range(num_slices):
                    plt.figure(figsize=(4, 2))  # Adjusting the figsize for side-by-side plots

                    # Plot the original image slice
                    slice_image = normalized_image[0, 0, slice_idx, :, :]
                    plt.subplot(1, 2, 1)
                    plt.imshow(slice_image, vmin=0, vmax=1, cmap="gray")
                    plt.title("Image")
                    plt.axis("off")

                    # Plot the reconstruction slice
                    slice_reconstruction = normalized_reconstruction[0, 0, slice_idx, :, :]
                    plt.subplot(1, 2, 2)
                    plt.imshow(slice_reconstruction, vmin=0, vmax=1, cmap="gray")
                    plt.title("Reconstruction")
                    plt.axis("off")

                    # Save the slice comparison with its index in the epoch folder
                    slice_file = os.path.join(epoch_dir, f"slice_{slice_idx}.png")
                    plt.tight_layout()
                    plt.savefig(slice_file, dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()  # Close the figure to free memory

                # Create GIF from folder
                create_gif_from_folder(epoch_dir, os.path.join(epoch_dir, f"epoch_{epoch}.gif"), 80)

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")


def main():
    args_mode = "train_vqgan"
    args = parse_arguments(description="Train a VQGAN Model", args_mode=args_mode)
    config = load_config(args.config)
    config = update_config_with_args(config, args, args_mode)
    config = validate_and_cast_config(config, args_mode)
    mode = "Training"
    model = "vqgan"
    save_dict, save_path = create_save_path_dict(config)
    print_configuration(config, save_path, mode, model=model)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    set_determinism(42)

    train_loader, val_loader = get_data_loaders(config)

    print(f"\nStarting training ddpm model...")
    train_vqgan(config, train_loader, val_loader, device, save_dict)