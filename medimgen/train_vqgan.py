import warnings
warnings.filterwarnings("ignore")

import torch
import time
import tqdm

from monai.utils import set_determinism
from torch.nn import L1Loss
from generative.networks.nets import VQVAE, PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, validate_and_cast_config,
                                    print_configuration, create_save_path_dict)


def train_vqgan(config, train_loader, val_loader, device, save_dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    model = VQVAE(**config['model_params'])
    model.to(device)

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)

    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    perceptual_loss = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    perceptual_loss.to(device)

    adv_weight = 0.01
    perc_weight = 0.001

    optimizer_g = torch.optim.Adam(params=model.parameters(), lr=config['g_learning_rate'])
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=config['d_learning_rate'])

    n_epochs = 100
    val_interval = 10
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    val_recon_epoch_loss_list = []
    intermediary_images = []
    n_example_images = 4

    total_start = time.time()
    for epoch in range(n_epochs):
        model.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)
            optimizer_g.zero_grad(set_to_none=True)

            # Generator part
            reconstruction, quantization_loss = model(images=images)
            logits_fake = discriminator(reconstruction.contiguous().float())[-1]

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g = recons_loss + quantization_loss + perc_weight * p_loss + adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            # Discriminator part
            optimizer_d.zero_grad(set_to_none=True)

            logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
            loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
            logits_real = discriminator(images.contiguous().detach())[-1]
            loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
            discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

            loss_d = adv_weight * discriminator_loss

            loss_d.backward()
            optimizer_d.step()

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

        if (epoch + 1) % val_interval == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch["image"].to(device)

                    reconstruction, quantization_loss = model(images=images)

                    # get the first sammple from the first validation batch for visualization
                    # purposes
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:n_example_images, 0])

                    recons_loss = l1_loss(reconstruction.float(), images.float())

                    val_loss += recons_loss.item()

            val_loss /= val_step
            val_recon_epoch_loss_list.append(val_loss)

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")


def main():
    args = parse_arguments(description="Train a VQGAN Model", args_mode="train_vqgan")
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    config = validate_and_cast_config(config)
    mode = "Training"
    model = "vqgan"
    save_dict, save_path = create_save_path_dict(config)
    print_configuration(config, mode, model, save_path)

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