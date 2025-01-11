import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import time
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.nn import L1Loss
from torchinfo import summary
from torch.cuda.amp import GradScaler, autocast
from monai.utils import set_determinism
from generative.networks.nets import VQVAE, AutoencoderKL, PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, filter_config_by_mode,
                                    print_configuration, create_save_path_dict)
from medimgen.utils import create_gif_from_images, save_main_losses, save_all_losses


class AutoEncoder:
    def __init__(self, config, autoencoder, device, save_dict):
        self.config = config
        self.autoencoder = autoencoder
        self.device = device
        self.save_dict = save_dict

        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

    @staticmethod
    def get_kl_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    def train_one_epoch(self, epoch, train_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d, scaler_g,
                        scaler_d):
        self.autoencoder.train()
        discriminator.train()
        epoch_recon_loss = 0
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_perc_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=150, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in progress_bar:
                images = batch["image"].to(self.device)

                # Generator part
                optimizer_g.zero_grad(set_to_none=True)
                with autocast(enabled=True):
                    if self.config['latent_space_type'] == 'vq':
                        reconstructions, quantization_loss = self.autoencoder(images)
                        loss_g = quantization_loss * self.config['q_weight']
                    elif self.config['latent_space_type'] == 'vae':
                        reconstructions, z_mu, z_sigma = self.autoencoder(images)
                        loss_g = self.get_kl_loss(z_mu, z_sigma) * self.config['kl_weight']

                    recons_loss = self.l1_loss(reconstructions.float(), images.float())
                    p_loss = perceptual_loss(reconstructions.float(), images.float()) * self.config['perc_weight']
                    loss_g += recons_loss + p_loss

                    if epoch >= self.config['autoencoder_warm_up_epochs']:
                        logits_fake = discriminator(reconstructions.contiguous().float())[-1]
                        generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False) * self.config['adv_weight']
                        loss_g += generator_loss
                        # print(self.get_kl_loss(z_mu, z_sigma) * self.config['kl_weight'], recons_loss, p_loss * self.config['perc_weight'], generator_loss * self.config['adv_weight'])
                        # print(quantization_loss * self.config['q_weight'], recons_loss, p_loss * self.config['perc_weight'], generator_loss * self.config['adv_weight'])

                scaler_g.scale(loss_g).backward()
                # scaler_g.unscale_(optimizer_g)
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                scaler_g.step(optimizer_g)
                scaler_g.update()

                # Discriminator part
                if epoch >= self.config['autoencoder_warm_up_epochs']:
                    optimizer_d.zero_grad(set_to_none=True)
                    with autocast(enabled=True):
                        logits_fake = discriminator(reconstructions.contiguous().detach())[-1]
                        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                        logits_real = discriminator(images.contiguous().detach())[-1]
                        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                        loss_d = discriminator_loss * self.config['adv_weight']

                    scaler_d.scale(loss_d).backward()
                    # scaler_d.unscale_(optimizer_d)
                    # torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                    scaler_d.step(optimizer_d)
                    scaler_d.update()

                epoch_recon_loss += recons_loss.item()
                epoch_perc_loss += p_loss.item()
                if epoch >= self.config['autoencoder_warm_up_epochs']:
                    epoch_gen_loss += generator_loss.item()
                    epoch_disc_loss += loss_d.item()

                progress_bar.set_postfix({
                    "recons_loss": epoch_recon_loss / (step + 1),
                    "gen_loss": epoch_gen_loss / (step + 1),
                    "disc_loss": epoch_disc_loss / (step + 1),
                    "perc_loss": epoch_perc_loss / (step + 1)
                })

        if disable_prog_bar:
            end = time.time() - start
            print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
            f"Train Loss: {epoch_recon_loss / len(train_loader):.4f}")

        return (epoch_recon_loss / len(train_loader), epoch_gen_loss / len(train_loader),
                epoch_disc_loss / len(train_loader), epoch_perc_loss / len(train_loader))

    def validate_one_epoch(self, val_loader, return_img_recon=False):
        self.autoencoder.eval()
        val_epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as val_progress_bar:
            for step, batch in val_progress_bar:
                images = batch["image"].to(self.device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        reconstructions, *_ = self.autoencoder(images)
                        recons_loss = self.l1_loss(reconstructions.float(), images.float())

                val_epoch_loss += recons_loss.item()
                val_progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

        if disable_prog_bar:
            end = time.time() - start
            print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Validation Loss: {val_epoch_loss / len(val_loader):.4f}")

        if return_img_recon:
            image = images[0].unsqueeze(0) if len(images[0].shape) < 5 else images[0]
            reconstruction = reconstructions[0].unsqueeze(0) if len(reconstructions[0].shape) < 5 else \
                reconstructions[0]
            return val_epoch_loss / len(val_loader), image, reconstruction
        else:
            return val_epoch_loss / len(val_loader)

    def save_plots(self, image, reconstruction, gif_output_path, recon_loss_list=None, val_loss_list=None,
                   gen_loss_list=None, disc_loss_list=None, perc_loss_list=None):
        if not os.path.exists(self.save_dict['plots']):
            os.makedirs(self.save_dict['plots'], exist_ok=True)

        num_slices = image.shape[2]  # Assuming the image is [batch, channel, x, y, z]
        gif_images = []

        for slice_idx in range(num_slices):
            plt.figure(figsize=(4, 2))  # Adjusting the figsize for side-by-side plots
            # Plot the original image slice
            slice_image = image.cpu()[0, 0, slice_idx, :, :]
            plt.subplot(1, 2, 1)
            plt.imshow(slice_image, vmin=0, vmax=1, cmap="gray")
            plt.title("Image")
            plt.axis("off")
            # Plot the reconstruction slice
            slice_reconstruction = reconstruction.cpu()[0, 0, slice_idx, :, :]
            plt.subplot(1, 2, 2)
            plt.imshow(slice_reconstruction, vmin=0, vmax=1, cmap="gray")
            plt.title("Reconstruction")
            plt.axis("off")

            buffer = BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            buffer.seek(0)
            gif_image = Image.open(buffer).copy()  # Fully load the image into memory
            gif_images.append(gif_image)
            buffer.close()

        # Create GIF from the list of images
        create_gif_from_images(gif_images, gif_output_path)

        save_all_losses_path = os.path.join(self.save_dict['plots'], f"loss.png")
        save_all_losses(gen_loss_list, disc_loss_list, recon_loss_list, val_loss_list, perc_loss_list, save_all_losses_path,  self.config['val_interval'])

    def save_model(self, epoch, validation_loss, optimizer, discriminator, disc_optimizer, scheduler=None,
                   disc_scheduler=None):
        if not os.path.exists(self.save_dict['checkpoints']):
            os.makedirs(self.save_dict['checkpoints'], exist_ok=True)

        last_checkpoint_path = os.path.join(self.save_dict['checkpoints'], 'last_model.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'network_state_dict': self.autoencoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        checkpoint['discriminator_state_dict'] = discriminator.state_dict()
        checkpoint['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
        if disc_scheduler:
            checkpoint['disc_scheduler_state_dict'] = disc_scheduler.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

        best_checkpoint_path = os.path.join(self.save_dict['checkpoints'], 'best_model.pth')
        if os.path.isfile(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path)
            best_loss = best_checkpoint.get('validation_loss', float('inf'))
            if validation_loss < best_loss:
                torch.save(checkpoint, best_checkpoint_path)
        else:
            torch.save(checkpoint, best_checkpoint_path)

    def load_model(self, load_model_path, optimizer=None, scheduler=None, discriminator=None, disc_optimizer=None,
                   disc_scheduler=None, for_training=False):
        print(f'Loading model from {load_model_path}...')
        checkpoint = torch.load(load_model_path)
        self.autoencoder.load_state_dict(checkpoint['network_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if discriminator and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        if disc_optimizer and 'disc_optimizer_state_dict' in checkpoint:
            disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])

        if disc_scheduler and 'disc_scheduler_state_dict' in checkpoint:
            disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])

        if for_training:
            return checkpoint['epoch']

    def train(self, train_loader, val_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d,
              g_lr_scheduler=None, d_lr_scheduler=None):
        scaler_g = GradScaler()
        scaler_d = GradScaler()
        total_start = time.time()
        start_epoch = 0
        recon_loss_list = []
        gen_loss_list = []
        disc_loss_list = []
        perc_loss_list = []
        val_loss_list = []

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer_g, scheduler=g_lr_scheduler,
                                          discriminator=discriminator, disc_optimizer=optimizer_d, disc_scheduler=d_lr_scheduler,
                                          for_training=True)

        for epoch in range(start_epoch, self.config['n_epochs']):
            recon_loss, gen_loss, disc_loss, perc_loss = self.train_one_epoch(epoch, train_loader, discriminator, perceptual_loss,
                                                                              optimizer_g, optimizer_d, scaler_g, scaler_d)
            recon_loss_list.append(recon_loss)
            gen_loss_list.append(gen_loss)
            disc_loss_list.append(disc_loss)
            perc_loss_list.append(perc_loss)

            if epoch % self.config['val_interval'] == 0:
                if self.save_dict['plots']:
                    val_loss, image, reconstruction = self.validate_one_epoch(val_loader, return_img_recon=True)
                    val_loss_list.append(val_loss)
                    gif_output_path = os.path.join(self.save_dict['plots'], f"epoch_{epoch}.gif")
                    self.save_plots(image, reconstruction, gif_output_path, recon_loss_list=recon_loss_list,
                                    val_loss_list=val_loss_list, gen_loss_list=gen_loss_list, disc_loss_list=disc_loss_list,
                                    perc_loss_list=perc_loss_list)
                else:
                    val_loss = self.validate_one_epoch(val_loader)
                    val_loss_list.append(val_loss)

                if self.save_dict['checkpoints']:
                    self.save_model(epoch, val_loss, optimizer_g, discriminator, optimizer_d, scheduler=g_lr_scheduler,
                                    disc_scheduler=d_lr_scheduler)

            if g_lr_scheduler:
                g_lr_scheduler.step()
                print(f"Adjusting learning rate of generator to {g_lr_scheduler.get_last_lr()[0]:.4e}.")

            if d_lr_scheduler:
                d_lr_scheduler.step()
                print(f"Adjusting learning rate of discriminator to {d_lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Training completed in {total_time:.2f} seconds.")


def main():
    args_mode = "train_autoencoder"
    args = parse_arguments(description="Train an Autoencoder Model to reconstruct the input", args_mode=args_mode)
    config = load_config(args.config)
    config = update_config_with_args(config, args, args_mode)
    config = filter_config_by_mode(config, args_mode)
    save_dict, save_path = create_save_path_dict(config)
    print_configuration(config, save_path, "Training", model="autoencoder")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    set_determinism(42)

    train_loader, val_loader = get_data_loaders(config)

    img_shape = config['transformations']['resize_shape'] if config['transformations']['resize_shape'] \
        else config['transformations']['patch_size']

    if config['latent_space_type'] == 'vq':
        autoencoder = VQVAE(**config['vqvae_params']).to(device)
        input_shape = (1, config['vqvae_params']['in_channels'], *img_shape)
    elif config['latent_space_type'] == 'vae':
        autoencoder = AutoencoderKL(**config['vae_params']).to(device)
        input_shape = (1, config['vae_params']['in_channels'], *img_shape)
    else:
        raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")

    summary(autoencoder, input_shape, batch_dim=None, depth=3)

    discriminator = PatchDiscriminator(**config['discriminator_params']).to(device)
    summary(discriminator, input_shape, batch_dim=None, depth=3)

    perceptual_loss = PerceptualLoss(**config['perceptual_params']).to(device)
    summary(perceptual_loss, [input_shape, input_shape], batch_dim=None, depth=3)

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=config['g_learning_rate'])
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=config['d_learning_rate'])

    if config["lr_scheduler"]:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["lr_scheduler"])  # Get the class dynamically
        g_lr_scheduler = scheduler_class(optimizer_g, **config["lr_scheduler_params"])
        d_lr_scheduler = scheduler_class(optimizer_d, **config["lr_scheduler_params"])
    else:
        g_lr_scheduler = None
        d_lr_scheduler = None

    model = AutoEncoder(config=config, autoencoder=autoencoder, device=device, save_dict=save_dict)

    print(f"\nStarting training autoencoder model...")
    model.train(train_loader=train_loader, val_loader=val_loader, discriminator=discriminator,
                perceptual_loss=perceptual_loss, optimizer_g=optimizer_g, optimizer_d=optimizer_d,
                g_lr_scheduler=g_lr_scheduler, d_lr_scheduler=d_lr_scheduler)
