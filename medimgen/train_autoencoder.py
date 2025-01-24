import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch
import time
import pickle
import matplotlib.pyplot as plt

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from torch.nn import L1Loss
from torchinfo import summary
from torch.cuda.amp import GradScaler, autocast
from monai.utils import set_determinism
from generative.networks.nets import VQVAE, PatchDiscriminator
from generative.losses import PatchAdversarialLoss, PerceptualLoss

from medimgen.data_processing import get_data_loaders
from medimgen.autoencoderkl_with_strides import AutoencoderKL
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, filter_config_by_mode,
                                    print_configuration, create_save_path_dict)
from medimgen.utils import create_gif_from_images, save_all_losses


class AutoEncoder:
    def __init__(self, config, autoencoder, device, save_dict):
        self.config = config
        self.autoencoder = autoencoder
        self.device = device
        self.save_dict = save_dict

        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        if self.config['load_model_path']:
            # update loss_dict from previous training, as we are continuing training
            loss_pickle_path = os.path.join("/".join(self.config['load_model_path'].split('/')[:-2]), 'loss_dict.pkl')
            if os.path.exists(loss_pickle_path):
                with open(loss_pickle_path, 'rb') as file:
                    self.loss_dict = pickle.load(file)
        else:
            self.loss_dict = {'rec_loss': [], 'reg_loss': [], 'gen_loss': [], 'disc_loss': [], 'perc_loss': [],
                              'val_rec_loss': []}

    @staticmethod
    def get_kl_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

    def train_one_epoch(self, epoch, train_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d, scaler_g,
                        scaler_d):
        self.autoencoder.train()
        discriminator.train()
        epoch_loss_dict = {'rec_loss': 0, 'reg_loss': 0, 'gen_loss': 0, 'disc_loss': 0, 'perc_loss': 0}
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=150, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                step_loss_dict = {}

                reconstructions = self.train_generator_step(discriminator, epoch, images, optimizer_g, perceptual_loss,
                                                            scaler_g, step, step_loss_dict, train_loader)
                # Discriminator part
                self.train_discriminator_step(discriminator, epoch, images, optimizer_d, reconstructions, scaler_d,
                                              step, step_loss_dict, train_loader)
                for key in step_loss_dict:
                    epoch_loss_dict[key] += step_loss_dict[key].item()

                progress_bar.set_postfix({key: value / (step + 1) for key, value in epoch_loss_dict.items()})

        epoch_loss_dict = {key: value / len(train_loader) for key, value in epoch_loss_dict.items()}

        if disable_prog_bar:
            end = time.time() - start
            print_string = f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))}"
            for key in epoch_loss_dict:
                print_string +=  f" - {key}: {epoch_loss_dict[key]:.4f}"
            print(print_string)

        for key in epoch_loss_dict:
            self.loss_dict[key].append(epoch_loss_dict[key])

    def train_discriminator_step(self, discriminator, epoch, images, optimizer_d, reconstructions, scaler_d, step,
                                 step_loss_dict, train_loader):
        if epoch >= self.config['autoencoder_warm_up_epochs']:
            for param in discriminator.parameters():
                param.requires_grad = True
            for param in self.autoencoder.parameters():
                param.requires_grad = False

            with autocast(enabled=True):
                logits_fake = discriminator(reconstructions.contiguous().detach())[-1]
                loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                step_loss_dict['disc_loss'] = discriminator_loss * self.config['adv_weight']

            scaler_d.scale(step_loss_dict['disc_loss']).backward()

            if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step + 1) == len(train_loader):
                # gradient clipping
                if self.config['grad_clip_max_norm']:
                    scaler_d.unscale_(optimizer_d)
                    torch.nn.utils.clip_grad_norm_(discriminator.parameters(),
                                                   max_norm=self.config['grad_clip_max_norm'])
                scaler_d.step(optimizer_d)
                scaler_d.update()
                optimizer_d.zero_grad(set_to_none=True)

    def train_generator_step(self, discriminator, epoch, images, optimizer_g, perceptual_loss, scaler_g, step,
                             step_loss_dict, train_loader):
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in self.autoencoder.parameters():
            param.requires_grad = True
        # Generator part
        with autocast(enabled=True):
            if self.config['latent_space_type'] == 'vq':
                reconstructions, quantization_loss = self.autoencoder(images)
                step_loss_dict['reg_loss'] = quantization_loss * self.config['q_weight']
            elif self.config['latent_space_type'] == 'vae':
                reconstructions, z_mu, z_sigma = self.autoencoder(images)
                step_loss_dict['reg_loss'] = self.get_kl_loss(z_mu, z_sigma) * self.config['kl_weight']

            step_loss_dict['rec_loss'] = self.l1_loss(reconstructions.float(), images.float())
            step_loss_dict['perc_loss'] = perceptual_loss(reconstructions.float(), images.float()) * self.config[
                'perc_weight']
            loss_g = step_loss_dict['rec_loss'] + step_loss_dict['perc_loss'] + step_loss_dict['reg_loss']

            if epoch >= self.config['autoencoder_warm_up_epochs']:
                logits_fake = discriminator(reconstructions.contiguous().float())[-1]
                step_loss_dict['gen_loss'] = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False) * \
                                             self.config['adv_weight']
                loss_g += step_loss_dict['gen_loss']
                # print(self.get_kl_loss(z_mu, z_sigma) * self.config['kl_weight'], recons_loss, p_loss * self.config['perc_weight'], generator_loss * self.config['adv_weight'])
                # print(quantization_loss * self.config['q_weight'], recons_loss, p_loss * self.config['perc_weight'], generator_loss * self.config['adv_weight'])
        scaler_g.scale(loss_g).backward()
        if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step + 1) == len(train_loader):
            # gradient clipping
            if self.config['grad_clip_max_norm']:
                scaler_g.unscale_(optimizer_g)
                torch.nn.utils.clip_grad_norm_(self.autoencoder.parameters(),
                                               max_norm=self.config['grad_clip_max_norm'])
            scaler_g.step(optimizer_g)
            scaler_g.update()
            optimizer_g.zero_grad(set_to_none=True)
        return reconstructions

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

        self.loss_dict['val_rec_loss'].append(val_epoch_loss / len(val_loader))

        if return_img_recon:
            image = images[0].unsqueeze(0) if len(images[0].shape) < 5 else images[0]
            reconstruction = reconstructions[0].unsqueeze(0) if len(reconstructions[0].shape) < 5 else \
                reconstructions[0]
            return image, reconstruction

    def save_plots(self, image, reconstruction, gif_output_path):
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
        save_all_losses(self.loss_dict, save_all_losses_path,  self.config['val_interval'])

        loss_pickle_path = os.path.join("/".join(self.save_dict['plots'].split('/')[:-1]), 'loss_dict.pkl')
        with open(loss_pickle_path, 'wb') as file:
            pickle.dump(self.loss_dict, file)

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

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer_g, scheduler=g_lr_scheduler,
                                          discriminator=discriminator, disc_optimizer=optimizer_d, disc_scheduler=d_lr_scheduler,
                                          for_training=True)

        for epoch in range(start_epoch, self.config['n_epochs']):
            self.train_one_epoch(epoch, train_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d, scaler_g, scaler_d)

            if epoch % self.config['val_interval'] == 0:
                image, reconstruction = self.validate_one_epoch(val_loader, return_img_recon=True)
                gif_output_path = os.path.join(self.save_dict['plots'], f"epoch_{epoch}.gif")
                self.save_plots(image, reconstruction, gif_output_path)
                self.save_model(epoch, self.loss_dict['val_rec_loss'][-1], optimizer_g, discriminator, optimizer_d,
                                scheduler=g_lr_scheduler, disc_scheduler=d_lr_scheduler)

            if g_lr_scheduler:
                g_lr_scheduler.step()
                print(f"Adjusting learning rate of generator to {g_lr_scheduler.get_last_lr()[0]:.4e}.")

            if d_lr_scheduler:
                d_lr_scheduler.step()
                print(f"Adjusting learning rate of discriminator to {d_lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")


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
