import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from torchinfo import summary
from monai.utils import set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import VQVAE

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, validate_and_cast_config,
                                    print_configuration, create_save_path_dict)
from medimgen.utils import create_gif_from_images, save_main_losses


class LDM:
    def __init__(self, config, network, vqvae, inferer, scheduler, device, save_dict):
        self.config = config
        self.network = network
        self.vqvae = vqvae
        self.inferer = inferer
        self.scheduler = scheduler
        self.device = device
        self.save_dict = save_dict

    def train_one_epoch(self, epoch, train_loader, z_shape, optimizer, scaler):
        self.network.train()
        self.vqvae.eval()
        epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    noise = torch.randn(*z_shape).to(self.device)
                    timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],),
                                              device=images.device).long()
                    noise_pred = self.inferer(inputs=images, autoencoder_model=self.vqvae, diffusion_model=self.network, noise=noise, timesteps=timesteps)
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        # Log epoch loss
        if disable_prog_bar:
            end = time.time() - start
            print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Train Loss: {epoch_loss / len(train_loader):.4f}")

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader, z_shape):
        self.network.eval()
        self.vqvae.eval()
        val_epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as val_progress_bar:
            for step, batch in val_progress_bar:
                images = batch["image"].to(self.device)

                with torch.no_grad():
                    with autocast(enabled=True):
                        noise = torch.randn(*z_shape).to(self.device)
                        timesteps = torch.randint(0, self.inferer.scheduler.num_train_timesteps, (images.shape[0],),
                                                  device=images.device).long()
                        noise_pred = self.inferer(inputs=images, autoencoder_model=self.vqvae, diffusion_model=self.network, noise=noise, timesteps=timesteps)
                        val_loss = F.mse_loss(noise_pred.float(), noise.float())

                val_epoch_loss += val_loss.item()
                val_progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

        if disable_prog_bar:
            end = time.time() - start
            print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Validation Loss: {val_epoch_loss / len(val_loader):.4f}")

        return val_epoch_loss / len(val_loader)

    def sample_image(self, z_shape, verbose=False):
        self.network.eval()
        self.vqvae.eval()
        image = torch.randn(1, *z_shape[1:]).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.config['n_infer_timesteps'])
        with torch.no_grad():
            with autocast(enabled=True):
                image = self.inferer.sample(input_noise=image, autoencoder_model=self.vqvae, diffusion_model=self.network, scheduler=self.scheduler,
                                            verbose=verbose)
        return image

    def save_plots(self, sampled_image, gif_output_path, epoch_loss_list=None, val_epoch_loss_list=None):
        if not os.path.exists(self.save_dict['plots']):
            os.makedirs(self.save_dict['plots'], exist_ok=True)

        num_slices = sampled_image.shape[2]
        gif_images = []

        for slice_idx in range(num_slices):
            plt.figure(figsize=(2, 2))
            slice_image = sampled_image.cpu()[0, 0, slice_idx, :, :]
            plt.imshow(slice_image, vmin=0, vmax=1, cmap="gray")
            plt.axis("off")

            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0)
            plt.close()

            buffer.seek(0)
            gif_image = Image.open(buffer).copy()
            gif_images.append(gif_image)
            buffer.close()

        create_gif_from_images(gif_images, gif_output_path)

        if epoch_loss_list is not None and val_epoch_loss_list is not None:
            save_main_losses_path = os.path.join(self.save_dict['plots'], "main_loss.png")
            save_main_losses(epoch_loss_list, val_epoch_loss_list, self.config['val_interval'], save_main_losses_path)

    def save_model(self, epoch, validation_loss, optimizer, scheduler=None):
        if not os.path.exists(self.save_dict['checkpoints']):
            os.makedirs(self.save_dict['checkpoints'], exist_ok=True)

        last_checkpoint_path = os.path.join(self.save_dict['checkpoints'], 'last_model.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

        best_checkpoint_path = os.path.join(self.save_dict['checkpoints'], 'best_model.pth')
        if os.path.isfile(best_checkpoint_path):
            best_checkpoint = torch.load(best_checkpoint_path)
            best_loss = best_checkpoint.get('validation_loss', float('inf'))
            if validation_loss < best_loss:
                torch.save(checkpoint, best_checkpoint_path)
        else:
            torch.save(checkpoint, best_checkpoint_path)

    def load_model(self, load_model_path, optimizer=None, lr_scheduler=None, for_training=False):
        print(f'Loading model from {load_model_path}...')
        checkpoint = torch.load(load_model_path)
        self.network.load_state_dict(checkpoint['network_state_dict'])

        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if lr_scheduler and 'scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if for_training:
            return checkpoint['epoch']

    def train(self, train_loader, val_loader, z_shape, optimizer, lr_scheduler=None):
        scaler = GradScaler()
        total_start = time.time()
        start_epoch = 0
        epoch_loss_list = []
        val_epoch_loss_list = []

        if self.config['load_model_path']:
            start_epoch, = self.load_model(self.config['load_model_path'], optimizer=optimizer, lr_scheduler=lr_scheduler,
                                           for_training=True)

        for epoch in range(start_epoch, self.config['n_epochs']):
            train_loss = self.train_one_epoch(epoch, train_loader, z_shape, optimizer, scaler)
            epoch_loss_list.append(train_loss)

            if epoch % self.config['val_interval'] == 0:
                val_loss = self.validate_epoch(val_loader, z_shape)
                val_epoch_loss_list.append(val_loss)
                if self.save_dict['plots']:
                    sample_verbose = not (self.config['output_mode'] == 'log' or not self.config['progress_bar'])
                    sampled_image = self.sample_image(z_shape, sample_verbose)
                    gif_output_path = os.path.join(self.save_dict['plots'], f"epoch_{epoch}.gif")
                    self.save_plots(sampled_image, gif_output_path, epoch_loss_list, val_epoch_loss_list)
                if self.save_dict['checkpoints']:
                    self.save_model(epoch, val_loss, optimizer, lr_scheduler)

            if lr_scheduler:
                lr_scheduler.step()
                print(f"Adjusting learning rate to {lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Training completed in {total_time:.2f} seconds.")


def main():
    args_mode = "train_ldm"
    args = parse_arguments(description="Train a Latent Diffusion Model", args_mode=args_mode)
    config = load_config(args.config)
    config = update_config_with_args(config, args, args_mode)
    config = validate_and_cast_config(config, args_mode)
    mode = "Training"
    network = "ldm"
    save_dict, save_path = create_save_path_dict(config)
    print_configuration(config, save_path, mode, model=network)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    set_determinism(42)

    train_loader, val_loader = get_data_loaders(config)

    scheduler = DDPMScheduler(num_train_timesteps=config['n_train_timesteps'], schedule=config['time_scheduler'],
                              beta_start=0.0015, beta_end=0.0195)
    # load vq-vae
    print(f"Loading VQ-VAE checkpoint from {config['load_vqvae_path']}...")
    checkpoint = torch.load(config['load_vqvae_path'])
    vqvae = VQVAE(**config['vqvae_params']).to(device)
    vqvae.load_state_dict(checkpoint['network_state_dict'])
    vqvae.eval()

    check_batch = next(iter(train_loader))['image']
    with torch.no_grad():
        with autocast(enabled=True):
            z = vqvae.encode_stage_2_inputs(check_batch.to(device))

    # for batch in train_loader:
    #     images = batch['image']
    #     with torch.no_grad():
    #         with autocast(enabled=True):
    #             z = vqvae.encode_stage_2_inputs(images.to(device))
    #             print(torch.min(z), torch.max(z))

    z_shape = tuple(z.shape)
    print(f"Latent shape: {z_shape}")
    print(f"Scaling factor set to {1 / torch.std(z)}")
    scale_factor = 1 / torch.std(z)

    input_shape = [(1, *z_shape[1:]), (1,)]
    network = DiffusionModelUNet(**config['model_params'])
    network.to(device)
    summary(network, input_shape, batch_dim=None, depth=3)

    inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    # inferer = LatentDiffusionInferer(scheduler)

    optimizer = torch.optim.Adam(params=network.parameters(), lr=config['learning_rate'])

    if config["lr_scheduler"]:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["lr_scheduler"])  # Get the class dynamically
        lr_scheduler = scheduler_class(optimizer, **config["lr_scheduler_params"])
    else:
        lr_scheduler = None

    model = LDM(config=config, network=network, vqvae=vqvae, inferer=inferer, scheduler=scheduler, device=device, save_dict=save_dict)

    print(f"\nStarting training ldm model...")
    model.train(train_loader=train_loader, val_loader=val_loader, z_shape=z_shape, optimizer=optimizer, lr_scheduler=lr_scheduler)