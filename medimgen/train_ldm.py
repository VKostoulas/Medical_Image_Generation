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
from generative.inferers import DiffusionInferer, LatentDiffusionInferer
from generative.networks.nets import DiffusionModelUNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import VQVAE

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, filter_config_by_mode,
                                    print_configuration, create_save_path_dict)
from medimgen.utils import create_gif_from_images, save_main_losses


class LDM:
    def __init__(self, config, ddpm, autoencoder, inferer, scheduler, device, save_dict):
        self.config = config
        self.ddpm = ddpm
        self.autoencoder = autoencoder
        self.inferer = inferer
        self.scheduler = scheduler
        self.device = device
        self.save_dict = save_dict

        if self.config['latent_space_type'] == 'vq':
            self.codebook_min, self.codebook_max = self.get_codebook_min_max()

    def get_codebook_min_max(self):
        codebook = self.autoencoder.quantizer.quantizer.embedding.weight.data  # [num_codes, embedding_dim]
        # Find min and max across all codebook vectors
        min_val = codebook.min().item()
        max_val = codebook.max().item()
        return min_val, max_val

    def codebook_min_max_normalize(self, tensor):
        return 2 * ((tensor - self.codebook_min) / (self.codebook_max - self.codebook_min)) - 1

    def codebook_min_max_renormalize(self, tensor):
        return ((tensor + 1) / 2) * (self.codebook_max - self.codebook_min) + self.codebook_min

    def train_one_epoch(self, epoch, train_loader, optimizer, scaler):
        self.ddpm.train()
        self.autoencoder.eval()
        epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")

            optimizer.zero_grad(set_to_none=True)
            for step, batch in progress_bar:
                images = batch["image"].to(self.device)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],),
                                          device=images.device).long()

                with autocast(enabled=True):
                    if self.config['latent_space_type'] == 'vq':
                        with torch.no_grad():
                            latents = self.autoencoder.encode(images)
                            latents_scaled = self.codebook_min_max_normalize(latents)

                    elif self.config['latent_space_type'] == 'vae':
                        with torch.no_grad():
                            latents = self.autoencoder.encode_stage_2_inputs(images)
                            latents_scaled = latents * self.inferer.scale_factor

                    noise = torch.randn_like(latents_scaled).to(self.device)
                    noisy_latents = self.scheduler.add_noise(original_samples=latents_scaled, noise=noise, timesteps=timesteps)
                    noise_pred = self.ddpm(x=noisy_latents, timesteps=timesteps)

                    if self.scheduler.prediction_type == "v_prediction":
                        # Use v-prediction parameterization
                        target = self.scheduler.get_velocity(latents_scaled, noise, timesteps)
                    elif self.scheduler.prediction_type == "epsilon":
                        target = noise

                    loss = F.mse_loss(noise_pred.float(), target.float())

                scaler.scale(loss).backward()

                if (step + 1) % self.config['grad_accumulate_step'] == 0 or (step +1) == len(train_loader):
                    # gradient clipping
                    if self.config['grad_clip_max_norm']:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(self.ddpm.parameters(), max_norm=self.config['grad_clip_max_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        # Log epoch loss
        if disable_prog_bar:
            end = time.time() - start
            print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Train Loss: {epoch_loss / len(train_loader):.4f}")

        return epoch_loss / len(train_loader)

    def validate_epoch(self, val_loader):
        self.ddpm.eval()
        self.autoencoder.eval()
        val_epoch_loss = 0
        disable_prog_bar = self.config['output_mode'] == 'log' or not self.config['progress_bar']
        start = time.time()

        with tqdm(enumerate(val_loader), total=len(val_loader), ncols=100, disable=disable_prog_bar, file=sys.stdout) as val_progress_bar:
            for step, batch in val_progress_bar:
                images = batch["image"].to(self.device)
                timesteps = torch.randint(0, self.scheduler.num_train_timesteps, (images.shape[0],),
                                          device=images.device).long()

                with torch.no_grad():
                    with autocast(enabled=True):
                        if self.config['latent_space_type'] == 'vq':
                            with torch.no_grad():
                                latents = self.autoencoder.encode(images)
                                latents_scaled = self.codebook_min_max_normalize(latents)

                        elif self.config['latent_space_type'] == 'vae':
                            with torch.no_grad():
                                latents = self.autoencoder.encode_stage_2_inputs(images)
                                latents_scaled = latents * self.inferer.scale_factor

                        noise = torch.randn_like(latents_scaled).to(self.device)
                        noisy_latents = self.scheduler.add_noise(original_samples=latents_scaled, noise=noise,
                                                                 timesteps=timesteps)
                        noise_pred = self.ddpm(x=noisy_latents, timesteps=timesteps)

                        if self.scheduler.prediction_type == "v_prediction":
                            # Use v-prediction parameterization
                            target = self.scheduler.get_velocity(latents_scaled, noise, timesteps)
                        elif self.scheduler.prediction_type == "epsilon":
                            target = noise

                        val_loss = F.mse_loss(noise_pred.float(), target.float())

                val_epoch_loss += val_loss.item()
                val_progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})

        if disable_prog_bar:
            end = time.time() - start
            print(f"Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                  f"Validation Loss: {val_epoch_loss / len(val_loader):.4f}")

        return val_epoch_loss / len(val_loader)

    def sample_image(self, z_shape, verbose=False):
        self.ddpm.eval()
        self.autoencoder.eval()
        input_noise = torch.randn(1, *z_shape[1:]).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.config['time_scheduler_params']['num_train_timesteps'])
        with torch.no_grad():
            with autocast(enabled=True):
                if self.config['latent_space_type'] == 'vq':
                    generated_latents = self.inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                                            scheduler=self.scheduler, verbose=verbose)
                    unscaled_latents = self.codebook_min_max_renormalize(generated_latents)
                    quantized_latents, _ = self.autoencoder.quantize(unscaled_latents)
                    image = self.autoencoder.decode(quantized_latents)
                elif self.config['latent_space_type'] == 'vae':
                    image = self.inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                                autoencoder_model=self.autoencoder, scheduler=self.scheduler,
                                                verbose=verbose)
                    # image = self.autoencoder.decode_stage_2_outputs(generated_latents)
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
            'network_state_dict': self.ddpm.state_dict(),
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
        self.ddpm.load_state_dict(checkpoint['network_state_dict'])

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
            train_loss = self.train_one_epoch(epoch, train_loader, optimizer, scaler)
            epoch_loss_list.append(train_loss)

            if epoch % self.config['val_interval'] == 0:
                val_loss = self.validate_epoch(val_loader)
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
    config = filter_config_by_mode(config, args_mode)
    save_dict, save_path = create_save_path_dict(config)
    print_configuration(config, save_path, "Training", model="ldm")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    set_determinism(42)

    train_loader, val_loader = get_data_loaders(config)

    check_batch = next(iter(train_loader))['image']

    img_shape = config['transformations']['resize_shape'] if config['transformations']['resize_shape'] \
        else config['transformations']['patch_size']

    # https://towardsdatascience.com/generating-medical-images-with-monai-e03310aa35e6
    scheduler = DDPMScheduler(**config['time_scheduler_params'])

    print(f"Loading autoencoder checkpoint from {config['load_autoencoder_path']}...")
    if config['latent_space_type'] == 'vq':
        autoencoder = VQVAE(**config['vqvae_params']).to(device)
        checkpoint = torch.load(config['load_autoencoder_path'])
        autoencoder.load_state_dict(checkpoint['network_state_dict'])
        autoencoder.eval()
        ae_input_shape = (1, config['vqvae_params']['in_channels'], *img_shape)
        inferer = DiffusionInferer(scheduler)
        with torch.no_grad():
            with autocast(enabled=True):
                z = autoencoder.encode(check_batch.to(device))
    elif config['latent_space_type'] == 'vae':
        autoencoder = AutoencoderKL(**config['vae_params']).to(device)
        checkpoint = torch.load(config['load_autoencoder_path'])
        autoencoder.load_state_dict(checkpoint['network_state_dict'])
        autoencoder.eval()
        ae_input_shape = (1, config['vae_params']['in_channels'], *img_shape)
        with torch.no_grad():
            with autocast(enabled=True):
                z = autoencoder.encode_stage_2_inputs(check_batch.to(device))
        print(f"Scaling factor set to {1 / torch.std(z)}")
        scale_factor = 1 / torch.std(z)
        inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)
    else:
        raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")

    summary(autoencoder, ae_input_shape, batch_dim=None, depth=3)

    z_shape = tuple(z.shape)
    print(f"Latent shape: {z_shape}")

    ddpm_input_shape = [(1, *z_shape[1:]), (1,)]
    ddpm = DiffusionModelUNet(**config['ddpm_params']).to(device)
    summary(ddpm, ddpm_input_shape, batch_dim=None, depth=3)

    optimizer = torch.optim.Adam(params=ddpm.parameters(), lr=config['ddpm_learning_rate'])

    if config["lr_scheduler"]:
        scheduler_class = getattr(torch.optim.lr_scheduler, config["lr_scheduler"])  # Get the class dynamically
        lr_scheduler = scheduler_class(optimizer, **config["lr_scheduler_params"])
    else:
        lr_scheduler = None

    model = LDM(config=config, ddpm=ddpm, autoencoder=autoencoder, inferer=inferer, scheduler=scheduler, device=device, save_dict=save_dict)

    print(f"\nStarting training ldm model...")
    model.train(train_loader=train_loader, val_loader=val_loader, z_shape=z_shape, optimizer=optimizer, lr_scheduler=lr_scheduler)