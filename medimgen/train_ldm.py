import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import os
import sys
import time
import glob
import traceback
import pickle
import argparse
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
# from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from generative.networks.nets import VQVAE

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import load_config
from medimgen.autoencoderkl_with_strides import AutoencoderKL
from medimgen.diffusion_model_unet_with_strides import DiffusionModelUNet
from medimgen.utils import create_gif_from_images, save_all_losses, create_2d_image_plot


class LDM:
    def __init__(self, config, latent_space_type='vae'):
        self.config = config
        self.latent_space_type = latent_space_type

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        print(f"Loading autoencoder checkpoint from {self.config['load_autoencoder_path']}...")
        if latent_space_type == 'vq':
            self.autoencoder = VQVAE(**self.config['vqvae_params']).to(self.device)
            checkpoint = torch.load(self.config['load_autoencoder_path'])
            self.autoencoder.load_state_dict(checkpoint['network_state_dict'])
            self.autoencoder.eval()
        elif self.latent_space_type == 'vae':
            self.autoencoder = AutoencoderKL(**self.config['vae_params']).to(self.device)
            checkpoint = torch.load(self.config['load_autoencoder_path'])
            self.autoencoder.load_state_dict(checkpoint['network_state_dict'])
            self.autoencoder.eval()
        else:
            raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")

        if latent_space_type == 'vq':
            self.codebook_min, self.codebook_max = self.get_codebook_min_max()

        self.ddpm = DiffusionModelUNet(**config['ddpm_params']).to(self.device)

        # https://towardsdatascience.com/generating-medical-images-with-monai-e03310aa35e6
        self.scheduler = DDPMScheduler(**self.config['time_scheduler_params'])

        if self.config['load_model_path']:
            # update loss_dict from previous training, as we are continuing training
            loss_pickle_path = os.path.join("/".join(self.config['load_model_path'].split('/')[:-2]), 'loss_dict.pkl')
            if os.path.exists(loss_pickle_path):
                with open(loss_pickle_path, 'rb') as file:
                    self.loss_dict = pickle.load(file)
        else:
            self.loss_dict = {'rec_loss': [], 'val_rec_loss': []}

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

    def get_inferer_and_latent_shape(self, train_loader):
        check_batch = next(iter(train_loader))['image']

        if self.latent_space_type == 'vq':
            inferer = DiffusionInferer(self.scheduler)
            with torch.no_grad():
                with autocast(enabled=True):
                    z = self.autoencoder.encode(check_batch.to(self.device))
        elif self.latent_space_type == 'vae':
            with torch.no_grad():
                with autocast(enabled=True):
                    z = self.autoencoder.encode_stage_2_inputs(check_batch.to(self.device))
            print(f"Scaling factor set to {1 / torch.std(z)}")
            scale_factor = 1 / torch.std(z)
            inferer = LatentDiffusionInferer(self.scheduler, scale_factor=scale_factor)
        else:
            raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")

        z_shape = tuple(z.shape)
        print(f"Latent shape: {z_shape}")
        return inferer, z_shape

    def get_optimizer_and_lr_schedule(self):
        # optimizer = torch.optim.Adam(params=self.ddpm.parameters(), lr=self.config['ddpm_learning_rate'])
        optimizer = torch.optim.SGD(self.ddpm.parameters(), self.config['ddpm_learning_rate'],
                                      weight_decay=self.config['weight_decay'], momentum=0.99, nesterov=True)
        if self.config["lr_scheduler"]:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])  # Get the class dynamically
            lr_scheduler = scheduler_class(optimizer, **self.config["lr_scheduler_params"])
        else:
            lr_scheduler = None

        return optimizer, lr_scheduler

    def train_one_epoch(self, epoch, train_loader, optimizer, scaler, inferer):
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
                    if self.latent_space_type == 'vq':
                        with torch.no_grad():
                            latents = self.autoencoder.encode(images)
                            latents_scaled = self.codebook_min_max_normalize(latents)

                    elif self.latent_space_type == 'vae':
                        with torch.no_grad():
                            latents = self.autoencoder.encode_stage_2_inputs(images)
                            latents_scaled = latents * inferer.scale_factor

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

        self.loss_dict['rec_loss'].append(epoch_loss / len(train_loader))

    def validate_epoch(self, val_loader, inferer):
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
                        if self.latent_space_type == 'vq':
                            with torch.no_grad():
                                latents = self.autoencoder.encode(images)
                                latents_scaled = self.codebook_min_max_normalize(latents)

                        elif self.latent_space_type == 'vae':
                            with torch.no_grad():
                                latents = self.autoencoder.encode_stage_2_inputs(images)
                                latents_scaled = latents * inferer.scale_factor

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

        self.loss_dict['val_rec_loss'].append(val_epoch_loss / len(val_loader))

    def sample_image(self, z_shape, inferer, verbose=False, seed=None):
        self.ddpm.eval()
        self.autoencoder.eval()
        if seed:
            # set seed for reproducible sampling
            torch.manual_seed(seed)
        input_noise = torch.randn(1, *z_shape[1:]).to(self.device)
        self.scheduler.set_timesteps(num_inference_steps=self.config['time_scheduler_params']['num_train_timesteps'])
        with torch.no_grad():
            with autocast(enabled=True):
                if self.latent_space_type == 'vq':
                    generated_latents = inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                                            scheduler=self.scheduler, verbose=verbose)
                    unscaled_latents = self.codebook_min_max_renormalize(generated_latents)
                    quantized_latents, _ = self.autoencoder.quantize(unscaled_latents)
                    image = self.autoencoder.decode(quantized_latents)
                elif self.latent_space_type == 'vae':
                    image = inferer.sample(input_noise=input_noise, diffusion_model=self.ddpm,
                                           autoencoder_model=self.autoencoder, scheduler=self.scheduler,
                                           verbose=verbose)
                    # image = self.autoencoder.decode_stage_2_outputs(generated_latents)
        return image

    def save_plots(self, sampled_image, plot_name):
        save_path = os.path.join(self.config['results_path'], 'plots')
        os.makedirs(save_path, exist_ok=True)

        is_3d = len(sampled_image.shape) == 5

        if is_3d:
            plot_save_path = os.path.join(save_path, f'{plot_name}.gif')
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

            create_gif_from_images(gif_images, plot_save_path)
        else:
            slice_image = sampled_image.cpu()[0, 0, :, :]
            plot_save_path = os.path.join(save_path, f'{plot_name}.png')
            create_2d_image_plot(slice_image, save_path=plot_save_path)

    def save_model(self, epoch, validation_loss, optimizer, scheduler=None):
        save_path = os.path.join(self.config['results_path'], 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'network_state_dict': self.ddpm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_loss': validation_loss
        }

        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        torch.save(checkpoint, last_checkpoint_path)

        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
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

    def train_main(self, train_loader, val_loader):
        scaler = GradScaler()
        total_start = time.time()
        start_epoch = 0
        sample_seed = 42
        plot_save_path = os.path.join(self.config['results_path'], 'plots')

        inferer, z_shape = self.get_inferer_and_latent_shape(train_loader)

        img_shape = self.config['transformations']['patch_size']
        ae_input_shape = (self.config['batch_size'], self.autoencoder.encoder.in_channels, *img_shape)
        ddpm_input_shape = [(self.config['batch_size'], *z_shape[1:]), (self.config['batch_size'],)]

        optimizer, lr_scheduler = self.get_optimizer_and_lr_schedule()

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer, lr_scheduler=lr_scheduler,
                                          for_training=True)

        print(f"\nStarting training ldm model...")
        summary(self.autoencoder, ae_input_shape, batch_dim=None, depth=3)
        summary(self.ddpm, ddpm_input_shape, batch_dim=None, depth=3)

        for epoch in range(start_epoch, self.config['n_epochs']):
            self.train_one_epoch(epoch, train_loader, optimizer, scaler, inferer)
            self.validate_epoch(val_loader, inferer)
            save_all_losses(self.loss_dict, plot_save_path)
            self.save_model(epoch, self.loss_dict['val_rec_loss'][-1], optimizer, lr_scheduler)

            loss_pickle_path = os.path.join("/".join(plot_save_path.split('/')[:-1]), 'loss_dict.pkl')
            with open(loss_pickle_path, 'wb') as file:
                pickle.dump(self.loss_dict, file)

            if epoch % self.config['val_plot_interval'] == 0:
                sample_verbose = not (self.config['output_mode'] == 'log' or not self.config['progress_bar'])
                sampled_image = self.sample_image(z_shape, inferer, sample_verbose, seed=sample_seed)
                self.save_plots(sampled_image, plot_name=f"epoch_{epoch}.gif")

            if lr_scheduler:
                lr_scheduler.step()
                print(f"Adjusting learning rate to {lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Training completed in {total_time:.2f} seconds.")

    def train(self, train_loader, val_loader):
        # unpack .npz files to .npy
        train_loader.dataset.unpack_dataset()

        try:
            self.train_main(train_loader=train_loader, val_loader=val_loader)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user (KeyboardInterrupt).")
        except Exception as e:
            traceback.print_exc()
        finally:
            # Clean up no matter what
            print("\nCleaning up dataset...")
            not_clean = True
            while not_clean:
                try:
                    train_loader.dataset.pack_dataset()
                    not_clean = False
                except BaseException:
                    continue


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an Autoencoder Model to reconstruct images.")
    parser.add_argument("dataset_id", type=str, help="Dataset ID")
    parser.add_argument("splitting", choices=["train-val-test", "5-fold"],
                        help="Choose either 'train-val-test' for a standard split or '5-fold' for cross-validation.")
    parser.add_argument("model_type", choices=["2d", "3d"],
                        help="Specify the model type: '2d' or '3d'.")
    parser.add_argument("-f", "--fold", type=int, choices=[0, 1, 2, 3, 4, 5], required=False, default=None,
                        help="Specify the fold index (0-5) when using 5-fold cross-validation.")
    parser.add_argument("-l", "--latent_space_type", type=str, default="vae", choices=["vae", "vq"],
                        help="Type of latent space to use: 'vae' or 'vq'. Default is 'vae'.")
    parser.add_argument("-p", "--progress_bar", action="store_true", help="Enable progress bar (default: False)")
    parser.add_argument("-c", "--continue_training", action="store_true",
                        help="Continue training from the last checkpoint (default: False)")
    args = parser.parse_args()

    # Ensure --fold is provided only when --splitting is "5-fold"
    if args.splitting == "5-fold" and args.fold is None:
        parser.error("--fold is required when --splitting is set to '5-fold'")

    # Ensure --fold is None when --splitting is "train-val-test"
    if args.splitting == "train-val-test" and args.fold is not None:
        parser.error("--fold should not be provided when --splitting is set to 'train-val-test'")

    return args


def get_config_for_current_task(dataset_id, model_type, progress_bar, continue_training, initial_config=None):
    preprocessed_dataset_path = glob.glob(os.getenv('nnUNet_preprocessed') + f'/Dataset{dataset_id}*/')[0]
    if not initial_config:
        config_path = os.path.join(preprocessed_dataset_path, 'medimgen_config.yaml')
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            raise FileNotFoundError(
                f"There is no medimgen configuration file for Dataset {dataset_id}. First run: medimgen_plan")
    else:
        config = initial_config
    config = config['2D'] if model_type == '2d' else config['3D']
    config['progress_bar'] = progress_bar
    config['output_mode'] = 'verbose'
    dataset_folder_name = preprocessed_dataset_path.split('/')[-2]

    main_results_path = os.path.join(os.getenv('medimgen_results'), dataset_folder_name, model_type)
    trained_ae_path = os.path.join(main_results_path, 'autoencoder', 'checkpoints', 'last_model.pth')
    if not os.path.isfile(trained_ae_path):
        raise FileNotFoundError(f"No pretrained autoencoder found. You should first train an autoencoder in order to "
                                f"train a latent diffusion model")
    config['load_autoencoder_path'] = trained_ae_path

    results_path = os.path.join(main_results_path, 'ldm')
    if os.path.exists(results_path) and not continue_training:
        raise FileExistsError(f"Results path {results_path} already exists.")
    config['results_path'] = results_path
    last_model_path = os.path.join(results_path, 'checkpoints', 'last_model.pth')
    config['load_model_path'] = last_model_path if continue_training else None
    return config


def main():

    args = parse_arguments()
    dataset_id = args.dataset_id
    splitting = args.splitting
    model_type = args.model_type
    fold = args.fold
    latent_space_type = args.latent_space_type
    progress_bar = args.progress_bar
    continue_training = args.continue_training

    config = get_config_for_current_task(dataset_id, model_type, progress_bar, continue_training)

    # TODO: need to add these to config
    # config['batch_size'] = 4
    # config['grad_accumulate_step'] = 1
    # config['ddpm_learning_rate'] = 1e-2
    # config['grad_clip_max_norm'] = 1

    train_loader, val_loader = get_data_loaders(config, dataset_id, splitting, model_type, fold)

    model = LDM(config=config, latent_space_type=latent_space_type)
    model.train(train_loader=train_loader, val_loader=val_loader)