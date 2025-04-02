import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('Agg')

import os
import sys
import glob
import torch
import time
import pickle
import traceback
import argparse
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
from medimgen.configuration import load_config
from medimgen.utils import create_2d_image_reconstruction_plot, create_gif_from_images, save_all_losses


class AutoEncoder:
    def __init__(self, config, latent_space_type='vae'):
        self.config = config

        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using device: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("Using device: CPU")

        if latent_space_type == 'vq':
            self.autoencoder = VQVAE(**config['vqvae_params']).to(self.device)
        elif latent_space_type == 'vae':
            self.autoencoder = AutoencoderKL(**config['vae_params']).to(self.device)
        else:
            raise ValueError("Invalid latent_space_type. Choose 'vq' or 'vae'.")

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
        kl_loss = 0.5 * (z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1)
        spatial_dims = list(range(1, len(z_mu.shape)))  # [1,2,3] for 2D, [1,2,3,4] for 3D
        kl_loss = torch.sum(kl_loss, dim=spatial_dims)
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
            if isinstance(self.autoencoder, VQVAE):
                reconstructions, quantization_loss = self.autoencoder(images)
                step_loss_dict['reg_loss'] = quantization_loss * self.config['q_weight']
            elif isinstance(self.autoencoder, AutoencoderKL):
                reconstructions, z_mu, z_sigma = self.autoencoder(images)
                step_loss_dict['reg_loss'] = self.get_kl_loss(z_mu, z_sigma) * self.config['kl_weight']

            step_loss_dict['rec_loss'] = self.l1_loss(reconstructions.float(), images.float())
            step_loss_dict['perc_loss'] = perceptual_loss(reconstructions.float(), images.float()) * self.config['perc_weight']
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

    def get_optimizers_and_lr_schedules(self, discriminator):
        # optimizer_g = torch.optim.Adam(params=self.autoencoder.parameters(), lr=self.config['ae_learning_rate'])
        # optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=self.config['d_learning_rate'])

        optimizer_g = torch.optim.SGD(self.autoencoder.parameters(), self.config['ae_learning_rate'],
                                      weight_decay=self.config['weight_decay'], momentum=0.99, nesterov=True)
        optimizer_d = torch.optim.SGD(discriminator.parameters(), self.config['d_learning_rate'],
                                      weight_decay=self.config['weight_decay'], momentum=0.99, nesterov=True)

        if self.config["lr_scheduler"]:
            scheduler_class = getattr(torch.optim.lr_scheduler, self.config["lr_scheduler"])  # Get the class dynamically
            g_lr_scheduler = scheduler_class(optimizer_g, **self.config["lr_scheduler_params"])
            d_lr_scheduler = scheduler_class(optimizer_d, **self.config["lr_scheduler_params"])
        else:
            g_lr_scheduler = None
            d_lr_scheduler = None

        return optimizer_g, optimizer_d, g_lr_scheduler, d_lr_scheduler

    def save_plots(self, image, reconstruction, plot_name):
        save_path = os.path.join(self.config['results_path'], 'plots')
        os.makedirs(save_path, exist_ok=True)

        is_3d = len(image.shape) == 5

        if is_3d:
            plot_save_path = os.path.join(save_path, f'{plot_name}.gif')
            num_slices = image.shape[2]  # Assuming the image is [batch, channel, x, y, z]
            gif_images = []

            for slice_idx in range(num_slices):
                plt.figure(figsize=(4, 2))  # Adjusting the figsize for side-by-side plots
                # Plot the original image slice
                slice_image = image.cpu()[0, 0, slice_idx, :, :]
                plt.subplot(1, 2, 1)
                plt.imshow(slice_image, cmap="gray")
                plt.title("Image")
                plt.axis("off")
                # Plot the reconstruction slice
                slice_reconstruction = reconstruction.cpu()[0, 0, slice_idx, :, :]
                plt.subplot(1, 2, 2)
                plt.imshow(slice_reconstruction, cmap="gray")
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
            create_gif_from_images(gif_images, plot_save_path)

        else:
            slice_image = image.cpu()[0, 0, :, :]
            slice_reconstruction = reconstruction.cpu()[0, 0, :, :]
            plot_save_path = os.path.join(save_path, f'{plot_name}.png')
            create_2d_image_reconstruction_plot(slice_image, slice_reconstruction, save_path=plot_save_path)

    def save_model(self, epoch, validation_loss, optimizer, discriminator, disc_optimizer, scheduler=None,
                   disc_scheduler=None):
        save_path = os.path.join(self.config['results_path'], 'checkpoints')
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        last_checkpoint_path = os.path.join(save_path, 'last_model.pth')
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

        best_checkpoint_path = os.path.join(save_path, 'best_model.pth')
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

    def train_main(self, train_loader, val_loader):
        scaler_g = GradScaler()
        scaler_d = GradScaler()
        total_start = time.time()
        start_epoch = 0
        plot_save_path = os.path.join(self.config['results_path'], 'plots')

        img_shape = self.config['ae_transformations']['patch_size']
        input_shape = (self.config['batch_size'], self.autoencoder.encoder.in_channels, *img_shape)

        discriminator = PatchDiscriminator(**self.config['discriminator_params']).to(self.device)
        perceptual_loss = PerceptualLoss(**self.config['perceptual_params']).to(self.device)

        optimizer_g, optimizer_d, g_lr_scheduler, d_lr_scheduler = self.get_optimizers_and_lr_schedules(discriminator)

        if self.config['load_model_path']:
            start_epoch = self.load_model(self.config['load_model_path'], optimizer=optimizer_g, scheduler=g_lr_scheduler,
                                          discriminator=discriminator, disc_optimizer=optimizer_d, disc_scheduler=d_lr_scheduler,
                                          for_training=True)

        print("\nStarting training autoencoder model...")
        summary(self.autoencoder, input_shape, batch_dim=None, depth=3)
        summary(discriminator, input_shape, batch_dim=None, depth=3)
        summary(perceptual_loss, [input_shape, input_shape], batch_dim=None, depth=3)

        for epoch in range(start_epoch, self.config['n_epochs']):
            self.train_one_epoch(epoch, train_loader, discriminator, perceptual_loss, optimizer_g, optimizer_d, scaler_g, scaler_d)
            image, reconstruction = self.validate_one_epoch(val_loader, return_img_recon=True)
            save_all_losses(self.loss_dict, plot_save_path)
            self.save_model(epoch, self.loss_dict['val_rec_loss'][-1], optimizer_g, discriminator, optimizer_d,
                            scheduler=g_lr_scheduler, disc_scheduler=d_lr_scheduler)

            loss_pickle_path = os.path.join("/".join(plot_save_path.split('/')[:-1]), 'loss_dict.pkl')
            with open(loss_pickle_path, 'wb') as file:
                pickle.dump(self.loss_dict, file)

            if epoch % self.config['val_plot_interval'] == 0:
                self.save_plots(image, reconstruction, plot_name=f"epoch_{epoch}")

            if g_lr_scheduler:
                g_lr_scheduler.step()
                print(f"Adjusting learning rate of generator to {g_lr_scheduler.get_last_lr()[0]:.4e}.")

            if d_lr_scheduler:
                d_lr_scheduler.step()
                print(f"Adjusting learning rate of discriminator to {d_lr_scheduler.get_last_lr()[0]:.4e}.")

        total_time = time.time() - total_start
        print(f"Total training time: {time.strftime('%H:%M:%S', time.gmtime(total_time))}")

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


def infer_loss_weights_and_fit_gpu(dataset_id, model_type, initial_config):

    test_config = get_config_for_current_task(dataset_id=dataset_id, model_type=model_type, progress_bar=False,
                                              continue_training=False, initial_config=initial_config)

    train_loader, val_loader = get_data_loaders(test_config, dataset_id, splitting="train_val_test", model_type=model_type)

    # initialize config
    test_config['n_epochs'] = 1
    test_config['autoencoder_warm_up_epochs'] = 0
    test_config['grad_accumulate_step'] = 1
    test_config['kl_weight'] = 1e-8
    test_config['adv_weight'] = 1
    test_config['perc_weight'] = 1

    # unpack .npz files to .npy
    train_loader.dataset.unpack_dataset()

    not_done = True
    while not_done:
        try:
            model = AutoEncoder(config=test_config, latent_space_type='vae')
            model.train_main(train_loader=train_loader, val_loader=val_loader)

            loss_dict = model.loss_dict
            print(loss_dict)

            # modify perceptual loss weight
            perc_loss_weight_not_defined = True
            while perc_loss_weight_not_defined:
                rec_perc_difference = loss_dict['rec_loss'] / (loss_dict['perc_loss'] * test_config['perc_weight'])
                if rec_perc_difference >= 0.5:
                    test_config['perc_weight'] = test_config['perc_weight'] * 2
                elif rec_perc_difference <= 0.25:
                    test_config['perc_weight'] = test_config['perc_weight'] * 0.5
                else:
                    print(f"Perceptual loss weight set to: {test_config['perc_weight']}")
                    perc_loss_weight_not_defined = False

            # modify adversarial loss weight
            perc_loss_weight_not_defined = True
            while perc_loss_weight_not_defined:
                rec_perc_difference = loss_dict['rec_loss'] / (loss_dict['gen_loss'] * test_config['adv_weight'])
                if rec_perc_difference >= 0.5:
                    test_config['adv_weight'] = test_config['adv_weight'] * 2
                elif rec_perc_difference <= 0.25:
                    test_config['adv_weight'] = test_config['adv_weight'] * 0.5
                else:
                    print(f"Adversarial loss weight set to: {test_config['adv_weight']}")
                    perc_loss_weight_not_defined = False

            train_loader.dataset.pack_dataset()
            not_done = False
        except RuntimeError as e:
            if "out of memory" in str(e):
                test_config['batch_size'] = test_config['batch_size'] // 2
                print(f"CUDA out of memory error caught. Reducing batch_size to: {test_config['batch_size']}")
                # Optionally clear cache to free memory
                torch.cuda.empty_cache()
            else:
                # If it's some other error, re-raise it
                raise e


    # delete splits_file

    # delete results_folder

    # remove unnecessary key-values
    keys_to_remove = ['progress_bar', 'output_mode', 'results_path', 'load_model_path']
    test_config = {key: value for key, value in test_config.items() if key not in keys_to_remove}

    return test_config


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
    results_path = os.path.join(os.getenv('medimgen_results'), dataset_folder_name, model_type, 'autoencoder')
    if os.path.exists(results_path) and not continue_training:
        raise FileExistsError(f"Results path {results_path} already exists.")
    config['results_path'] = results_path
    last_model_path = os.path.join(results_path, 'checkpoints', 'last_model.pth')
    config['load_model_path'] = last_model_path if continue_training else None
    return config


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
    # config['autoencoder_warm_up_epochs'] = 5
    # config['grad_accumulate_step'] = 1
    config['kl_weight'] = 1e-8
    config['adv_weight'] = 0.25
    config['perc_weight'] = 1
    # config['ae_learning_rate'] = 1e-2
    # config['d_learning_rate'] = 1e-2
    # config['grad_clip_max_norm'] = 1

    # colon: adv_weight = 1, perc_weight = 4
    # brain:

    # config['transformations']['patch_size'] = (64, 64) # TODO: remove this
    transformations = config['ae_transformations']
    train_loader, val_loader = get_data_loaders(config, dataset_id, splitting, model_type, transformations, fold)

    model = AutoEncoder(config=config, latent_space_type=latent_space_type)
    model.train(train_loader=train_loader, val_loader=val_loader)

