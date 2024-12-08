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

from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from medimgen.data_processing import get_data_loaders
from medimgen.configuration import (load_config, parse_arguments, update_config_with_args, validate_and_cast_config,
                                    print_configuration, create_save_path_dict)
from medimgen.utils import create_gif_from_folder


def train_ddpm(config, train_loader, val_loader, device, save_dict):
    model = DiffusionModelUNet(**config['model_params'])
    model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=config['n_train_timesteps'], schedule=config['time_scheduler'],
                              beta_start=0.0005, beta_end=0.0195)
    inferer = DiffusionInferer(scheduler)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

    epoch_loss_list = []
    val_epoch_loss_list = []

    disable_prog_bar = config['output_mode'] == 'log' or not config['progress_bar']
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(config['n_epochs']):
        start = time.time()
        model.train()
        epoch_loss = 0
        with tqdm(enumerate(train_loader), total=len(train_loader), ncols=70, disable=disable_prog_bar, file=sys.stdout) as progress_bar:
            progress_bar.set_description(f"Epoch {epoch}")
            for step, batch in progress_bar:
                images = batch["image"].to(device)
                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    # Generate random noise
                    noise = torch.randn_like(images).to(device)
                    # Create timesteps
                    timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],),
                                              device=images.device).long()
                    # Get model prediction
                    noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                    loss = F.mse_loss(noise_pred.float(), noise.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
            epoch_loss_list.append(epoch_loss / (step + 1))

        # Log epoch loss
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
                    noise = torch.randn_like(images).to(device)
                    with torch.no_grad():
                        with autocast(enabled=True):
                            timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (images.shape[0],),
                                                      device=images.device).long()
                            # Get model prediction
                            noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                            val_loss = F.mse_loss(noise_pred.float(), noise.float())

                    val_epoch_loss += val_loss.item()
                    val_progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training
            img_shape = config['transformations']['resize_shape'] if config['transformations']['resize_shape'] else config['transformations']['patch_size']
            image = torch.randn((1, 1) + img_shape)
            image = image.to(device)
            scheduler.set_timesteps(num_inference_steps=config['n_infer_timesteps'])
            with autocast(enabled=True):
                image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler, verbose=not disable_prog_bar)

            # Log validation loss
            if disable_prog_bar:
                end = time.time() - start
                print(f"Epoch {epoch} - Time: {time.strftime('%H:%M:%S', time.gmtime(end))} - "
                      f"Validation Loss: {val_epoch_loss / len(val_loader):.4f}")

            if config['save_plots']:
                # Create a directory for the current epoch
                epoch_dir = os.path.join(save_dict['plots'], f"epoch_{epoch}")
                os.makedirs(epoch_dir, exist_ok=True)

                # Get the number of slices along the desired axis (e.g., the 4th dimension)
                num_slices = image.shape[2]  # Assuming the image is [batch, channel, x, y, z]
                # Normalize whole volume to 0-1
                image_min, image_max = image.cpu().min(), image.cpu().max()
                normalized_image = (image.cpu() - image_min) / (image_max - image_min)
                for slice_idx in range(num_slices):
                    plt.figure(figsize=(2, 2))
                    slice_image = normalized_image[0, 0, slice_idx, :, :]
                    plt.imshow(slice_image, vmin=0, vmax=1, cmap="gray")
                    plt.tight_layout()
                    plt.axis("off")
                    # Save the slice with its index in the epoch folder
                    slice_file = os.path.join(epoch_dir, f"slice_{slice_idx}.png")
                    plt.savefig(slice_file, dpi=300, bbox_inches='tight', pad_inches=0)
                    plt.close()  # Close the figure to free memory
                create_gif_from_folder(epoch_dir, os.path.join(epoch_dir, f"epoch_{epoch}.gif"), 80)
    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")



class ModelTrainer:
    def __init__(self, model, optimizer_dict, loss_dict, scheduler_dict, train_loader, evaluator, max_epochs=100,
                 use_mixed_precision=False, epoch_repeats=1, disable_bar=False, save_dict={}):
        self.model = model
        self.optimizer = choose_optimizer(optimizer_dict['optimizer'], optimizer_dict['params'], self.model)
        self.loss_function = choose_loss_function(loss_dict['loss'], loss_dict['params'])
        self.scheduler = choose_scheduler(scheduler_dict['scheduler'], scheduler_dict['params'], self.optimizer)
        self.train_loader = train_loader
        self.evaluator = evaluator
        self.max_epochs = max_epochs
        self.use_mixed_precision = use_mixed_precision
        self.epoch_repeats = epoch_repeats
        self.disable_bar = disable_bar
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dict = save_dict
        self.val_dice_scores = {}

        if save_dict['graph']:
            create_network_graph(trainloader=train_loader, network=self.model, save_path=save_dict['graph'],
                                 device=self.device)

    def train_one_epoch(self, current_epoch, grad_scaler):
        self.model.train()
        num_train_batches = len(self.train_loader) * self.epoch_repeats
        epoch_loss = 0
        start = time.time()
        prof = self.start_profiler(current_epoch)

        print(f'Epoch {current_epoch + 1}/{self.max_epochs}')
        with tqdm(total=num_train_batches, unit='batches', file=sys.stdout, disable=self.disable_bar, leave=False,
                  bar_format=' ', ascii='.>=') as pbar:
            pbar.bar_format = "{n_fmt}/{total_fmt}   [{bar:30}] - ETA: {remaining} {postfix}"

            for i in range(self.epoch_repeats):
                for batch_num, batch in enumerate(self.train_loader):
                    images = batch['image'].to(device=self.device, dtype=torch.float32)
                    targets = batch['mask'].to(device=self.device, dtype=torch.long)
                    loss_args = []
                    if 'weight_map' in batch.keys():
                        weight_maps = batch['weight_map']
                        weight_maps = weight_maps.to(device=self.device, dtype=torch.float32)
                        loss_args.append(weight_maps)
                    with torch.cuda.amp.autocast(enabled=self.use_mixed_precision):
                        masks_pred = self.model(images)
                        if hasattr(self.loss_function, 'blend_classes'):
                            loss_args.append(self.model.network_1_outputs)
                        loss = self.loss_function(masks_pred, targets, *loss_args)

                    self.optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                    epoch_loss += loss.item()
                    self.profiler_step(current_epoch, prof)
                    pbar.set_postfix_str(f'loss: {epoch_loss / (len(self.train_loader) * i + batch_num + 1):.4f}')
                    pbar.update()

        self.stop_profiler(current_epoch, prof)
        end = time.time() - start
        print(f"Epoch Time: {time.strftime('%H:%M:%S', time.gmtime(end))} , "
              f"loss: {epoch_loss / (len(self.train_loader) * i + batch_num + 1):.4f}")

        return epoch_loss / (len(self.train_loader) * i + batch_num + 1)

    def train(self, spacing, hd_percentile=95, disable_bar=False, slice_dimension=None, load_model_path=''):
        grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)
        current_epoch = self.load_model(load_model_path)

        for epoch in range(current_epoch, self.max_epochs):
            train_loss = self.train_one_epoch(epoch, grad_scaler)
            self.evaluator.compute_metrics(disable_bar, spacing, hd_percentile, slice_dimension)
            val_dice_score = self.evaluator.avg_metrics['Dice Score']['average']
            self.evaluator.reset_patient_data(reset_all=True)

            if 'metrics' in inspect.getfullargspec(self.scheduler.step)[0]:
                self.scheduler.step(train_loss)
            else:
                self.scheduler.step()

            self.save_plot(epoch, val_dice_score)
            self.save_model(epoch, train_loss, val_dice_score)

    def save_plot(self, epoch, val_dice_score):
        if self.save_dict['plots']:
            self.val_dice_scores[epoch] = val_dice_score
            plt.xlabel('Epoch')
            plt.ylabel('Validation Dice')
            plt.plot(list(self.val_dice_scores.keys()), list(self.val_dice_scores.values()))
            plt.savefig(self.save_dict['plots'] + 'progress.png')
            plt.close()

            save_pickle_path = self.save_dict['plots'] + 'progress_items.pickle'
            with open(save_pickle_path, 'wb') as pickle_file:
                pickle.dump(self.val_dice_scores, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_model(self, epoch, train_loss, val_dice_score):
        if self.save_dict['checkpoints']:
            Path(self.save_dict['checkpoints']).mkdir(parents=True, exist_ok=True)
            last_checkpoint_path = self.save_dict['checkpoints'] + 'last_model.pth'
            torch.save({'epoch': epoch + 1, 'network_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(), 'train_loss': train_loss,
                        'val_dice_score': val_dice_score}, last_checkpoint_path)
            best_checkpoint_path = self.save_dict['checkpoints'] + 'best_model.pth'
            if os.path.isfile(best_checkpoint_path):
                best_checkpoint = torch.load(best_checkpoint_path)
                dice_score = best_checkpoint['val_dice_score']
                if val_dice_score > dice_score:
                    torch.save({'epoch': epoch + 1, 'network_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'scheduler_state_dict': self.scheduler.state_dict(), 'train_loss': train_loss,
                                'val_dice_score': val_dice_score}, best_checkpoint_path)
            else:
                torch.save({'epoch': epoch + 1, 'network_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'scheduler_state_dict': self.scheduler.state_dict(), 'train_loss': train_loss,
                            'val_dice_score': val_dice_score}, best_checkpoint_path)

    def load_model(self, load_model_path):
        if load_model_path:
            print(f'Continue training: Loading model, optimizer and scheduler from {load_model_path}...')
            checkpoint = torch.load(load_model_path)
            self.model.load_state_dict(checkpoint['network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            current_epoch = checkpoint['epoch']
        else:
            current_epoch = 0
        return current_epoch

    def start_profiler(self, current_epoch):
        if self.save_dict['profile']:
            if current_epoch == 0:
                prof = torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=2, warmup=2, active=6, repeat=2, skip_first=50),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(self.save_dict['profile']),
                    record_shapes=True, profile_memory=True, with_stack=True)
                prof.start()
                return prof
            else:
                return None

    def profiler_step(self, current_epoch, prof):
        if self.save_dict['profile']:
            if current_epoch == 0:
                prof.step()

    def stop_profiler(self, current_epoch, prof):
        if self.save_dict['profile']:
            if current_epoch == 0:
                prof.stop()

def main():
    args = parse_arguments(description="Train a Denoising Diffusion Probabilistic Model", args_mode="train_ddpm")
    config = load_config(args.config)
    config = update_config_with_args(config, args)
    config = validate_and_cast_config(config)
    mode = "Training"
    model = "ddpm"
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
    train_ddpm(config, train_loader, val_loader, device, save_dict)