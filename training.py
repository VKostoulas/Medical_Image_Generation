import os
import time
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from monai.utils import set_determinism
from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

from data_processing import get_data_loaders


def train_model(model, config, save_dict):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    set_determinism(42)

    train_loader, val_loader = get_data_loaders(config)

    if model == 'ddpm':
        train_ddpm(config, train_loader, val_loader, device, save_dict)
    elif model == 'lddpm':
        train_lddpm(config)
    else:
        raise ValueError("model not implemented.")


def train_ddpm(config, train_loader, val_loader, device, save_dict):
    model = DiffusionModelUNet(**config['model_params'])
    model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=config['n_train_timesteps'], schedule=config['time_scheduler'],
                              beta_start=0.0005, beta_end=0.0195)
    inferer = DiffusionInferer(scheduler)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config['learning_rate'])

    epoch_loss_list = []
    val_epoch_loss_list = []

    disable_prog_bar = True if config['output_mode'] == 'log' else False
    scaler = GradScaler()
    total_start = time.time()
    for epoch in range(config['n_epochs']):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70, disable=disable_prog_bar)
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

        if epoch % config['val_interval'] == 0:
            model.eval()
            val_epoch_loss = 0
            for step, batch in enumerate(val_loader):
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
                progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            val_epoch_loss_list.append(val_epoch_loss / (step + 1))

            # Sampling image during training
            image = torch.randn((1, 1) + config['resized_size'])
            image = image.to(device)
            scheduler.set_timesteps(num_inference_steps=config['n_infer_timesteps'])
            with autocast(enabled=True):
                image = inferer.sample(input_noise=image, diffusion_model=model, scheduler=scheduler)

            if config['save_plots']:
                plt.figure(figsize=(2, 2))
                plt.imshow(image[0, 0, :, :, 15].cpu(), vmin=0, vmax=1, cmap="gray")
                plt.tight_layout()
                plt.axis("off")
                # Save the figure with the epoch as the filename
                save_file = os.path.join(save_dict['plots'], f"epoch_{epoch}.png")
                plt.savefig(save_file, dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()  # Close the figure to free memory

    total_time = time.time() - total_start
    print(f"train completed, total time: {total_time}.")


def train_lddpm(config):
    return


