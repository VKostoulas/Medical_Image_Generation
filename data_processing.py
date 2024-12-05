import tempfile

from monai.apps import DecathlonDataset
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    CenterSpatialCropd,
    Compose,
    Lambdad,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)


def get_data_loaders(config):
    print(config['data_path'])
    print(tempfile.gettempdir())

    data_transform = Compose(
        [
            LoadImaged(keys=["image"]),
            Lambdad(keys="image", func=lambda x: x[:, :, :, 1]),
            EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
            ScaleIntensityd(keys=["image"]),
            CenterSpatialCropd(keys=["image"], roi_size=config['center_crop_size']),
            Resized(keys=["image"], spatial_size=config['resized_size']),
        ]
    )
    progress = False if (config['output_mode'] == 'log' or not config['progress_bar']) else True
    train_ds = DecathlonDataset(root_dir=config['data_path'], task="Task01_BrainTumour", transform=data_transform,
                                section="training", download=True, progress=progress)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=8, persistent_workers=True,
                              pin_memory=True)
    val_ds = DecathlonDataset(root_dir=config['data_path'], task="Task01_BrainTumour", transform=data_transform,
                              section="validation", download=True, progress=progress)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=8, persistent_workers=True,
                            pin_memory=True)
    return train_loader, val_loader