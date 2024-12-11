import os
import torch
import glob
import random
import numpy as np
import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.color_transforms import GammaTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform, ResizeTransform
from nnunet.training.data_augmentation.custom_transforms import Convert3DTo2DTransform, Convert2DTo3DTransform, \
    ConvertSegmentationToRegionsTransform


def get_data_loaders(config):
    train_ds = MedicalDataset(root_dir=config['data_path'], task=config['task'], section="training",
                              split_ratios=config['splitting'], transformation_args=config['transformations'],
                              channel_ids=config['channels'])
    val_ds = MedicalDataset(root_dir=config['data_path'], task=config['task'], section="validation",
                            split_ratios=config['splitting'], transformation_args=config['transformations'],
                            channel_ids=config['channels'])

    loader_args = dict(batch_size=config['batch_size'], num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, prefetch_factor=config['batch_size'], **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, prefetch_factor=config['batch_size'], **loader_args)
    return train_loader, val_loader


class MedicalDataset(Dataset):
    def __init__(self, root_dir, task, section, split_ratios, transformation_args, channel_ids=None):
        self.root_dir = root_dir
        self.task = task
        self.section = section
        self.transformation_args = transformation_args
        self.channel_ids = channel_ids

        self.data_path = os.path.join(root_dir, task, 'imagesTr')
        self.ids = self.split_image_ids(split_ratios)

    def __len__(self):
        return len(self.ids)

    def split_image_ids(self, split_ratios):
        # Get all .nii.gz files in the directory
        all_files = [f for f in os.listdir(self.data_path) if not f.startswith('.') and f.endswith('.nii.gz')]

        # Extract unique image IDs (assuming IDs are the filenames without extensions)
        image_ids = [os.path.splitext(os.path.splitext(f)[0])[0] for f in all_files]
        unique_image_ids = sorted(set(image_ids))  # Ensure IDs are unique and sorted for reproducibility

        # Set random seed for reproducibility
        random.seed(42)

        # Shuffle the IDs for random splitting
        random.shuffle(unique_image_ids)

        # Calculate split sizes
        total_count = len(unique_image_ids)
        train_count = int(split_ratios[0] * total_count)
        val_count = int(split_ratios[1] * total_count)

        # Split the IDs
        train_ids = unique_image_ids[:train_count]
        val_ids = unique_image_ids[train_count:]

        if self.section == "training":
            print(f'{len(train_ids)} images for training')
            return train_ids
        elif self.section == "validation":
            print(f'{len(val_ids)} images for validation')
            return val_ids

    def transform(self, image):
        apply_transforms = True if self.section == "training" else False
        transformations = define_nnunet_transformations(self.transformation_args, apply_transforms)
        transformed = transformations(data=image)
        transformed_image = transformed["data"]
        return transformed_image

    def load_image(self, name):
        image_path = glob.glob(os.path.join(self.data_path, name) + '.*')
        if not image_path:
            raise FileNotFoundError(f"Image file for ID '{name}' not found in {self.data_path}")
        # Load the NIfTI file
        nifti_image = sitk.ReadImage(image_path[0])
        image = sitk.GetArrayFromImage(nifti_image)
        # Select specific channels if channel_ids is provided
        if self.channel_ids is not None:
            image = image[self.channel_ids, ...]
        if len(image.shape) < 4:
            image = np.expand_dims(image, axis=0)  # add channel dimension
        image = np.expand_dims(image, axis=0)  # add batch dimension
        return image

    def __getitem__(self, idx):
        name = self.ids[idx]
        image = self.load_image(name)
        # scale to 0-1 for augmentations
        min_val = np.min(image)
        max_val = np.max(image)
        image = (image - min_val) / (max_val - min_val)
        image = self.transform(image)
        # scale to -1 1
        # min_val = np.min(image)
        # max_val = np.max(image)
        # image = 2 * (image - min_val) / (max_val - min_val) - 1
        image = torch.as_tensor(image).float()
        image = torch.squeeze(image, dim=0)
        image = image.contiguous()
        return {'id': name, 'image': image}


def define_nnunet_transformations(params, validation=False, border_val_seg=-1, regions=None):
    if not validation:
        tr_transforms = []

        # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
        if params.get("dummy_2D"):
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = params.get("patch_size")[1:]
        else:
            patch_size_spatial = params.get("patch_size")

        elastic_deform_alpha = (0., 200.)
        elastic_deform_sigma = (9., 13.)
        p_eldef = 0.2
        scale_range = (0.85, 1.25)
        independent_scale_factor_for_each_axis = False
        p_scale = 0.2
        # rotation_x = (-0.349066, 0.349066)
        # rotation_y = (-0.349066, 0.349066)
        # rotation_z = (-0.349066, 0.349066)
        rotation_x = (-0.174533, 0.174533)
        rotation_y = (-0.174533, 0.174533)
        rotation_z = (-0.174533, 0.174533)
        p_rot = 0.2
        gamma_retain_stats = True
        gamma_range = (0.7, 1.5)
        p_gamma = 0.15
        mirror_axes = (0, 1)
        border_mode_data = "constant"

        tr_transforms.append(SpatialTransform(
            patch_size_spatial, do_elastic_deform=params.get("elastic"), alpha=elastic_deform_alpha, sigma=elastic_deform_sigma,
            do_rotation=params.get("rotation"), angle_x=rotation_x, angle_y=rotation_y, angle_z=rotation_z,
            do_scale=params.get("scaling"), scale=scale_range, border_mode_data=border_mode_data,
            border_cval_data=0, order_data=3, border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=1,
            random_crop=False, p_el_per_sample=p_eldef, p_scale_per_sample=p_scale,
            p_rot_per_sample=p_rot, independent_scale_for_each_axis=independent_scale_factor_for_each_axis
        ))
        if params.get("dummy_2D"):
            tr_transforms.append(Convert2DTo3DTransform())

        if params.get("resize_shape"):
            tr_transforms.append(ResizeTransform(params.get("resize_shape")))

        if params.get("gaussian_noise"):
            tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))

        if params.get("_gaussian_blur"):
            tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.1,
                                                       p_per_channel=0.5))
        if params.get("brightness"):
            tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))

        if params.get("contrast"):
            tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))

        if params.get("gamma"):
            tr_transforms.append(
                GammaTransform(gamma_range, False, True, retain_stats=gamma_retain_stats,
                               p_per_sample=p_gamma))

        if params.get("mirror"):
            tr_transforms.append(MirrorTransform(mirror_axes))

        # tr_transforms.append(RemoveLabelTransform(-1, 0))

        # tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

        tr_transforms = Compose(tr_transforms)

        return tr_transforms
    else:
        # val_transforms = [RemoveLabelTransform(-1, 0)]
        val_transforms = [SpatialTransform(
            params.get("patch_size"), do_elastic_deform=False, do_rotation=False, do_scale=False,
            border_cval_seg=border_val_seg, order_seg=1, random_crop=False
        )]

        if params.get("resize_shape") is not None:
            val_transforms.append(ResizeTransform(params.get("resize_shape")))

        # val_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))

        val_transforms = Compose(val_transforms)

        return val_transforms