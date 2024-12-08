import os
import glob
import nibabel as nib
import numpy as np
import scipy.ndimage

from skimage import exposure

from medimgen.configuration import parse_arguments, print_configuration


def calculate_median_spacing(image_paths):
    spacings = []
    for path in image_paths:
        img = nib.load(path)
        spacing = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))  # Voxel spacing from affine
        spacings.append(spacing)
    return np.median(spacings, axis=0)


def crop_image(input_path):
    print("    Cropping image...")
    img = nib.load(input_path)
    data = img.get_fdata()
    nonzero_mask = data != 0
    nonzero_coords = np.array(np.where(nonzero_mask))
    min_coords = nonzero_coords.min(axis=1)
    max_coords = nonzero_coords.max(axis=1)
    cropped_data = data[
        min_coords[0]:max_coords[0]+1,
        min_coords[1]:max_coords[1]+1,
        min_coords[2]:max_coords[2]+1
    ]
    print(f"        Original size: {data.shape} - Cropped size: {cropped_data.shape}")
    cropped_img = nib.Nifti1Image(cropped_data, img.affine, img.header)
    return cropped_img


def resample_image(image, target_spacing):
    data = image.get_fdata()
    original_spacing = np.sqrt(np.sum(image.affine[:3, :3] ** 2, axis=0))
    if tuple(original_spacing) != tuple(target_spacing):
        print("    Difference with target spacing. Resampling image...")
        print(f"        Original spacing: {original_spacing} - Final spacing: {target_spacing}")
        zoom_factors = original_spacing / target_spacing
        resampled_data = scipy.ndimage.zoom(data, zoom_factors, order=3)  # Trilinear interpolation
        new_affine = np.copy(image.affine)
        new_affine[:3, :3] = new_affine[:3, :3] / zoom_factors[:, np.newaxis]
        return nib.Nifti1Image(resampled_data, new_affine, image.header)
    else:
        return image


def adjust_contrast(image):
    data = image.get_fdata()
    adjusted_data = exposure.equalize_adapthist(data, clip_limit=0.03)  # CLAHE
    adjusted_data = adjusted_data * np.max(data)  # Scale back to original intensity range
    return nib.Nifti1Image(adjusted_data, image.affine, image.header)


def main():
    args = parse_arguments(description="Crop and resample dataset. Optionally, normalize color intensities.",
                           args_mode="preprocess_data")
    config = {"task": str(args.task), "data_path": str(os.getenv('DATAPATH')), "save_path": str(os.getenv('SAVEPATH')),
              "intensity": bool(args.intensity)}

    save_path = os.path.join(config["data_path"], config["task"] + "_preprocessed", "imagesTr")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        raise FileExistsError(f"Folder {save_path} already exists.")

    mode = "Preprocess Dataset"
    print_configuration(config, save_path, mode)

    data_path = os.path.join(config["data_path"], config["task"], "imagesTr")
    image_paths = glob.glob(data_path + "/*.nii.gz")

    # Step 1: Calculate median voxel spacing
    print("Calculating median voxel spacing of the whole dataset...")
    median_spacing = calculate_median_spacing(image_paths)
    print(f"    Median voxel spacing: {median_spacing}")

    image_shapes = []
    for input_path in image_paths:
        print(f"\nProcessing {input_path}...")
        output_path = os.path.join(save_path, os.path.basename(input_path))
        cropped_image = crop_image(input_path)
        final_image = resample_image(cropped_image, median_spacing)
        image_shapes.append(final_image.shape)
        if config["intensity"]:
            # Step 4: Adjust contrast
            # TODO: Check https://github.com/jcreinhold/intensity-normalization/blob/master/tutorials/5min_tutorial.rst
            #  section: Saving fit information for sample-based methods
            #  or this https://github.com/sergivalverde/MRI_intensity_normalization/blob/master/Intensity%20normalization%20test.ipynb
            #  implementation of https://onlinelibrary.wiley.com/doi/pdf/10.1002/%28SICI%291522-2594%28199912%2942%3A6%3C1072%3A%3AAID-MRM11%3E3.0.CO%3B2-M
            print("    Adjusting intensity...")
            # final_image = adjust_contrast(final_image)

        # Save processed image
        nib.save(final_image, output_path)
        print(f"    Saved processed image to {output_path}")

    median_shape = tuple(np.median(np.array(image_shapes), axis=0).astype(int))
    min_shape = tuple(np.min(np.array(image_shapes), axis=0))
    max_shape = tuple(np.max(np.array(image_shapes), axis=0))
    print(f"Median Shape: {median_shape}")
    print(f"Min Shape: {min_shape}")
    print(f"Max Shape: {max_shape}")
