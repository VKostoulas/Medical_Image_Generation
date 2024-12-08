import os
from PIL import Image


def create_gif_from_folder(folder_path, output_path, duration=100):
    """
    Creates a GIF from a folder of PNG images named slice_0, slice_1, ..., slice_n.

    Args:
        folder_path (str): The path to the folder containing PNG images.
        output_path (str): The path to save the output GIF.
        duration (int): The duration of each frame in milliseconds (default: 100ms).

    Returns:
        None
    """
    # Get a sorted list of PNG files in the folder
    image_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )

    # Create a list to hold the images
    images = []

    # Load images into the list
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        images.append(Image.open(image_path))

    if not images:
        raise ValueError("No images found in the specified folder.")

    # Save the images as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

    print(f"GIF created successfully at {output_path}")