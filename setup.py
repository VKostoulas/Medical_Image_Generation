from setuptools import setup, find_packages

# Dynamically read requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    # Metadata
    name="medimgen",  # Name for `pip install`
    version="1.0.0",    # Version number
    description="A tool to train image generation models.",
    author="Vangelis Kostoulas",
    author_email="vageliskos93@gmail.com",
    url="https://github.com/VKostoulas/Medical_Image_Generation",  # GitHub or documentation link

    # Package Information
    packages=find_packages(),  # Automatically find subpackages in the project
    python_requires=">=3.9",  # Minimum Python version supported

    # Dependencies
    install_requires=parse_requirements("requirements.txt"),

    # CLI Commands
    entry_points={
        "console_scripts": [
            "normalize_dataset=medimgen.equalize_and_normalize_dataset:main",
            "train_ddpm=medimgen.train_ddpm:main",
            "train_vqgan=medimgen.train_vqgan:main",
            "train_ldm=medimgen.train_ldm:main",
            "sample_ddpm=medimgen.sample_ddpm:main",
            "sample_ldm=medimgen.sample_ldm:main"
        ],
    },
)
