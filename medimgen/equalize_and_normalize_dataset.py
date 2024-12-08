from medimgen.configuration import parse_arguments


def main():
    args = parse_arguments(description="Resample images to the median voxel spacing of the dataset and perform "
                                       "histogram equalization",
                           args_mode="normalize_data")