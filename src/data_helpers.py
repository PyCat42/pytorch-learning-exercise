import torch
from torch.utils.data import Dataset

import os
from pathlib import Path

from PIL import Image

from typing import Dict, List, Tuple

def dummy_data_lin_reg(weight, bias,
                       start, end, step):
    """
    Create simple linear regression dummy data
    :param weight:
    :param bias:
    :param start:
    :param end:
    :param step:
    :return: X, y
    """
    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    return X, y

def walk_through_dir(dir_path):
    """
    Walk through a directory and print all files and directories.
    :param dir_path:
    :return:
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in {dirpath}")

def find_classes(directory: Path) -> Tuple[List[str], Dict[str, int]]:
    """
    Get the class names by scanning data directory subdirectories.
    Helper function for creating ImageFolderCustom function.
    :param directory: data directory
    :return:
    """
    # Get the class names by scanning directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # Raise errors if classes couldn't be found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}... please check file structure.")

    # Create dict of indices and labels
    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}

    return classes, class_to_idx

class ImageFolderCustom(Dataset):
    """
    Dataloading class that has to be able to:
    1. load images from file
    2. get class names as list and as dict
    Requires standard data format.

    All subclasses of Dataset should overwrite:
    __getitem__(), supporting fetching a data sample for a given key
    Could also optionally overwrite:
    __len__(), which is expected to return the size of the dataset by DataLoader
    """
    def __init__(self, target_dir: Path, transform: None):
        self.paths = list(Path(target_dir).glob("*/*.jpg")) # get all of the image paths
        self.transform = transform # setup transform
        self.classes, self.class_to_idx = find_classes(target_dir) # for 2.

    def load_image(self, index: int) -> Image.Image:
        """
        Function for 1.
        :param index: index of image to load.
        :return:
        """
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        """
        Returns dataset size.
        :return:
        """
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Returns one sample of data and label, (X, y).
        :param index: index of image to load.
        :return:
        """
        img = self.load_image(index)
        class_name = self.paths[index].parent.name # data_folder/class_name/image.jpg
        class_idx = self.class_to_idx[class_name]

        # transform if necesarry
        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
