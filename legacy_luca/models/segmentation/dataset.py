import collections
from pathlib import Path
from typing import List

import numpy as np
import tifffile as tif
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from .transforms import compose, hard_transforms, post_transforms, scale, contrast, new_axis

class SegmentationDataset(Dataset):
    """
    Dataset for segmentation tasks.

    Args:
        images (List[Path]): List of paths to image files.
        masks (List[Path], optional): List of paths to mask files. Defaults to None.
        transforms (callable, optional): Transformations to apply. Defaults to None.
    """
    def __init__(self, images: List[Path], masks: List[Path] = None,
                 transforms=None) -> None:
        self.images = images
        self.masks = masks
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        image_path = self.images[idx]
        image = tif.imread(image_path)
        
        image = contrast(image, .1, 99.9)
        image = scale(image)
        image = new_axis(image)

        result = {"image": image}

        if self.masks is not None:
            mask = tif.imread(self.masks[idx]).astype(int)
            result["mask"] = mask

        if self.transforms is not None:
            result = self.transforms(**result)

        result["filename"] = image_path.name

        return result

def train_datasets(images_path: Path, masks_path: Path,
                    train_transforms_fn, valid_transforms_fn,
                    seed, test_size, format='tif') -> tuple:
    """
    Create training and validation datasets from directories.

    Args:
        images_path (Path): Path to images directory.
        masks_path (Path): Path to masks directory.
        train_transforms_fn (callable): Transforms for training set.
        valid_transforms_fn (callable): Transforms for validation set.
        seed (int): Random seed for split.
        test_size (float): Fraction of data to use for validation.
        format (str, optional): File extension. Defaults to 'tif'.

    Returns:
        tuple: (train_dataset, valid_dataset)
    """

    all_images = sorted(images_path.glob(f"*.{format}"))
    all_masks = sorted(masks_path.glob(f"*.{format}")) if masks_path else None

    # Split the image and mask dataset into training and validation sets
    train_images, valid_images = train_test_split(all_images, test_size=test_size, random_state=seed)
    train_masks, valid_masks = train_test_split(all_masks, test_size=test_size, random_state=seed)
    
    return (SegmentationDataset(train_images, train_masks, train_transforms_fn),
            SegmentationDataset(valid_images, valid_masks, valid_transforms_fn))


def get_loaders(train_dataset: Dataset, valid_dataset: Dataset,
                batch_size: int, num_workers: int) -> dict:
    """
    Create data loaders for training and validation.

    Args:
        train_dataset (Dataset): Training dataset.
        valid_dataset (Dataset): Validation dataset.
        batch_size (int): Batch size.
        num_workers (int): Number of workers.

    Returns:
        dict: Dictionary containing 'train' and 'valid' loaders.
    """

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              drop_last=True,
                              pin_memory=True)

    valid_loader = DataLoader(valid_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              drop_last=False,
                              pin_memory=True)

    loaders = collections.OrderedDict()
    loaders["train"] = train_loader
    loaders["valid"] = valid_loader

    return loaders


def load_data(images_path_str: str, masks_path_str: str=None,
              batch_size=16, seed=0, test_size=0.1, format='tif') -> dict:
    """
    High-level function to load data and create loaders.

    Args:
        images_path_str (str): Path to images directory.
        masks_path_str (str, optional): Path to masks directory. Defaults to None.
        batch_size (int, optional): Batch size. Defaults to 16.
        seed (int, optional): Random seed. Defaults to 0.
        test_size (float, optional): Validation split fraction. Defaults to 0.1.
        format (str, optional): File format. Defaults to 'tif'.

    Returns:
        dict or DataLoader: If masks provided, returns dict of loaders. 
            Otherwise returns inference loader.
    """

    images_path = Path(images_path_str)

    train_transforms = compose([
          hard_transforms(), 
          post_transforms()
          ])

    valid_transforms = compose([
          post_transforms()
          ])

    if masks_path_str:
        
        masks_path=Path(masks_path_str)
        
        # Create datasets with transformations applied.
        train_dataset, valid_dataset = train_datasets(
            images_path,
            masks_path,
            train_transforms,
            valid_transforms, 
            seed, 
            test_size,
            format
            )
               
        # Get data loaders for training and validation subsets.
        return get_loaders(train_dataset,valid_dataset,batch_size,num_workers=0)
         
    else:

        inference_dataset=SegmentationDataset(
            list(images_path.glob(f'*{format}')),transforms=valid_transforms)
            
        inference_loader=DataLoader(inference_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              drop_last=False,
                              pin_memory=True)
        
        return inference_loader