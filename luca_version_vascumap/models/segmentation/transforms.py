import numpy as np

import albumentations as albu
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensor
from skimage.filters import apply_hysteresis_threshold

def scale(arr):
    """
    Scales the input array to be in the range [0, 1].

    Args:
        arr (numpy.array): The input array to be scaled.

    Returns:
        numpy.array: The scaled array with values in the range [0, 1].
    """

    if np.mean(arr) == 0 or np.mean(arr) == 1:
        return arr

    else:
        return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def contrast(arr, low, top):
    """
    Enhances the contrast of an array by clipping its values based on given percentiles.

    Args:
        arr (numpy.ndarray): Input array to enhance the contrast.
        low (float): Lower percentile value for clipping.
        top (float): Upper percentile value for clipping.

    Returns:
        numpy.ndarray: Array with enhanced contrast.
    """

    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, top))

def new_axis(arr):
    """
    Adds a new axis to the end of an array (e.g., for channel dimension).

    Args:
        arr (numpy.array): The input array.

    Returns:
        numpy.array: The array with an added axis.
    """

    return arr[..., np.newaxis]

def hysteresis_thresholding(arr, low=0.15, high=0.5):
    """
    Apply hysteresis thresholding to an array.

    Args:
        arr (np.ndarray): Input probability map.
        low (float): Low threshold.
        high (float): High threshold.

    Returns:
        np.ndarray: Binary mask (int).
    """
    return apply_hysteresis_threshold(arr, low, high).astype(int)

def single_level_thresholding(arr, threshold=0.5):
    """
    Apply simple thresholding.

    Args:
        arr (np.ndarray): Input array.
        threshold (float): Threshold value.

    Returns:
        np.ndarray: Binary mask (int).
    """
    return np.array(arr > threshold).astype(int)

def hard_transforms():
    """
    Returns a list of 'hard' augmentations for training.
    
    Includes: Flip, ShiftScaleRotate.

    Returns:
        list: List of Albumentations transforms.
    """
    return [
        albu.Flip(),
        albu.ShiftScaleRotate(),
    ]

def post_transforms():
    """
    Returns a list of post-processing transforms.
    
    Includes: ToTensor (converts to PyTorch tensor).

    Returns:
        list: List of Albumentations transforms.
    """
    return [ToTensor()]

def compose(transforms_to_compose):
    """
    Composes multiple lists of transformations into a single pipeline.

    Args:
        transforms_to_compose (list): A list of lists containing transformations.

    Returns:
        albumentations.Compose: Composed transformation pipeline.
    """

    # Flatten the list of lists and create a composition
    result = albu.Compose([
    item for sublist in transforms_to_compose for item in sublist
    ])

    return result
