"""
VascuMap3D - 2D Segmentation Module

This module provides U-Net based segmentation models for vessel segmentation
using 2D slices with 2.5D inference capabilities.

Classes:
    - SegmentationDataset: PyTorch Dataset for 2D images
    - SegmentationModule: PyTorch Lightning module for training

Functions:
    - train: Train a segmentation model
    - build_model: Factory function for creating models
"""

from .model import SegmentationModule
from .dataset import SegmentationDataset, load_data
from .model_utils import build_model, adapt_input_model
from .training import train
from .transforms import compose, hard_transforms, post_transforms

__all__ = [
    'SegmentationModule',
    'SegmentationDataset',
    'load_data',
    'build_model',
    'adapt_input_model',
    'train',
    'compose',
    'hard_transforms',
    'post_transforms',
]

