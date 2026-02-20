"""
VascuMap3D - Image Translation Module

This module provides the Pix2Pix 3D GAN architecture for translating
brightfield Z-stacks to fluorescence-like volumes.

Classes:
    - ImageDataset3D: PyTorch Dataset for 3D NIfTI volumes
    - Generator: 3D U-Net generator
    - Discriminator: 3D PatchGAN discriminator  
    - Pix2Pix: PyTorch Lightning module for training
"""

from .utils import ImageDataset3D, Generator, Discriminator, Pix2Pix
from .train import train

__all__ = [
    'ImageDataset3D',
    'Generator', 
    'Discriminator',
    'Pix2Pix',
    'train',
]

