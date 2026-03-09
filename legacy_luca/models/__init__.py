"""
VascuMap3D - Deep Learning Models

This package contains the neural network architectures and training code
for the VascuMap3D pipeline.

Submodules:
    - image_translation: Pix2Pix 3D GAN for brightfield to fluorescence translation
    - segmentation: U-Net based 2D segmentation with 2.5D inference
"""

from . import image_translation
from . import segmentation

__all__ = ['image_translation', 'segmentation']

