"""
VascuMap3D - 3D Vascular Network Segmentation and Analysis

A comprehensive pipeline for segmenting and analyzing 3D vascular networks
from microscopy images.

Pipeline Steps:
    1. Image Translation: Brightfield to fluorescence using Pix2Pix 3D GAN
    2. Segmentation: 2D U-Net with 2.5D inference (axial, coronal, sagittal)
    3. Graph Analysis: Skeletonization and morphometric analysis

Modules:
    - models: Neural network architectures (Pix2Pix, U-Net)
    - pipeline: Inference and analysis pipelines

Example:
    >>> from vascumap.pipeline import run
    >>> run(lif_dir_path="/path/to/data")
"""

__version__ = "1.0.0"
__author__ = "VascuMap3D Team"

from . import models
from . import pipeline

__all__ = ['models', 'pipeline']

