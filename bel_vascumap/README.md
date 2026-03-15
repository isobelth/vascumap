# VascuMap3D

**3D Vascular Network Segmentation and Graph Analysis Pipeline**

VascuMap3D is a comprehensive Python pipeline for segmenting and analyzing 3D vascular networks from microscopy images. It combines deep learning-based image translation and segmentation with graph-based topological analysis.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch-Lightning-792ee5.svg)](https://lightning.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

The pipeline processes 3D microscopy data through four main stages:

1. **Image Translation**: Converts brightfield Z-stacks to fluorescence-like volumes using a Pix2Pix 3D GAN
2. **Segmentation**: Segments vessels using a 2D U-Net with 2.5D inference (multi-plane averaging)
3. **Skeletonization**: Extracts vessel centerlines via thinning or mesh contraction
4. **Graph Analysis**: Computes comprehensive morphometric and topological metrics

---

## Project Structure

```
VascuMap3D/
├── vascumap/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── image_translation/          # Pix2Pix 3D GAN
│   │   │   ├── __init__.py
│   │   │   ├── train.py                # Training script with CLI
│   │   │   └── utils.py                # Generator, Discriminator, Dataset
│   │   └── segmentation/               # 2D U-Net Segmentation
│   │       ├── __init__.py
│   │       ├── training.py             # Training script with CLI
│   │       ├── model.py                # SegmentationModule (Lightning)
│   │       ├── model_utils.py          # Model factory functions
│   │       ├── dataset.py              # Dataset and data loading
│   │       └── transforms.py           # Albumentations transforms
│   └── pipeline/
│       ├── __init__.py
│       ├── main.py                     # Main inference pipeline
│       ├── graph_main.py               # Graph analysis pipeline
│       ├── data.py                     # LIF file loading
│       ├── preprocessing.py            # Auto-focus, registration, cropping
│       ├── prediction.py               # Model inference functions
│       ├── skeletonization.py          # Thinning-based skeletonization
│       ├── mesh_contraction.py         # Mesh contraction skeletonization
│       ├── metrics.py                  # Graph metric calculations
│       ├── visualization.py            # Plotting functions
│       └── utils.py                    # GPU utilities, image processing
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended)
- CUDA Toolkit 11.x or 12.x

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/VascuMap3D.git
cd VascuMap3D

# Create conda environment
conda create -n vascumap python=3.9
conda activate vascumap

# Install dependencies
pip install -r requirements.txt
```

> **Note**: Adjust `cupy-cuda11x` in `requirements.txt` to match your CUDA version (e.g., `cupy-cuda12x` for CUDA 12).

---

## Usage

### 1. Training Models

#### Image Translation (Pix2Pix 3D)

Train the GAN to translate brightfield to fluorescence:

```bash
python -m vascumap.models.image_translation.train \
    --input_path /path/to/brightfield/volumes \
    --target_path /path/to/fluorescence/volumes \
    --model_path /path/to/save/checkpoints \
    --batch_size 4 \
    --epochs 200 \
    --generator_lr 1e-3 \
    --discriminator_lr 1e-5
```

#### Segmentation (2D U-Net)

Train the segmentation model:

```bash
python -m vascumap.models.segmentation.training \
    --images_dir_path /path/to/training/images \
    --masks_dir_path /path/to/training/masks \
    --model_dir_path /path/to/save/checkpoints \
    --model_architecture Unet \
    --encoder_architecture mit_b5 \
    --batch_size 16 \
    --epochs 200 \
    --learning_rate 1e-3 \
    --fp16
```

### 2. Inference Pipeline

Process LIF files to generate vessel masks:

```bash
python -m vascumap.pipeline.main \
    --lif_dir_path /path/to/lif/files \
    --model_p2p_path /path/to/translation.ckpt \
    --model_smp_path /path/to/segmentation.ckpt
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--lif_dir_path` | Directory containing `.lif` microscopy files |
| `--model_p2p_path` | Path to trained Pix2Pix checkpoint |
| `--model_smp_path` | Path to trained segmentation checkpoint |

### 3. Graph Analysis

Extract and analyze vascular network topology:

```bash
python -m vascumap.pipeline.graph_main \
    --masks_dir_path /path/to/masks \
    --file_suffix _vessel_mask \
    --method thinning \
    --save_local_metrics \
    --graph_visualization
```

**Arguments:**
| Argument | Description |
|----------|-------------|
| `--masks_dir_path` | Directory containing mask files |
| `--file_suffix` | Mask file suffix (default: `_mask`) |
| `--method` | Skeletonization: `thinning` (fast) or `contraction` (robust) |
| `--save_local_metrics` | Save per-vessel and per-junction metrics |
| `--graph_visualization` | Generate network visualization images |

---

## Output Metrics

### Global Metrics (per sample)
- **Volume**: Total vessel volume, explant volume, coverage ratio
- **Network**: Total length, number of vessels, sprouts, junctions
- **Complexity**: Fractal dimension, lacunarity

### Vessel Metrics (per segment)
- Length, volume, tortuosity
- Cross-sectional area (mean, median, std)
- Orientation vector
- Connectivity (node degrees)

### Junction Metrics (per node)
- Position (z, y, x)
- Degree (number of connected vessels)
- Distance to nearest junction/endpoint
- Neighborhood density

---

## Key Features

- **PyTorch Lightning**: Scalable, reproducible training with automatic logging
- **GPU Acceleration**: CuPy-based operations for filtering and mesh processing
- **2.5D Inference**: Multi-plane averaging (axial, coronal, sagittal) for 3D consistency
- **Dual Skeletonization**: 
  - *Thinning*: Fast, suitable for uniform vessel diameters
  - *Mesh Contraction*: Robust for varying diameters, preserves topology
- **Comprehensive Metrics**: 50+ morphometric and topological features

---

## Dependencies

Core dependencies (see `requirements.txt` for full list):

- `pytorch-lightning` - Training framework
- `segmentation-models-pytorch` - U-Net architectures
- `monai` - Medical imaging utilities
- `cupy` - GPU-accelerated computing
- `networkx` - Graph analysis
- `trimesh`, `pygel3d`, `skeletor` - Mesh processing
- `sknw` - Skeleton to graph conversion

---

## Citation

If you use VascuMap3D in your research, please cite:

```bibtex
@software{vascumap3d,
  title = {VascuMap3D: 3D Vascular Network Segmentation and Analysis},
  author = {VascuMap3D Team},
  year = {2024},
  url = {https://github.com/yourusername/VascuMap3D}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
