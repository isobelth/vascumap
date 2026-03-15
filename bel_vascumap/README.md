# VascuMap

**3D Vascular Network Analysis Pipeline for Microfluidic Chip Imaging**

VascuMap is a Python pipeline for segmenting and analysing 3D vascular networks imaged inside microfluidic chips. It combines a napari-based interactive GUI for device segmentation with deep learning inference (3D Pix2Pix image translation + 2.5D U-Net segmentation) and graph-based network analysis.

---

## Overview

The pipeline processes `.lif`, `.tif`, or `.tiff` microscopy stacks through five main stages:

1. **Device Segmentation** – Locates and crops the microfluidic device region using a napari GUI or a fully automatic mode.
2. **Preprocess** – Selects a focus-informed z-range from per-plane vote counts and resamples the stack to a standard 5×2×2 µm voxel size.
3. **Model Inference** – Runs 3D Pix2Pix image translation (brightfield → fluorescence-like) and 2.5D U-Net segmentation to produce a binary vessel mask at 2 µm isotropic resolution.
4. **Postprocess** – Trims the volume to strongly-focused z-planes and crops XY edges by the device width.
5. **Skeletonisation and Analysis** – Cleans the mask, extracts a vessel skeleton, builds a NetworkX graph, and computes morphometric and topological metrics.

---

## Project Structure

```
bel_vascumap/
├── vascumap.py                  # VascuMap class + CLI entry point
├── gui_device_segmentation.py   # DeviceSegmentationApp (napari GUI + auto mode)
├── models.py                    # Pix2Pix (Generator, Discriminator), segmentation loaders
├── skeletonisation.py           # Graph construction, pruning, and metric computation
├── utils.py                     # Array utilities (scale, resize_dask, cupy_chunk_processing)
└── README.md
```

---

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (required; CuPy and GPU inference are mandatory)
- CUDA Toolkit 11.x or 12.x

### Setup

```bash
conda env create -f env_backup_base_20260309_172159.yml
conda activate vascumap
```

---

## Usage

### GUI Mode (interactive, single image)

```python
from vascumap import VascuMap

vm = VascuMap()          # Opens napari – segment device, then close the viewer
vm.pipeline(output_dir="outputs/")
```

Or via the command line:

```bash
python vascumap.py --output-dir outputs/
```

On startup, a napari window opens with a control panel on the right:
1. **Load images** – Browse to a `.lif`, `.tif`, or folder.
2. **Segment + View** – Detect the device boundary; optionally tick *Mask central region* to exclude an organoid or chip centre.
3. **Device width (um)** – Set the device exclusion margin.
4. **Create cropped aligned** – Confirm the crop and store the result.

Close the napari viewer to proceed with the rest of the pipeline.

### No-GUI Batch Mode

```bash
python vascumap.py --no-gui \
    --image-dir /path/to/microscopy/files \
    --output-dir /path/to/outputs \
    [--save-all-interim]
```

All `.lif`, `.tif`, and `.tiff` files in `--image-dir` are processed automatically. Each LIF sub-image is processed individually. Results are saved in per-image sub-folders under `--output-dir`.

---

## Output Files

For each processed image named `{name}`:

| File | Description |
|------|-------------|
| `{name}_overlay_geometry_0.tif` | 2D overlay showing detected device ROI boundaries |
| `{name}_cropped_stack_aligned.npy` | Brightfield stack at 2 µm isotropic resolution |
| `{name}_vessel_translation_aligned.npy` | Pix2Pix pseudo-fluorescence volume |
| `{name}_clean_segmentation.npy` | Cleaned binary vessel mask |
| `{name}_skeleton.npy` | 1-voxel-wide skeleton derived from the cleaned graph |
| `{name}_analysis_metrics.csv` | Global morphometric metrics (one row per image) |
| `{name}_organoid_mask.npy` | XY exclusion mask (only if *Mask central region* was used) |

With `--save-all-interim` (or `save_all_interim=True`):

| File | Description |
|------|-------------|
| `{name}_holes.npy` | Binary internal pore map |
| `{name}_hole_labels_per_slice.npy` | Per-slice labelled pore regions |
| `{name}_hole_distance_per_slice_um.npy` | Per-slice pore distance transform (µm) |
| `{name}_full_graph_skeleton.npy` | Skeleton before graph cleaning/pruning |
| `{name}_vessel_mask.npy` | Raw binary vessel mask (before cleaning) |
| `{name}_graph_nodes.npz` | Node coordinates + sprout flag |
| `{name}_clean_graph.pkl` | Serialised NetworkX vascular graph |

---

## Output Metrics

The `_analysis_metrics.csv` file contains one row per image.

### Volume & Coverage
| Metric | Description |
|--------|-------------|
| `chip_volume_um3` | Total analysed chip volume (µm³) |
| `vessel_volume_um3` | Total vessel volume (µm³) |
| `vessel_volume_fraction` | Vessel volume / chip volume |

### Network Topology
| Metric | Description |
|--------|-------------|
| `total_vessel_length_um` | Cumulative centreline length (µm) |
| `vessel_length_per_chip_volume_um_inverse2` | Vessel length density (µm⁻²) |
| `sprouts_per_vessel_length_um_inverse` | Sprout density along the network (µm⁻¹) |
| `junctions_per_vessel_length_um_inverse` | Branching frequency (µm⁻¹) |
| `sprouts_per_chip_volume_um_inverse3` | Sprout count per chip volume (µm⁻³) |
| `junctions_per_chip_volume_um_inverse3` | Junction count per chip volume (µm⁻³) |

### Network Complexity
| Metric | Description |
|--------|-------------|
| `skeleton_fractal_dimension` | Fractal dimension of the skeleton |
| `skeleton_lacunarity` | Lacunarity (gap structure) of the skeleton |

### Vessel Morphology
| Metric | Description |
|--------|-------------|
| `median_sprout_and_branch_tortuosity` | Median tortuosity (path length / straight-line distance) |
| `p90_minus_p10_sprout_and_branch_tortuosity` | Spread of tortuosity (P90 − P10) |
| `median_sprout_and_branch_median_cs_area_um2` | Median cross-sectional area (µm²) |
| `p90_minus_p10_sprout_and_branch_median_cs_area_um2` | Spread of cross-sectional area (µm²) |

### Junction & Sprout Distances
| Metric | Description |
|--------|-------------|
| `median_junction_dist_nearest_junction_um` | Median nearest-junction spacing (µm) |
| `p90_minus_p10_junction_dist_nearest_junction_um` | Spread of junction spacing (µm) |
| `median_sprout_dist_nearest_endpoint_um` | Median nearest-sprout spacing (µm) |
| `p90_minus_p10_sprout_dist_nearest_endpoint_um` | Spread of sprout spacing (µm) |

### Internal Pores
| Metric | Description |
|--------|-------------|
| `total_internal_pore_count` | Number of enclosed internal pores |
| `internal_pore_area_fraction_in_filled_vascular_area` | Pore area / filled vessel area |
| `median_internal_pore_area_um2` | Median pore area (µm²) |
| `p90_minus_p10_internal_pore_area_um2` | Spread of pore area (µm²) |
| `median_internal_pore_max_inscribed_radius_um` | Median max inscribed pore radius (µm) |
| `p90_minus_p10_internal_pore_max_inscribed_radius_um` | Spread of pore radius (µm) |
| `total_internal_pore_density_per_vessel_volume_um_inverse3` | Pore count per vessel volume (µm⁻³) |

---

## Key Features

- **Interactive napari GUI** for device segmentation, with a fully automatic batch fallback
- **Organoid/central region masking** – optionally exclude a defined XY region from all analysis steps
- **GPU-accelerated** image processing via CuPy (Gaussian smoothing, EDT, hole-filling, median filtering)
- **3D Pix2Pix** brightfield-to-fluorescence translation with sliding-window inference
- **2.5D segmentation** – U-Net inference on axial, coronal, and sagittal planes averaged for 3D consistency
- **Graph-based analysis** using `sknw` and NetworkX with automatic edge pruning and topology extraction
- **Comprehensive metrics** covering volume, length, tortuosity, cross-section, pores, and fractal properties
- **Batch CLI** for unattended processing of entire directories

---

## Dependencies

Core dependencies:

- `torch`, `pytorch-lightning` – deep learning
- `monai` – sliding-window inference
- `segmentation-models-pytorch` – U-Net with MiT-B5 encoder
- `napari`, `magicgui` – GUI framework
- `cupy`, `cupyx` – GPU-accelerated array processing
- `sknw`, `networkx` – skeleton graph extraction and analysis
- `dask` – memory-efficient array operations
- `liffile` – Leica `.lif` file reading
- `tifffile`, `scikit-image`, `scipy`, `numpy`, `pandas`
