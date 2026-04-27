# VascuMap Technical Documentation

> **A Complete Reference for the bel_vascumap Pipeline**

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [VascuMap Class](#2-vascumap-class)
3. [Device Segmentation App](#3-device-segmentation-app)
4. [Model Inference](#4-model-inference)
5. [Skeletonisation and Analysis](#5-skeletonisation-and-analysis)
6. [Utilities](#6-utilities)
7. [Output Files and Metrics](#7-output-files-and-metrics)
8. [System Requirements and Installation](#8-system-requirements-and-installation)

---

# 1. Pipeline Overview

VascuMap is an end-to-end pipeline for quantifying 3D vascular networks inside microfluidic chips. It accepts brightfield microscopy stacks (`.lif`, `.tif`, `.tiff`) and produces morphometric metrics and visualisation outputs.

## 1.1 Data Flow

```
.lif / .tif / .tiff file
  -> Stage 1: Device Segmentation (DeviceSegmentationApp)
      - Detects device ROI
      - Crops XY using device boundaries
      - Computes per-plane z-vote counts
      - Optionally masks central organoid region

  -> cropped_stack (Z, Y, X)

  -> Stage 2: Preprocess (VascuMap.preprocess)
      - Selects focus-informed z-range
      - Resamples to 5x2x2 um voxel size

  -> standardised stack

  -> Stage 3: Model Inference (VascuMap.model_inference)
      - Pix2Pix 3D: brightfield -> fluorescence
      - Resample to 2 um isotropic
      - 2.5D U-Net segmentation
      - Hysteresis thresholding -> binary mask

  -> vessel_pred_iso, vessel_proba_iso, vessel_mask_iso

  -> Stage 4: Postprocess (VascuMap.postprocess)
      - Trims to strongly-focused z-planes
      - Crops XY edges by device_width_um

  -> trimmed volumes

  -> Stage 5: Skeletonisation and Analysis (VascuMap.skeletonisation_and_analysis)
      - GPU smoothing + hole filling + EDT
      - 3D skeletonisation (skimage)
      - sknw graph extraction
      - Prune + clean graph
      - Compute global morphometric metrics

  -> analysis_results dict

  -> Stage 6: Save Outputs (VascuMap.pipeline)
      - .npy volumes, .csv metrics, .tif overlays
```

## 1.2 Key Concepts

### Voxel Standardisation

Raw microscopy data is anisotropic (Z spacing ≠ XY spacing). The pipeline standardises in two steps:

| Stage | Voxel Size |
|-------|-----------|
| After preprocess | 5 × 2 × 2 µm (Z × Y × X) |
| After model inference | 2 × 2 × 2 µm isotropic |

All downstream analysis uses the 2 µm isotropic representation.

### Z-Vote System

During device segmentation, `DeviceSegmentationApp` generates a per-plane "vote count" (number of in-focus features detected per z-slice). The `preprocess` stage uses these votes to:
1. Select the **longest contiguous run** of planes with ≥ 1 vote (the well-focused slab).
2. Expand the range downward first, then upward, to meet a minimum `min_span_um` (default 160 µm).

The `postprocess` stage further trims to planes with **strong** votes (> 2), reducing noise from barely-focused edges.

### 2.5D Segmentation

The U-Net operates in 2D, but is run on all three orthogonal cross-sections (axial, coronal, sagittal) and the probability maps are averaged. This gives better 3D consistency than axial-only inference without the memory cost of a full 3D model.

### Graph Pruning

After `sknw` builds the initial graph from the skeleton:
1. **`prune_graph`**: Removes terminal branches where the endpoint-end of the branch has a small EDT relative to the junction end (i.e., branches that taper off into noise). A `length_cutoff` also removes very short stubs.
2. **`remove_mid_node`**: Merges degree-2 nodes (passthrough nodes) to produce a clean graph where every node is either a junction (degree ≥ 3) or an endpoint (degree = 1).
3. **`prune_out_of_bounds_edges`**: In a single pass, removes edges that pass within `vicinity_xy=50` pixels of the XY border (device-wall artefacts) and, if an organoid mask is supplied, edges passing through that exclusion zone. Isolated nodes are dropped afterwards.

---

# 2. VascuMap Class

**Source:** `vascumap.py`

The top-level workflow container. Handles the full pipeline from device segmentation to output saving.

## 2.1 Constructor

```python
vm = VascuMap(
    pix2pix_model_path: str,          # Path to .ckpt checkpoint
    unet_model_path: str,             # Path to .pth checkpoint
    use_device_segmentation_app: bool = True,
    # --- No-GUI only ---
    image_source_path: str | None = None,
    image_index: int = 0,
    device_width_um: float = 35.0,
    mask_central_region: bool = False,
    brightfield_channel: int = 0,
)
```

**GUI mode (`use_device_segmentation_app=True`):**
Launches `DeviceSegmentationApp` in napari. The constructor blocks until the viewer is closed, then collects the cropped stack and metadata.

**No-GUI mode (`use_device_segmentation_app=False`):**
Runs `DeviceSegmentationApp.run_automatic(...)` headlessly. Requires `image_source_path` to exist and be `.tif`/`.tiff`/`.lif`.

**Instance attributes set after construction:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `cropped_stack` | ndarray | XY-cropped brightfield stack (Z, Y, X) |
| `device_width_um` | float | Device width margin in µm |
| `pixel_size_um` | dict | Keys: `z_um`, `y_um`, `x_um` |
| `z_votes` | dict | Per-plane focus vote counts |
| `image_name` | str | Image identifier (from filename/metadata) |
| `mask_central_region_enabled` | bool | Whether organoid masking was used |
| `cropped_organoid_mask_xy` | ndarray or None | 2D organoid exclusion mask |

## 2.2 `preprocess()`

```python
vm.preprocess(
    cropped_stack=None,     # Defaults to self.cropped_stack
    pixel_size_um=None,     # Defaults to self.pixel_size_um
    z_votes=None,           # Defaults to self.z_votes
    min_span_um=160.0       # Minimum z-slab thickness (µm)
)
```

1. Parses `z_votes` into a `{int: int}` vote map (keys can be integers or `"z12"` strings).
2. Finds the longest contiguous run of z-planes with ≥ 1 vote.
3. Expands the range to cover at least `min_span_um` (extends downward first, then upward).
4. Crops `cropped_stack` to the selected z range.
5. Calls `resize_dask` to resample the stack to 5×2×2 µm (using `pixel_size_um`).

Sets `self.initial_z_range`, `self.cropped_stack`, `self._pre_z_start_global`, `self._pre_z_um`.

## 2.3 `model_inference()`

```python
vm.model_inference(
    cropped_stack=None,        # Defaults to self.cropped_stack
    device='cuda'
)
```

1. **Organoid masking** (if enabled): Fills the organoid XY region in each z slice with the slice's mean intensity, so the model sees only background there.
2. **Pix2Pix** (`self.model_p2p.predict`): Translates the brightfield stack to a fluorescence-like volume using sliding-window inference (ROI: 32×512×512, overlap: 75%/25%/25%). Output is resampled with factor `[2.5, 1, 1]` to produce `vessel_pred_iso` at 2 µm isotropic.
3. **U-Net segmentation** (`predict_mask_ortho`): Runs 2.5D inference along axial, coronal, and sagittal planes, averages the probability maps. Result stored as `vessel_proba_iso`.
4. **Thresholding** (`process_vessel_mask`): Applies 3D median filter (size=7) then hysteresis thresholding (low=0.2, high=0.5). Result stored as `vessel_mask_iso`.

## 2.4 `postprocess()`

```python
vm.postprocess()
```

1. Finds the **longest contiguous run** of z-planes with `initial_z_range[z] > 2` (strong votes).
2. Converts the strong-vote z-range from original voxel space to 2 µm isotropic indices.
3. Trims `vessel_pred_iso`, `vessel_proba_iso`, and `vessel_mask_iso` to `[z_start:z_stop, ptr:-ptr, ptr:-ptr]` where `ptr = int(device_width_um)` pixels.

Sets `self._z_start_final`, `self._z_stop_final`, `self._pixels_to_remove`.

## 2.5 `skeletonisation_and_analysis()`

```python
vm.skeletonisation_and_analysis(
    voxel_size_um=(2.0, 2.0, 2.0),
)
```

Calls `clean_and_analyse` from `skeletonisation.py`. If `mask_central_region_enabled`, also builds an aligned XY exclusion mask matching the final volume shape and passes it to `clean_and_analyse`.

Stores the result dict in `self.analysis_results` and prints the global metrics table.

## 2.6 `pipeline()`

```python
vm.pipeline(
    output_dir: str | Path | None = None,
    save_all_interim: bool = False
)
```

Runs all stages in sequence and saves outputs. See [Section 7](#7-output-files-and-metrics) for details.

## 2.7 Command-Line Interface

```bash
# GUI mode
python vascumap.py [--output-dir /path/to/outputs] [--save-all-interim]

# Batch / no-GUI mode
python vascumap.py --no-gui \
    --image-dir /path/to/microscopy/files \
    --output-dir /path/to/outputs \
    [--save-all-interim]
```

In batch mode, all `.lif`, `.tif`, and `.tiff` files in `--image-dir` are processed. LIF files with multiple sub-images are iterated. Each image gets its own sub-folder under `--output-dir`. Failures are logged at the end rather than aborting the run.

---

# 3. Device Segmentation App

**Source:** `device_segmentation.py`

`DeviceSegmentationApp` handles loading microscopy files, detecting the microfluidic device ROI, cropping the stack, and returning structured outputs.

## 3.1 Constructor

```python
app = DeviceSegmentationApp(
    enable_gui: bool = True,
    low_frac: float = 0.82,       # Lower bound of expected device intensity
    high_frac: float = 1.10,      # Upper bound
    smooth_window: int = 5,
    bin_size: float = 2.0,
    min_run_frac: float = 0.25,
    typical_pct: float = 50.0,
    line_length: int = 400,       # Hough line detection min length
    line_gap: int = 900,
    hough_threshold: int = 70,
    mask_sigma: float = 2.0,
    mask_frac_thresh: float = 0.70,
)
```

Always creates a `napari.Viewer`. When `enable_gui=False`, the viewer is hidden and the app runs headlessly.

## 3.2 GUI Widgets

The napari control panel (right dock) contains:

| Widget | Action |
|--------|--------|
| **Load images** | Browse to a `.lif`, `.tif`, or folder. Populates the image dropdown. |
| **Segment + View** | Runs device detection; optionally ticks *Mask central region*. Displays the detected device boundary in the viewer. |
| **Device width (um)** | Slider: sets device exclusion margin (µm). Updates the inner/outer ROI layers live. |
| **Create cropped aligned** | Confirms the crop and stores the result for collection by `VascuMap`. |

## 3.3 `run_automatic()`

```python
outputs = app.run_automatic(
    image_source: Path,
    image_index: int = 0,
    device_width_um: float = 35.0,
    mask_central_region: bool = False,
)
```

Programmatic equivalent of: load images → segment → set device width → crop. Raises `RuntimeError` if no valid crop is found.

Returns the same tuple as `get_cropped_outputs()`.

## 3.4 `get_cropped_outputs()`

Returns a tuple:
```python
(
    cropped_stack,           # np.ndarray (Z, Y, X)
    device_width_um,         # float
    pixel_size_um,           # dict {z_um, y_um, x_um}
    z_votes,                 # dict {int: int}  per-plane vote counts
    image_name,              # str
    mask_central_region_enabled,  # bool
    cropped_organoid_mask_xy,     # np.ndarray or None
)
```

## 3.5 `save_overlay_and_slice_tifs()`

```python
app.save_overlay_and_slice_tifs(
    name_prefix: str,
    run_suffix: int,
    output_dir: Path | None = None,
)
```

Saves a 2D RGB overlay TIFF showing the detected ROI boundaries drawn over the mean-projection of the last loaded image. Inner boundary = red, outer boundary = yellow.

## 3.6 Helper Functions

### `read_voxel_size_um`

```python
z_um, y_um, x_um = read_voxel_size_um(
    source_path: Path | None,
    source_is_lif: bool,
    selected_lif: Path | None = None,
    image_index: int | None = None,
)
```

Reads voxel size metadata from LIF or TIFF files:
- **LIF**: Uses `liffile.LifFile` → `LifImage.coords` (primary), then `asxarray().coords` (fallback). Returns `(z_um, y_um, x_um)` in micrometres.
- **TIFF**: Uses `tifffile` XResolution/YResolution tags; tries IJMetadata or ImageDescription for z spacing.
- Returns `(None, None, None)` on failure.

### `um_to_xy_pixels`

```python
x_px, y_px = um_to_xy_pixels(width_um, x_um, y_um)
```

Converts a physical width in µm to pixel counts using voxel size metadata. Returns `None` if inputs are invalid.

---

# 4. Model Inference

**Source:** `models.py`

## 4.1 Pix2Pix (3D GAN)

### `Generator`

3D U-Net using MONAI's `UNet`:

```
Channels: (32, 64, 128, 256, 512)
Strides:  (1, 2, 2, 2, 1)
Residual units: 3 per level
Dropout: configurable (default 0.4)
Activation: ReLU on output
```

### `Discriminator`

3D PatchGAN with 5 Conv3d layers. Input is the brightfield + fake/real fluorescence concatenated (2 channels). Output is a patch-level real/fake score.

### `Pix2Pix`

PyTorch Lightning module wrapping Generator and Discriminator.

**Loss functions:**
- **Generator**: `BCE(fake_label, ones) + λ·L1(pred, target) + λ·MSE(pred, target)` with λ=100
- **Discriminator**: `BCE(real_label, ones) + BCE(fake_label, zeros)`

**Optimisation:** Adam for both; CosineAnnealingWarmRestarts scheduler.

**Checkpoints:** Loaded by passing `model_path` to the constructor. The checkpoint `state_dict` is loaded with `strict=True`.

### `Pix2Pix.predict()`

```python
vessel_pred = model_p2p.predict(
    stack_bf: np.ndarray,   # (Z, Y, X) normalised to [0, 1]
    device: str,            # 'cuda' or 'cpu'
    n_iter: int = 1         # Monte Carlo averaging passes
)
```

Uses MONAI `SlidingWindowInferer` (ROI 32×512×512, overlap 0.75/0.25/0.25). Returns `np.ndarray` (Z, Y, X) clipped to [0, 1].

## 4.2 Segmentation Model

### `load_segmentation_model()`

```python
model = load_segmentation_model(
    model_path: str,       # .pth or .ckpt file
    device: str = 'cuda'
)
```

Loads a `smp.Unet` with `mit_b5` encoder, then calls `adapt_input_model` to convert the first layer from 3-channel to 1-channel (grayscale). Supports both raw state dicts and PyTorch Lightning checkpoints (strips `model.` prefix if present). Returns the model in eval mode.

### `adapt_input_model()`

```python
model = adapt_input_model(model)
```

Converts `encoder.patch_embed1.proj` from a 3-channel Conv2d to 1-channel by summing the RGB weight channels (TIMM-style adaptation).

### `predict_mask_ortho()`

```python
vessel_proba = predict_mask_ortho(
    model_smp,
    vessel_pred_iso: np.ndarray,   # (Z, Y, X)
    device: str
)
```

Runs sigmoid-activated 2D U-Net inference in three orientations:
- **Axial** (z plane): `SliceInferer` with `roi_size=(1024, 1024)`, Gaussian mode, overlap 0.5
- **Coronal** (y plane): `SliceInferer` with `roi_size=(256, 256)`, `spatial_dim=1`
- **Sagittal** (x plane): `SliceInferer` with `roi_size=(256, 256)`, `spatial_dim=2`

If the z dimension is < 256, pads with -1 before coronal/sagittal inference and crops back afterward.

Returns the mean of the three probability volumes.

### `process_vessel_mask()`

```python
mask = process_vessel_mask(vessel_proba, ortho=False)
```

- `ortho=True`: applies `median_filter_3d_gpu` (size=7, chunk 32×1024×1024) then hysteresis threshold (low=0.2, high=0.5)
- `ortho=False`: hysteresis threshold only (low=0.1, high=0.5)

### `median_filter_3d_gpu()`

```python
filtered = median_filter_3d_gpu(volume, size=3, chunk_size=(64, 64, 64))
```

Applies `cupyx.scipy.ndimage.median_filter` via `cupy_chunk_processing`.

---

# 5. Skeletonisation and Analysis

**Source:** `skeletonisation.py`

## 5.1 `clean_and_analyse()`

Main entry point. Runs the complete skeletonisation and metric computation pipeline.

```python
results = clean_and_analyse(
    vasculature_segmentation: np.ndarray,   # 3D binary mask (Z, Y, X)
    voxel_size_um=(2.0, 2.0, 2.0),
    organoid_mask=None,                  # 2D bool mask for organoid region
)
```

Returns a dict with keys:

| Key | Description |
|-----|-------------|
| `global_metrics` | dict of all scalar metrics |
| `global_metrics_df` | single-row DataFrame |
| `clean_segmentation` | float32 binary volume after smoothing+hole-filling |
| `binary_edt` | 3D Euclidean distance transform (µm) |
| `skeleton` | bool 3D skeleton from `skimage.morphology.skeletonize` |
| `graph` | raw `sknw` NetworkX graph |
| `area_image` | float64 cross-sectional area volume |
| `pruned_graph` | graph after `prune_graph` |
| `clean_graph` | graph after `remove_mid_node` + border/exclusion trimming |
| `skeleton_from_graph` | uint8 skeleton derived from `clean_graph` edges |

**Processing steps:**
1. Subtract exclusion mask from segmentation (if provided).
2. GPU Gaussian smooth (σ=3) + threshold at 0.5 → `clean_segmentation`.
3. GPU binary fill holes.
4. GPU EDT (returns µm distances with `sampling=voxel_size_um`).
5. Dask-parallel `skeletonize_3d` with depth=(2,2,2) overlap.
6. `sknw.build_sknw` to build graph.
7. `compute_cross_sectional_areas` at skeleton points.
8. Prune, clean, trim border/exclusion edges.
9. Compute all metrics.

## 5.2 Graph Helper Functions

### `prune_graph(graph, area_3d, edt_cutoff=0.20, length_cutoff=25)`

Iteratively removes terminal branches. A branch from endpoint `e` to junction `j` is removed if:
- `length ≤ length_cutoff`, OR
- `mean_edt(tip_20_percent) / edt(junction_point) > edt_cutoff`

This eliminates thin noise stubs while keeping meaningful sprouts.

### `remove_mid_node(graph)`

Iteratively removes degree-2 nodes by merging their two incident edges into one. Edge point coordinates are concatenated in the correct orientation (using pairwise distance checks).

### `prune_out_of_bounds_edges(graph, image_shape, organoid_mask=None, vicinity_xy=50, inplace=False)`

Single-pass edge pruner: removes any edge with a point within `vicinity_xy` pixels of the XY border, or (if `organoid_mask` is provided) any point falling inside that mask. Isolated nodes are removed afterwards. Pass `inplace=True` to mutate the input graph and avoid the copy.

## 5.3 Metric Functions

### `compute_cross_sectional_areas(mask, skeleton, binary_edt, voxel_size_um)`

At each skeleton voxel `(z, y, x)`:
- `r_major = EDT_2D(y, x)` — 2D distance to vessel wall in the XY max projection
- `r_minor = EDT_3D(z, y, x)` — 3D distance to vessel wall
- `area = π × r_major × r_minor`

Returns a 3D array with areas only at skeleton locations.

### `summarize_network_headline_metrics(graph, voxel_size_um=(2.0, 2.0, 2.0))`

Computes nearest-neighbour distance summaries between same-type nodes
(junction↔junction, sprout↔sprout) using **both** distance modes in a
single call:

- `skeleton` — Dijkstra path length along the vessel graph, weighted by
  polyline µm length
- `euclidean` — straight-line µm distance

Returns a dict with eight keys of the form
`{median,spread}_{junction,sprout}_{skeleton,euclidean}_dist_nearest_{junction,endpoint}`.

### `compute_internal_pore_headline_metrics(mask, ...)`

Per-slice pore detection:
1. Fill holes in each z-slice: `filled_slice = binary_fill_holes(vessel_slice)`
2. `pores = filled_slice & ~vessel_slice`
3. Label connected components; filter by `min_pore_area_um2` and `max_pore_area_fraction_of_slice`
4. GPU EDT for inscribed radius

Returns counts, area fractions, and inscribed radius distributions.

### `fractal_dimension_and_lacunarity(binary, ...)`

Box-counting method over log-spaced scales from `2^max` down to 1:
- `D = slope(log N vs log 1/ε)` where N = non-empty boxes
- `λ = mean(Var(counts)/Mean(counts)²)` over all scales

---

# 6. Utilities

**Source:** `utils.py`

## `scale(arr)`

Min-max normalises `arr` to [0, 1]. Operates on float32.

## `resize_dask(stack, rescale_factor)`

Rescales a 3D numpy array using Dask + `skimage.transform.resize` (cubic interpolation). Processes in chunks to avoid loading the entire array into memory at once.

```python
rescaled = resize_dask(stack, [z_factor, y_factor, x_factor])
```

## `cupy_chunk_processing(volume, processing_func, chunk_size, overlap, **kwargs)`

Applies any CuPy-compatible function to a 3D volume in overlapping GPU chunks:

1. Iterates over (z, y, x) chunk positions.
2. Copies chunk + overlap halo to GPU.
3. Calls `processing_func(chunk_gpu, **kwargs)`.
4. Writes back only the non-overlapping core region to the CPU result array.
5. Frees GPU memory after each chunk.

Used for Gaussian filtering, hole filling, EDT, and median filtering throughout the pipeline.

---

# 7. Output Files and Metrics

## 7.1 Output Files

`VascuMap.pipeline()` creates the following files in `output_dir`:

| File | Description |
|------|-------------|
| `{name}_overlay_geometry_0.tif` | RGB overlay showing inner (red) and outer (yellow) ROI boundaries |
| `{name}_cropped_stack_aligned.npy` | Brightfield at 2 µm isotropic (post-preprocess + z/XY trim) |
| `{name}_vessel_translation_aligned.npy` | Pix2Pix pseudo-fluorescence volume |
| `{name}_clean_segmentation.npy` | Smoothed + hole-filled binary vessel mask |
| `{name}_skeleton.npy` | 1-voxel skeleton from clean graph edges (uint8) |
| `{name}_analysis_metrics.csv` | One-row global metrics DataFrame |
| `{name}_organoid_mask.npy` | XY exclusion mask (uint8, only if organoid masking used) |

With `save_all_interim=True`:

| File | Description |
|------|-------------|
| `{name}_holes.npy` | Binary internal pore map (uint8) |
| `{name}_hole_labels_per_slice.npy` | Per-slice integer pore labels |
| `{name}_hole_distance_per_slice_um.npy` | Per-slice pore EDT in µm (float32) |
| `{name}_full_graph_skeleton.npy` | Skeleton from raw (pre-prune) graph (int32) |
| `{name}_vessel_mask.npy` | Raw binary vessel mask before cleaning |
| `{name}_graph_nodes.npz` | `pts` (N×3 coordinates) and `is_sprout` (N bool) arrays |
| `{name}_clean_graph.pkl` | Serialised NetworkX clean graph |

## 7.2 Global Metrics (analysis_metrics.csv)

All lengths and areas are in µm or µm-derived units (voxel size = 2 µm isotropic).

### Volume
| Metric | Description |
|--------|-------------|
| `chip_volume_um3` | Total analysed chip volume (µm³), minus exclusion mask if used |
| `vessel_volume_um3` | Clean segmentation voxel count × 8 µm³ |
| `vessel_volume_fraction` | `vessel_volume_um3 / chip_volume_um3` |

### Network Topology
| Metric | Description |
|--------|-------------|
| `total_vessel_length_um` | Cumulative clean graph edge length (µm) |
| `vessel_length_per_chip_volume_um_inverse2` | Length density (µm²) |
| `sprouts_per_vessel_length_um_inverse` | Sprout count / total length (µm¹) |
| `junctions_per_vessel_length_um_inverse` | Junction count / total length (µm¹) |
| `sprouts_per_chip_volume_um_inverse3` | Sprout count / chip volume (µm³) |
| `junctions_per_chip_volume_um_inverse3` | Junction count / chip volume (µm³) |

### Complexity
| Metric | Description |
|--------|-------------|
| `skeleton_fractal_dimension` | Box-counting fractal dimension of clean skeleton |
| `skeleton_lacunarity` | Mean lacunarity across box scales |

### Vessel Morphology
| Metric | Description |
|--------|-------------|
| `median_sprout_and_branch_tortuosity` | Median path/chord ratio across all edges |
| `p90_minus_p10_sprout_and_branch_tortuosity` | P90−P10 spread |
| `median_sprout_and_branch_median_cs_area_um2` | Median of per-edge median cross-sectional areas (µm²) |
| `p90_minus_p10_sprout_and_branch_median_cs_area_um2` | P90−P10 spread |

### Spatial Distances
| Metric | Description |
|--------|-------------|
| `median_junction_dist_nearest_junction_um` | Median pairwise nearest-junction distance (µm), Dijkstra or Euclidean |
| `p90_minus_p10_junction_dist_nearest_junction_um` | P90−P10 spread |
| `median_sprout_dist_nearest_endpoint_um` | Median pairwise nearest-endpoint distance (µm) |
| `p90_minus_p10_sprout_dist_nearest_endpoint_um` | P90−P10 spread |

### Internal Pores
| Metric | Description |
|--------|-------------|
| `total_internal_pore_count` | Total pores across all z-slices |
| `internal_pore_area_fraction_in_filled_vascular_area` | Pore area / filled vessel area |
| `median_internal_pore_area_um2` | Median pore area (µm²) |
| `p90_minus_p10_internal_pore_area_um2` | P90−P10 spread (µm²) |
| `median_internal_pore_max_inscribed_radius_um` | Median max inscribed radius (µm) |
| `p90_minus_p10_internal_pore_max_inscribed_radius_um` | P90−P10 spread |
| `total_internal_pore_density_per_vessel_volume_um_inverse3` | Pore count / vessel volume (µm³) |

---

# 8. System Requirements and Installation

## 8.1 Prerequisites

| Component | Requirement |
|-----------|-------------|
| Python | 3.10+ |
| GPU | CUDA-capable; ≥ 8 GB VRAM recommended |
| CUDA Toolkit | 11.x or 12.x |
| RAM | 16 GB minimum; 32 GB+ recommended |

## 8.2 Installation

```bash
conda env create -f env_backup_base_20260309_172159.yml
conda activate vascumap
```

## 8.3 GPU Memory Notes

- The pipeline does not tile the Pix2Pix inference — the full stack is processed as a single sliding-window call. Very large stacks (> ~500 z-slices) may require additional chunking.
- `cupy_chunk_processing` default chunk size is `(64, 512, 512)` with overlap `(15, 15, 15)`. Reduce `chunk_size` if CUDA OOM errors occur during skeletonisation.
- GPU memory is explicitly freed (`torch.cuda.empty_cache()`, `cp.get_default_memory_pool().free_all_blocks()`) between major stages.

## 8.4 Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| `CUDA out of memory` during model inference | Stack too large | Reduce z-span or XY size of input |
| `CUDA out of memory` during skeletonisation | Chunk size too large | Pass smaller `chunk_size` to `cupy_chunk_processing` |
| Empty/flat `analysis_metrics.csv` | No strong vote planes found | Check that device segmentation produced non-zero `z_votes` |
| `ValueError: z_step_um must be positive` | Missing voxel metadata | Verify LIF/TIFF metadata; provide `pixel_size_um` manually if needed |
| All metrics NaN | Empty clean graph (no vessels detected) | Inspect `clean_segmentation.npy`; check thresholds in `process_vessel_mask` |

