# VascuMap

**3D vascular network analysis pipeline for microfluidic chip imaging**

VascuMap takes a brightfield microscopy stack of a vascularised microfluidic chip and returns a clean 3-D vessel mask, a topological graph of the vessel network, and a curated set of morphometric metrics suitable for PCA / clustering / cohort comparison. It combines a napari-based interactive curation GUI with deep-learning inference (3-D Pix2Pix image translation + 2.5-D U-Net segmentation) and graph-based skeleton analysis.

The package is intended to be run from the repository root with `python -m vascumap`, which opens a launcher GUI for selecting folders, model checkpoints, and run mode.

---

## 1. Quick start

From the repository root:

```bash
python -m vascumap
```

This opens the **launcher GUI**. Fill in:

| Field | Meaning |
|-------|---------|
| **Source Folder** | Folder containing `.lif`, `.tif`, or `.tiff` files to analyse |
| **Output Folder** | Where per-image result folders will be written |
| **Skip Folder** *(optional)* | Any image whose expected output folder name already exists here will be skipped (useful when resuming partial runs) |
| **Device Width (µm)** | Width of the microfluidic device — used both for outer-geometry padding during segmentation and for trimming XY edges of the analysed volume |
| **Brightfield Channel** | Channel index for multi-channel TIFFs |
| **Organoid masking** | `No organoid masking` / `Dark organoid` / `Light organoid` / `Infer from name`. Selects whether to detect and exclude a central organoid region from segmentation and analysis. |
| **Require 'merged' in name** | If on, only images whose name contains `merged` are processed (typical for LIF series) |
| **Save All Intermediates** | If on, also saves cropped stacks, vessel masks, skeletons, the cleaned graph, and the organoid mask |
| **Pix2Pix Checkpoint** | Path to a `.ckpt` Pix2Pix generator |
| **U-Net Checkpoint** | Path to a `.pth` 2-D U-Net checkpoint |

Then click one of:

- **Use GUI (Curation)** — opens napari, lets you review/edit the device ROI and organoid mask for every image, then runs the full pipeline unattended on the curated set.
- **Run Automatically** — skips curation; uses the automatic device segmentation for every image.

Progress is printed to the terminal. Failures on individual images are logged at the end of the batch rather than aborting the run.

---

## 2. Pipeline overview

For every image, the pipeline runs five stages:

```
.lif / .tif / .tiff
  │
  ▼  Stage 1: Device segmentation (device_segmentation.py)
        - Locate microfluidic device ROI
        - Crop XY to the device + outer margin
        - Compute per-z-slice "focus vote" counts
        - (optional) detect and mask the central organoid region
  │
  ▼  Stage 2: Preprocess (core.py → VascuMap.preprocess)
        - Pick the longest contiguous focused z-slab from the votes
        - Resample to 5 × 2 × 2 µm voxels (Z × Y × X)
  │
  ▼  Stage 3: Model inference (core.py → VascuMap.model_inference)
        - 3-D Pix2Pix: brightfield → pseudo-fluorescence
        - Resample to 2 µm isotropic
        - 2.5-D U-Net inference (axial + coronal + sagittal, averaged)
        - Hysteresis threshold → binary vessel mask
  │
  ▼  Stage 4: Postprocess (core.py → VascuMap.postprocess)
        - Trim to z-planes with strong focus votes (> 2)
        - Crop XY edges by `device_width_um` pixels
  │
  ▼  Stage 5: Skeletonisation & analysis (skeletonisation.py)
        - GPU smoothing + 3-D hole filling + EDT
        - 3-D skeletonisation → sknw graph
        - Prune sprouts/stubs, merge degree-2 nodes, trim border + organoid edges
        - Per-branch and per-junction metrics
        - Curated headline metrics for downstream PCA / clustering
```

All downstream analysis (graph, metrics, fractal dimension, cross-sectional areas) is performed on the 2 µm isotropic representation.

### 2.1 Z-vote system

`DeviceSegmentationApp` records, for each z-plane, how many sampled patches voted that plane as the most in-focus. `preprocess` takes the longest contiguous run of planes with ≥ 1 vote and expands downward then upward until it covers at least `min_span_um` (default 160 µm). `postprocess` then trims again to the longest run of *strong* votes (> 2), which removes barely-focused fringe planes from the analysed volume.

### 2.2 2.5-D segmentation

The U-Net is 2-D, but is applied along all three orthogonal cross-sections (axial / coronal / sagittal) and the three probability volumes are averaged. This gives much better 3-D consistency than axial-only inference at a fraction of the memory cost of a full 3-D model.

### 2.3 Graph cleaning

After `sknw.build_sknw` produces a raw graph from the skeleton:

1. **`prune_graph`** removes terminal branches whose endpoint EDT is small relative to the junction-side EDT (i.e. tapering noise stubs) and also removes very short stubs (`length_cutoff` voxels).
2. **`remove_mid_node`** merges degree-2 nodes so every node in the cleaned graph is either a junction (deg ≥ 3) or an endpoint (deg = 1).
3. **`prune_out_of_bounds_edges`** drops edges that pass within `vicinity_xy = 50` pixels of the XY border (device-wall artefacts) and, if an organoid mask was supplied, edges that pass through it.
4. A separate **branch-only graph** (sprouts removed) is also built so that branch-statistics aren't biased by short detached/attached sprouts.

---

## 3. Repository layout

```
vascumap/
├── __main__.py             Launcher entry point — `python -m vascumap`
├── launcher_gui.py         Initial settings GUI (folders, models, organoid mode)
├── pipeline.py             discover_jobs / filter_jobs / run_batch / run_batch_from_curation
├── core.py                 VascuMap class — preprocess → inference → postprocess → analysis → save
├── device_segmentation.py  DeviceSegmentationApp — napari ROI detection + cropping
├── gui_region_detection.py CurationApp — multi-image curation viewer
├── models.py               Pix2Pix (3-D GAN), 2.5-D U-Net loader, ortho prediction, post-processing
├── skeletonisation.py      clean_and_analyse + all metric / graph / curation helpers
├── utils.py                scale, resize_dask, cupy_chunk_processing
├── README.md               (this file)
├── DOCUMENTATION.md        Full technical reference
└── Analysis_README.md      Plain-language description of the analysis metrics
```

The `vascumap_output_analysis/` folder at the repository root contains exploratory notebooks (`plotting_outputs_*.ipynb`) and shared plotting helpers in `plotting.py`.

---

## 4. Output files

For each image named `{name}`, a sub-folder `{output_dir}/{name}/` is created containing:

### Always written

| File | Description |
|------|-------------|
| `{name}_overlay_geometry_0.tif` | RGB overlay of the brightfield mean projection with detected ROI: inner boundary in red, outer (device-width-padded) boundary in yellow, organoid region (if any) in cyan. |
| `{name}_analysis_metrics.csv` | One-row curated metric panel (see [§5](#5-output-metrics)). |
| `{name}_all_morphological_params.csv` | One-row dump of *every* metric computed (densities, junction stats, branch stats, percentiles, etc.). Use this if a metric you need is missing from the curated panel. |
| `{name}_branch_metrics.csv` | One row per graph edge (length, tortuosity, median cross-sectional area, orientation, etc.) — useful for per-branch distributions. |

### With **Save All Intermediates** turned on

| File | Description |
|------|-------------|
| `{name}_cropped_stack_aligned.npy` | Brightfield stack at 2 µm isotropic, after the same z- and XY-trim that the analysed volume received. |
| `{name}_vessel_translation_aligned.npy` | Pix2Pix pseudo-fluorescence volume (2 µm iso). |
| `{name}_vessel_mask.npy` | Raw binary vessel mask before smoothing/hole-filling. |
| `{name}_clean_segmentation.npy` | Final binary mask used for skeletonisation (smoothed + filled). |
| `{name}_skeleton.npy` | 1-voxel skeleton derived from the cleaned graph edges. |
| `{name}_full_graph_skeleton.npy` | Skeleton from the *raw* (unpruned) graph — useful when comparing pruning effects. |
| `{name}_graph_nodes.npz` | `pts` (N × 3) and `is_sprout` (N,) arrays for the cleaned graph. |
| `{name}_clean_graph.pkl` | Pickled NetworkX graph (cleaned). |
| `{name}_organoid_mask.npy` | XY exclusion mask used during analysis (only if organoid masking was on). |

Failed images, if any, leave a `FAILED_diagnostics/{name}_debug.txt` under the output root explaining what went wrong (which threshold step failed, missing voxel metadata, etc.).

---

## 5. Output metrics

`{name}_analysis_metrics.csv` is the **curated 18-metric panel** intended for cohort-level analysis (PCA, clustering, group comparisons). The schema is stable across images — missing values are filled with `NaN` so DataFrames concatenate cleanly.

Voxel size is 2 µm isotropic, so all lengths/areas/volumes are in µm units. Densities are normalised by the **convex hull volume** of the cleaned vessel skeleton (not the raw chip volume), so they are robust to differences in how much of the chip happened to be in focus.

### Density (5)
| Column | Meaning |
|--------|---------|
| `vessel_volume_fraction` | Vessel voxel volume / convex-hull volume of the analysed region. |
| `branch_length_per_volume` | Total *branch* (sprout-free) skeleton length per hull volume — length density. |
| `attached_sprouts_per_volume` | Count of sprouts attached to a junction, per hull volume. |
| `junctions_per_volume` | Junction count per hull volume. |
| `branches_per_volume` | Edge count per hull volume. |

### Topology (2)
| Column | Meaning |
|--------|---------|
| `skeleton_fractal_dimension` | Box-counting fractal dimension of the cleaned skeleton. |
| `skeleton_lacunarity` | Mean lacunarity (gap-structure heterogeneity) across box scales. |

### Branch geometry — branches only (4)
| Column | Meaning |
|--------|---------|
| `median_branch_length` | Median per-branch centreline length (µm). |
| `spread_branch_length` | P90 − P10 of per-branch length (µm). |
| `median_branch_median_cs_area` | Median of per-branch median cross-sectional area (µm²). |
| `spread_branch_median_cs_area` | P90 − P10 of the same (µm²). |

### Tortuosity — branches only (2)
| Column | Meaning |
|--------|---------|
| `median_branch_tortuosity` | Median path-length / chord-length ratio across branches. |
| `spread_branch_tortuosity` | P90 − P10 of branch tortuosity. |

### Junction connectivity — branch-only graph (2)
| Column | Meaning |
|--------|---------|
| `median_branch_only_junction_degree` | Median degree of junctions in the branch-only graph. |
| `spread_branch_only_junction_degree` | P90 − P10 of the same. |

### Orientation — sprouts + branches (2)
| Column | Meaning |
|--------|---------|
| `median_sprout_and_branch_orientation` | Median angle of edges relative to the device axis. |
| `spread_sprout_and_branch_orientation` | P90 − P10 of the same. |

### Vessel retraction / detachment proxy (1)
| Column | Meaning |
|--------|---------|
| `floating_sprouts_per_volume` | Sprouts not attached to any junction (likely retracted/detached vessels) per hull volume. |

`{name}_all_morphological_params.csv` additionally exposes raw counts, `chip_volume`, `convex_hull_volume`, `vessel_volume`, `total_vessel_length`, `average_vessel_volume`, the four nearest-neighbour distance summaries (`{median,spread}_{junction,sprout}_{skeleton,euclidean}_dist_nearest_{junction,endpoint}`), and the full per-junction / per-branch descriptive stats. See [Analysis_README.md](Analysis_README.md) for plain-language definitions of each parameter group.

---


For the full technical reference (class signatures, helper-function semantics, GPU-memory notes), see [DOCUMENTATION.md](DOCUMENTATION.md). For the plain-language explanation of every analysis parameter, see [Analysis_README.md](Analysis_README.md).
