# VascuMap Technical Documentation

> Complete reference for the `vascumap` package. The [README](README.md) is the
> short user-facing guide; this file is the detailed technical companion.

---

## Table of contents

1. [Pipeline overview](#1-pipeline-overview)
2. [Entry point and launcher](#2-entry-point-and-launcher)
3. [Batch helpers (`pipeline.py`)](#3-batch-helpers-pipelinepy)
4. [`VascuMap` class (`core.py`)](#4-vascumap-class-corepy)
5. [`DeviceSegmentationApp` (`device_segmentation.py`)](#5-devicesegmentationapp-device_segmentationpy)
6. [`CurationApp` (`gui_region_detection.py`)](#6-curationapp-gui_region_detectionpy)
7. [Model inference (`models.py`)](#7-model-inference-modelspy)
8. [Skeletonisation and analysis (`skeletonisation.py`)](#8-skeletonisation-and-analysis-skeletonisationpy)
9. [Utilities (`utils.py`)](#9-utilities-utilspy)
10. [Output files and metrics](#10-output-files-and-metrics)
11. [System requirements and troubleshooting](#11-system-requirements-and-troubleshooting)

---

## 1. Pipeline overview

VascuMap quantifies 3D vasculature in microfluidic chips from brightfield
z-stacks (`.lif`, `.tif`, `.tiff`). One image runs through five stages:

```
.lif / .tif / .tiff
  │
  ├─ Stage 1: Device segmentation         (DeviceSegmentationApp)
  │     • Detect device ROI (Hough lines + per-pixel z-focus map)
  │     • Optional organoid mask (dark / light / off)
  │     • Crop XY to device interior, return per-z "vote" counts
  │
  ├─ Stage 2: Z-selection + isotropic resize  (VascuMap.preprocess)
  │     • Pick the longest contiguous run of z-slices with vote > 0,
  │       then expand downward toward the device floor
  │     • Resample to 5 µm voxels (anisotropic) for translation,
  │       then to 2 µm isotropic after segmentation
  │
  ├─ Stage 3: Translation + segmentation  (VascuMap.model_inference)
  │     • Pix2Pix turns brightfield into a synthetic vessel image
  │     • 2-D U-Net (mit_b5) predicts per-orthoplane probabilities
  │       (XY/XZ/YZ), averaged into a 3-D probability volume
  │     • Threshold → 3-D vessel mask
  │
  ├─ Stage 4: Crop + trim                  (VascuMap.postprocess + trim)
  │     • Map z-vote span to isotropic coordinates and crop XY edges
  │     • Drop edge slices whose fill exceeds 75 %
  │
  └─ Stage 5: Skeletonisation + analysis  (skeletonisation.clean_and_analyse)
        • Skeleton + sknw graph
        • Prune short branches, merge mid-nodes, drop OOB edges
        • Build branch-only graph (no sprout endpoints)
        • Per-branch / per-junction metrics + global headline panel
        • Curated 18-metric DataFrame for downstream PCA / clustering
```

Two ways to drive it:

- **GUI curation** (recommended) — `CurationApp` runs Stage 1 for every
  image up front, the user reviews/edits each ROI in napari, then the
  remaining stages are batch-run unattended.
- **Headless automatic** — `DeviceSegmentationApp.run_automatic` does
  Stage 1 without GUI, no human in the loop.

All densities are normalised by the **convex-hull volume** of the cleaned
vessel skeleton (`convex_hull_volume_um3`), not the raw chip-crop volume.

---

## 2. Entry point and launcher

### `python -m vascumap`

`__main__.py` is the canonical entry point. From the repository root:

```powershell
python -m vascumap
```

It performs, in order:

1. Build a `QApplication` (Qt event loop for napari + magicgui widgets).
2. Open `VascuMapLauncherGUI` and wait for it to close.
3. Load the Pix2Pix and U-Net checkpoints **once** (they are reused for
   every image in the batch via the `model_p2p` / `model_unet` kwargs).
4. Call `discover_jobs(...)` to enumerate `.lif` / `.tif` / `.tiff` files.
5. Call `filter_jobs(...)` to skip images whose output folder already
   exists in the optional skip directory.
6. Dispatch:
   - `mode == "gui"` → instantiate `CurationApp`, call `app.open()`, then
     `qapp.exec_()`. The curation app's `on_done` callback then calls
     `run_batch_from_curation`.
   - `mode == "auto"` → call `run_batch` directly.

### `VascuMapLauncherGUI` (`launcher_gui.py`)

A magicgui `Container` with five panels:

| Panel | Fields |
| --- | --- |
| Folders | `source_dir` (required), `output_dir` (required, auto-created), `skip_dir` (optional) |
| Organoid masking | radio: `No organoid masking` / `Dark organoid` / `Light organoid` / `Infer from name` |
| Options | `device_width_um` (default 35), `brightfield_channel` (default 0), `require_merged` (filter LIF images), `save_all_interim` |
| Models | `pix2pix_ckpt`, `unet_ckpt` (must point to existing files) |
| Run | `Use GUI (Curation)` / `Run Automatically` buttons |

Validation runs when either action button is pressed; errors are printed
into the in-window log and the launcher stays open until the inputs are
valid. The resulting `cfg` dict has these keys:

```python
{
  "source_dir", "output_dir", "skip_dir",
  "force_mask",          # False | "dark" | "light" | None ("infer")
  "device_width_um", "brightfield_channel",
  "require_merged", "save_all_interim",
  "pix2pix_ckpt", "unet_ckpt",
  "mode",                # "gui" | "auto"
}
```

`ORGANOID_TO_FORCE_MASK` maps the radio labels to the values consumed by
`pipeline.determine_mask_mode`.

---

## 3. Batch helpers (`pipeline.py`)

All four helpers are pure functions; nothing is mutated globally.

### `determine_mask_mode(file_path, image_name="", force_mask=None)`

Returns one of `"dark"`, `"light"`, or `False`. If `force_mask` is given
it is returned verbatim. Otherwise the filename + image name are scanned
for `marina` / `bead` keywords as a heuristic.

### `discover_jobs(source_dir, force_mask=None, require_merged=True)`

Scans `source_dir` for `.lif` / `.tif` / `.tiff` files. LIF files are
expanded into one job per contained image. When `require_merged` is true
(the default) only LIF images / TIFF stems containing `merged` are kept.

Returns `(source_path, image_files, jobs)` where each job is a tuple:

```python
(file_path: Path, image_index: int, mask_flag, image_name: str)
```

### `expected_output_name(file_path, image_index, image_name)`

Returns the output folder name the pipeline will use for a job:

- `.lif` → `"{lif_stem}_{image_name}_img{image_index}"` with `/`/`\\`
  replaced by `_`
- `.tif/.tiff` → `file_path.stem`

### `filter_jobs(jobs, skip_names)`

Drops jobs whose `expected_output_name` already appears in `skip_names`
(typically the set of folder names under `skip_dir`).

### `run_batch(jobs, output_base, device_width_um, brightfield_channel=0, save_all_interim=False, model_p2p=None, model_unet=None, start_index=1)`

Headless automatic batch. For each job it constructs a fresh `VascuMap`
in headless mode, sets `image_name`, and calls `vm.pipeline(...)`.
Failures are caught per-image and a `<output_base>/FAILED_diagnostics/`
folder collects per-image `_debug.txt` notes. Returns a list of
`(name, status, message)` tuples.

### `run_batch_from_curation(curated_jobs, output_base, save_all_interim=False, model_p2p=None, model_unet=None)`

GUI batch. Iterates only over jobs with `status == "curated"` and a
populated `finalised_outputs` dict, constructs `VascuMap(curated_outputs=...)`,
and calls `vm.pipeline(...)`. Skipped jobs are tallied and reported.

---

## 4. `VascuMap` class (`core.py`)

The orchestrator that wires Stage 2 → Stage 5. One instance handles one
image.

### Constructor

```python
VascuMap(
    pix2pix_model_path: str = r"...\luca_models\epoch=117-...ckpt",
    unet_model_path:    str = r"...\luca_models\best_full.pth",
    curated_outputs:    dict | None = None,
    image_source_path:  str | None = None,
    image_index:        int = 0,
    device_width_um:    float = 35.0,
    hough_line_length:  int = 400,
    mask_central_region        = False,    # False | "dark" | "light"
    brightfield_channel: int = 0,
    model_p2p           = None,            # pre-loaded Pix2Pix
    model_unet          = None,            # pre-loaded U-Net
    failure_output_dir: str | None = None,
)
```

Two mutually exclusive paths:

- **Curated** — pass `curated_outputs` (a dict produced by
  `CurationApp`, see §6). All Stage-1 fields are populated directly:
  `cropped_stack`, `device_width_um`, `pixel_size_um`, `z_votes`,
  `image_name`, `mask_central_region_enabled`, `cropped_organoid_mask_xy`,
  `image_source_path`, `image_index`.
- **Headless automatic** — pass `image_source_path` (must exist and be
  `.tif/.tiff/.lif`). A `DeviceSegmentationApp(enable_gui=False,
  line_length=hough_line_length)` is created and `run_automatic` is
  called; failures emit a `*_debug.txt` into `failure_output_dir`.

Pre-loading the heavy PyTorch models once (`model_p2p`, `model_unet`)
and reusing them across a batch saves seconds per image.

### Key instance attributes

| Attribute | Purpose |
| --- | --- |
| `cropped_stack` | Raw or 2 µm-resampled brightfield z-stack. |
| `pixel_size_um` | dict with `z_um`, `y_um`, `x_um`. |
| `z_votes` | `{global_z: vote_count}` from device segmentation. |
| `cropped_organoid_mask_xy` | Optional 2-D bool mask of the organoid region. |
| `vessel_pred_iso` | Pix2Pix translation, 2 µm isotropic. |
| `vessel_proba_iso` | U-Net 3-D probability map. |
| `vessel_mask_iso` | Final binary vessel mask. |
| `analysis_results` | Full dict returned by `clean_and_analyse` (see §8). |
| `app` | Backing `DeviceSegmentationApp` (auto mode only). |

### Methods

| Method | Description |
| --- | --- |
| `preprocess()` | Pick longest contiguous run of strong-vote z-slices, expand downward toward the device floor (160 µm minimum span), resample to 5 µm voxels for translation. |
| `model_inference(cropped_stack=None, device='cuda')` | Optionally fill the organoid region with the per-slice mean of the rest of the image, run Pix2Pix → 2.5× upscale (z) → U-Net `predict_mask_ortho` → `process_vessel_mask`. |
| `postprocess()` | Map the strong-vote sub-span to isotropic coordinates, crop `device_width_um` pixels off each XY edge. |
| `skeletonisation_and_analysis(voxel_size_um=(2,2,2))` | Build the organoid exclusion mask at the cropped resolution, zero-out vessels inside it, then call `clean_and_analyse`. Stores results on `self.analysis_results`. |
| `pipeline(output_dir=None, save_all_interim=False)` | Top-level driver — runs all stages, saves CSVs and overlay TIFFs (and optional intermediates), and writes a `*_debug.txt` whenever a stage gives up. |
| `write_debug_txt(out, name_prefix, error_category, body_lines)` | Internal helper used by `pipeline()` to record per-image failure metadata. |

`pipeline()` aborts gracefully (with a `*_debug.txt`) for the following
documented categories:

- `organoid_too_large` (organoid covers > 40 % of the image)
- `no_valid_z_range` (preprocess could not find any cropped stack)
- `no_strong_vote_planes` (postprocess found no contiguous run)
- `all_slices_trimmed` (every slice exceeded the 75 % fill threshold)

---

## 5. `DeviceSegmentationApp` (`device_segmentation.py`)

A 1600-line module that owns Stage 1 (device ROI detection, organoid
masking, z-focus map, cropping) and the napari widgets used by
`CurationApp`. The same class supports both GUI and headless use via
`enable_gui`.

### Constructor

```python
DeviceSegmentationApp(
    enable_gui:     bool  = True,
    low_frac:       float = 0.82,
    smooth_window:  int   = 5,
    bin_size:       float = 2.0,
    min_run_frac:   float = 0.25,
    typical_pct:    float = 50.0,
    line_length:    int   = 300,
    line_gap:       int   = 100,
    hough_threshold: int  = 120,
    mask_sigma:     float = 5.0,
)
```

Parameters control the per-pixel z-focus map (`low_frac`, `smooth_window`,
`bin_size`, `min_run_frac`, `typical_pct`), the Hough-line device-edge
detection (`line_length`, `line_gap`, `hough_threshold`), and the Gaussian
smoothing of the organoid mask (`mask_sigma`). `VascuMap` instantiates
this with `enable_gui=False` and the constructor's `line_length` set to
its own `hough_line_length` (default 400 — longer than the GUI default
because automatic mode needs to be more conservative).

### Key methods

- `list_images(path)` — populate `image_choice_map` from a `.lif` /
  `.tif` / `.tiff` source.
- `segment_and_view(image_choice, focus_downsample, focus_n_sampling,
  focus_patch, mask_central_region, clear_layers=True)` — run the full
  Stage-1 detection on one image and (optionally) push napari layers.
- `apply_device_width_layer(width_um)` — draw the inner/outer device
  geometry from a single µm width.
- `apply_crop_from_roi()` — crop the loaded stack to the current ROI and
  populate `cropped_xyz`.
- `run_automatic(image_source, image_index=0, device_width_um=35.0,
  mask_central_region=False, failure_output_dir=None)` — convenience for
  headless mode: lists images, runs `segment_and_view`, sets the device
  width, crops, and returns `get_cropped_outputs()`. On failure it
  writes a `*_debug.txt` into `failure_output_dir` (if given) and raises.
- `get_cropped_outputs()` — returns
  `(cropped_stack, device_width_um, pixel_size_um, z_votes, image_name,
    mask_central_region_enabled, cropped_organoid_mask_xy)`.
- `save_overlay_and_slice_tifs(name_prefix, run_suffix, output_dir=None)`
  — writes the per-image device geometry overlay TIFFs.

### Module-level helpers

- `read_voxel_size_um(image_path, image_index=0)` — best-effort µm/px
  reader for both LIF (via `liffile`) and TIFF (via `tifffile`).
- `um_to_xy_pixels(width_um, x_um, y_um)` — convert a µm width to
  per-axis pixel sizes.

---

## 6. `CurationApp` (`gui_region_detection.py`)

Multi-image napari curation GUI used in GUI mode.

```
Phase A: discover jobs → auto-detect device ROI + organoid mask for all
         → user navigates and curates in a single napari session
Phase B: close viewer → return curated jobs to run_batch_from_curation
```

### `CuratedJob` (dataclass)

Per-image curation state tracked through the GUI:

```python
@dataclass
class CuratedJob:
    source_path: Path
    image_index: int
    image_name:  str
    organoid_mode: str          # "dark" | "light" | "off"
    device_width_um: float
    status: str = "pending"     # "pending" | "curated" | "skip" | "failed"
    error_msg: str = ""

    inner_corners:    np.ndarray | None       # (4, 2) xy
    organoid_mask_xy: np.ndarray | None       # full-res 2-D bool
    pixel_size_um:    dict | None
    focus_plane:      np.ndarray | None
    z_step_um, y_step_um, x_step_um, xy_step_um
    focus_zmap_full:  np.ndarray | None
    focus_downsample: int = 4
    focus_n_sampling: int = 10
    focus_patch:      int = 50
    segment_debug:    dict | None
```

After the user accepts a job, `finalised_outputs` is populated:

```python
job.finalised_outputs = {
    "cropped_stack":               cropped_stack,
    "device_width_um":             job.device_width_um,
    "pixel_size_um":               job.pixel_size_um,
    "z_votes":                     z_votes,
    "image_name":                  job.image_name,
    "mask_central_region_enabled": job.organoid_mode != "off",
    "cropped_organoid_mask_xy":    cropped_organoid_mask,
    "source_path":                 str(job.source_path),
    "image_index":                 job.image_index,
}
```

This dict is exactly what `VascuMap(curated_outputs=...)` consumes.

### `CurationApp` constructor

```python
CurationApp(
    jobs,                           # list of (path, idx, mask_flag, name)
    device_width_um: float = 35.0,
    brightfield_channel: int = 0,
    default_organoid_mode: str = "infer",   # "infer" | "dark" | "light" | "off"
    on_done = None,                 # callback(curated_jobs) when user clicks Done
)
```

Internally it owns one persistent `DeviceSegmentationApp(enable_gui=False)`
which it reuses for every job's auto-detection step.

### Key methods

- `auto_detect_all()` — run `auto_detect_single` on every job. Successful
  jobs are pre-accepted (`status="curated"`), failures are recorded.
- `auto_detect_single(job)` — run device + organoid detection for one
  job and stash everything needed for review on `job`.
- `open()` — open the napari viewer with navigation widgets (Next / Prev /
  Accept / Skip / Done).
- `infer_organoid_mode(source_path, image_name)` — module-level
  heuristic mirroring `pipeline.determine_mask_mode`.

When the user clicks **Done**, every `status == "curated"` job has its
`finalised_outputs` populated and `on_done(jobs)` fires (which is wired
in `__main__` to `run_batch_from_curation`).

---

## 7. Model inference (`models.py`)

### `Pix2Pix`

Lightning-style wrapper around the brightfield → synthetic-vessel
generator (UNet generator + PatchGAN discriminator). Loaded from a `.ckpt`
file produced by the original training pipeline.

- `Pix2Pix(model_path)` — load the generator from a Lightning checkpoint.
- `Pix2Pix.predict(stack_bf, device='cuda', n_iter=1)` — run the generator
  on a 3-D brightfield stack and return a 3-D synthetic-vessel volume.

### `load_segmentation_model(unet_path)`

Loads the 2-D MONAI / `segmentation_models_pytorch` U-Net used for vessel
segmentation. Returns a model already moved to GPU with weights from
`unet_path` (`best_full.pth` by default). Wraps `adapt_input_model` to
remap older checkpoints onto the current architecture.

### `predict_mask_ortho(model, vessel_pred_iso, device='cuda')`

Run the 2-D U-Net on every slice along XY, XZ, and YZ planes and return
the average 3-D probability volume. This is the "2.5-D" inference that
gives 3-D-consistent vessel topology without a fully 3-D network.

### `process_vessel_mask(proba_volume, smooth=True)`

Threshold the probability volume into a binary mask, with optional
GPU-accelerated 3-D median smoothing via `median_filter_3d_gpu`.

---

## 8. Skeletonisation and analysis (`skeletonisation.py`)

The analytical heart of the pipeline. The single public entry point is
`clean_and_analyse`; everything else is either a helper used by it or a
metric/aggregation utility usable on the returned graphs.

### `clean_and_analyse(vasculature_segmentation, voxel_size_um=(2,2,2), organoid_mask=None)`

Runs:

1. Connected-component cleanup of the binary mask.
2. 3-D Euclidean distance transform.
3. Skeletonisation (`skimage.morphology.skeletonize_3d`).
4. Graph extraction with `sknw`.
5. Pruning via `prune_graph` (drops short/dead-end branches under
   length and EDT cutoffs) and `remove_mid_node` (collapses degree-2
   chains into single edges).
6. Out-of-bounds edge pruning (`prune_out_of_bounds_edges`) — drops
   edges that touch the image border or pass through `organoid_mask`.
7. Sprout classification (`classify_sprout_edges`) — degree-1 nodes are
   sprouts; degree ≥ 2 are junctions.
8. Branch-only graph (`build_branch_only_graph`) — sprout endpoints
   removed so junction degrees and branch metrics are unbiased.
9. Per-branch cross-sectional areas (`compute_cross_sectional_areas`).
10. Per-branch / per-junction DataFrames + headline aggregations.
11. Curated 18-metric DataFrame (`build_curated_analysis_metrics_df`).

Return dict keys:

| Key | Type | What it is |
| --- | --- | --- |
| `global_metrics` | dict | All scalar totals (volumes, counts, headline densities). |
| `global_metrics_df` | DataFrame | One-row version of `global_metrics`. |
| `branch_metrics_df` | DataFrame | Per-edge metrics on the *full* clean graph. |
| `branch_only_metrics_df` | DataFrame | Per-edge metrics on the branch-only graph. |
| `junction_metrics_df` | DataFrame | Per-junction metrics on the full graph. |
| `branch_only_junction_metrics_df` | DataFrame | Per-junction metrics on the branch-only graph. |
| `all_morphological_params_df` | DataFrame | One-row union of every scalar metric the module computes. |
| `analysis_metrics_df` | DataFrame | One-row curated 18-metric panel (see §10). |
| `voxel_size_um`, `chunk_size` | – | Run metadata. |
| `clean_segmentation`, `binary_edt`, `skeleton`, `area_image`, `skeleton_from_graph` | ndarray | Volumes used downstream / for napari. |
| `graph`, `pruned_graph`, `clean_graph`, `branch_only_graph` | networkx.Graph | Skeleton graphs at successive cleanup stages. |

### Helper functions worth knowing

| Function | Purpose |
| --- | --- |
| `safe_divide / safe_median / safe_percentile_spread` | Metric-aggregation guards that return `nan` instead of raising on empty inputs. |
| `trim_segmentation(seg, fill_threshold=0.75)` | Drop edge slices with > 75 % fill (over-segmentation guard). Used by `VascuMap.pipeline`. |
| `prune_graph(graph, area_3d, edt_cutoff=0.25, length_cutoff=25)` | Drop short dead-end edges. |
| `remove_mid_node(graph)` | Collapse degree-2 chains into single edges. |
| `classify_sprout_edges(graph)` | Tag each node as `sprout=True/False`. |
| `build_branch_only_graph(graph)` | Return a copy with sprouts and their edges removed. |
| `prune_out_of_bounds_edges(graph, image_shape, organoid_mask=None, ...)` | Remove border-touching / organoid-overlapping edges. |
| `compute_cross_sectional_areas(mask, skeleton, binary_edt, voxel_size_um)` | Cross-section area image used by branch metrics. |
| `compute_branch_metrics_df(graph, area_image, voxel_size_um, device_axis='x')` | Per-edge length, tortuosity, median CS area, orientation to device axis. |
| `compute_junction_metrics_df(graph, voxel_size_um, distance_threshold_um=500)` | Per-junction degree + nearest-neighbour distances. |
| `compute_all_morphological_params(...)` | Combine globals + branch/junction summaries into one row. |
| `summarize_network_headline_metrics(graph, voxel_size_um)` | 8 nearest-neighbour distance summaries (median/spread × {junction, sprout} × {skeleton, euclidean}). |
| `fractal_dimension_and_lacunarity(binary, ...)` | 3-D box-counting fractal dimension + lacunarity for the skeleton. |
| `orientation_to_device_axis_deg(pts_um, device_axis='x')` | Per-edge orientation in degrees from the device long axis. |
| `graph2image(graph, shape)` | Rasterise a graph back to a labelled volume (used for `_full_graph_skeleton.npy`). |
| `overlay_skeleton_on_seg(seg_bg, skeleton_volume)` | RGB overlay used by the visualisation helpers. |
| `generate_skeleton_overview_plot(...)` | Saves the per-image `*_skeleton_overview.png`. |

### `ANALYSIS_METRICS_COLUMNS`

Module-level constant listing the 18 columns of the curated panel — this
is the single source of truth and is what `build_curated_analysis_metrics_df`
selects out of `all_morphological_params_df`.

---

## 9. Utilities (`utils.py`)

| Function | Purpose |
| --- | --- |
| `scale(array)` | Min–max normalise to `[0, 1]` for model input. |
| `resize_dask(array, factors)` | Lazy / chunked anisotropic resampling used to go between native, 5 µm, and 2 µm voxels. |
| `cupy_chunk_processing(...)` | Chunked GPU processing helper used by the segmentation post-processing. |

---

## 10. Output files and metrics

### Always written to `<output_dir>/<image_name>/`

| File | Description |
| --- | --- |
| `<name>_overlay_geometry_0.tif` | Brightfield slice with detected device geometry overlaid. |
| `<name>_skeleton_overview.png` | Diagnostic figure (brightfield + segmentation + skeleton + organoid mask). |
| `<name>_analysis_metrics.csv` | One row, 18 curated metrics + image identifiers. |
| `<name>_all_morphological_params.csv` | One row, every scalar metric the analysis can produce. |
| `<name>_branch_metrics.csv` | One row per edge in the *full* clean graph. |

### Additional outputs when `save_all_interim=True`

| File | Description |
| --- | --- |
| `<name>_cropped_stack_aligned.npy` | Brightfield stack aligned to the analysis crop. |
| `<name>_vessel_translation_aligned.npy` | Pix2Pix synthetic vessel volume. |
| `<name>_vessel_mask.npy` | Pre-trim vessel mask. |
| `<name>_clean_segmentation.npy` | Largest-component-cleaned vessel mask. |
| `<name>_skeleton.npy` | 3-D skeleton from the clean graph. |
| `<name>_full_graph_skeleton.npy` | Rasterised full clean graph (label per edge). |
| `<name>_graph_nodes.npz` | `pts` (N×3) + `is_sprout` (N) for the clean graph nodes. |
| `<name>_clean_graph.pkl` | Pickled `networkx.Graph` of the clean graph. |
| `<name>_organoid_mask.npy` | Aligned organoid exclusion mask (only when masking is enabled). |

### Failure diagnostics

- Per-image failures inside `pipeline()` write
  `<output_dir>/<image_name>/<name>_debug.txt` with the failing stage,
  per-slice statistics, and a short explanation.
- Stage-1 failures from `run_batch` write into
  `<output_dir>/FAILED_diagnostics/<name>_debug.txt`.

### The curated 18-metric panel

`ANALYSIS_METRICS_COLUMNS` (defined in `skeletonisation.py`):

| Group | Metric | Notes |
| --- | --- | --- |
| **Density (5)** | `vessel_volume_fraction` | Vessel voxels / convex-hull voxels. |
| | `branch_length_per_volume` | Total branch length (µm) / convex-hull volume (µm³). |
| | `attached_sprouts_per_volume` | Sprouts connected to the network. |
| | `junctions_per_volume` | Branch points. |
| | `branches_per_volume` | Edges in the branch-only graph. |
| **Topology (2)** | `skeleton_fractal_dimension` | 3-D box-counting on the skeleton. |
| | `skeleton_lacunarity` | Companion to the fractal dimension. |
| **Branch geometry (4)** | `median_branch_length` | Branch-only graph. |
| | `spread_branch_length` | Inter-percentile spread (10–90). |
| | `median_branch_median_cs_area` | Median over branches of each branch's median CS area (µm²). |
| | `spread_branch_median_cs_area` | Inter-percentile spread. |
| **Tortuosity (2)** | `median_branch_tortuosity` | Polyline / Euclidean ratio. |
| | `spread_branch_tortuosity` | Inter-percentile spread. |
| **Junction connectivity (2)** | `median_branch_only_junction_degree` | Degree distribution of the branch-only graph. |
| | `spread_branch_only_junction_degree` | Inter-percentile spread. |
| **Orientation (2)** | `median_sprout_and_branch_orientation` | Degrees from the device long axis. |
| | `spread_sprout_and_branch_orientation` | Inter-percentile spread. |
| **Detachment (1)** | `floating_sprouts_per_volume` | Disconnected sprout count / convex-hull volume — proxy for vessel retraction. |

`all_morphological_params_df` is a strict superset of these columns and
also includes raw counts (`n_branches`, `n_junctions`, …), volumes
(`convex_hull_volume_um3`, `chip_volume_um3`, …), and the eight
`summarize_network_headline_metrics` keys.

---

## 11. System requirements and troubleshooting

### Required

- Windows 10/11 or Linux.
- Python 3.10+.
- CUDA-capable GPU. Pix2Pix + U-Net inference is GPU-only in practice
  (CPU fallback exists but is impractically slow for 3-D stacks).
- A working Qt environment (PyQt5 / PySide2 — provided by `napari`).

### Key dependencies

`napari`, `magicgui`, `qtpy`, `pytorch`, `pytorch-lightning`, `monai`,
`segmentation-models-pytorch` (U-Net `mit_b5` encoder), `cupy` and
`cupyx` for GPU-accelerated filtering, `scikit-image`, `dask`, `sknw`,
`networkx`, `liffile`, `tifffile`, `pandas`, `numpy`.

A working Conda specification lives in `oldclean_vascumap.yml` /
`env_backup_base_20260309_172159.yml` at the repository root.

### Common issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ImportError: liffile` | Missing PyPI package. | `pip install liffile`. |
| `RuntimeError: CUDA out of memory` during inference. | Stack too large or other GPU consumers. | Close other GPU apps; reduce input or run one image at a time. |
| Launcher closes immediately. | Validation failed; no `cfg`. | Re-open the launcher, check the in-window log for missing folders / checkpoints. |
| All images fail with `no_strong_vote_planes`. | Brightfield channel wrong, or organoid masking inverted. | Adjust **Brightfield Channel** and the **Organoid Masking** radio. |
| `_debug.txt` reports `all_slices_trimmed`. | Over-segmentation. | Verify the device crop in GUI mode; inspect `*_overlay_geometry_0.tif`. |
| Output folder already exists and is being skipped. | Folder name appears in `skip_dir`. | Remove it from `skip_dir` (or pick a different output base). |

### Re-visualising results

Open the saved `_clean_graph.pkl`, `_skeleton.npy`, and
`_cropped_stack_aligned.npy` in any napari session — see the top-level
`basic_napari_viewer.ipynb` for an example.

