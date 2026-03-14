# VascuMap

Automated 3D vascular network analysis pipeline for microfluidic chip microscopy data. Combines deep-learning image translation and segmentation with graph-based morphometric analysis.

## Pipeline Overview

```
Input (.lif / .tif / .tiff)
        │
        ▼
┌─────────────────────────┐
│ DeviceSegmentationApp   │  Load image, curved-plane refocus, segment
│ (GUI or headless)       │  device boundary, crop & align z-stack
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│                         │  Select best z-range via focus votes,
│ VascuMap.preprocess()   │  enforce minimum z-span, resample to
│                         │  ~2 µm isotropic
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│                         │  1. Pix2Pix 3D: brightfield → fluorescence
│  model_inference()      │  2. U-Net 2.5D: vessel probability map
│                         │  3. Hysteresis threshold → binary mask
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ postprocess()           │  Refine z-range to strongly-voted planes,
│                         │  trim device walls, save intermediates
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│ skeletonisation_and_    │  GPU smoothing, EDT, skeletonise, build
│ analysis()              │  graph, prune, compute 25+ vascular
│                         │  metrics → CSV
└─────────────────────────┘
```

