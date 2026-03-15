# VascuMap3D Technical Documentation

> **A Complete Guide to 3D Vascular Network Segmentation and Analysis**

---

## Table of Contents

1. [Introduction & Background](#1-introduction--background)
   - [1.1 What is VascuMap3D?](#11-what-is-vascumap3d)
   - [1.2 Pipeline Overview](#12-pipeline-overview)
   - [1.3 Key Concepts](#13-key-concepts)
   - [1.4 System Requirements](#14-system-requirements)

2. [Detailed Workflow](#2-detailed-workflow)
   - [2.1 Training](#21-training)
     - [2.1.1 Image Translation (3D Pix2Pix)](#211-image-translation-3d-pix2pix)
     - [2.1.2 Segmentation (2D U-Net)](#212-segmentation-2d-u-net)
   - [2.2 Inference Pipeline](#22-inference-pipeline)
     - [2.2.1 Data Loading & Voxel Normalization](#221-data-loading--voxel-normalization)
     - [2.2.2 Preprocessing & Cropping](#222-preprocessing--cropping)
     - [2.2.3 Image Translation Inference](#223-image-translation-inference)
     - [2.2.4 Segmentation Inference (2.5D)](#224-segmentation-inference-25d)
   - [2.3 Graph Analysis](#23-graph-analysis)
     - [2.3.1 Skeletonization](#231-skeletonization)
     - [2.3.2 Graph Construction](#232-graph-construction)
     - [2.3.3 Metrics Computation](#233-metrics-computation)
     - [2.3.4 Output Files](#234-output-files)

3. [Function Reference](#3-function-reference)
   - [3.1 Image Translation Module](#31-image-translation-module)
   - [3.2 Segmentation Module](#32-segmentation-module)
   - [3.3 Pipeline Module](#33-pipeline-module)
   - [3.4 Graph Analysis Module](#34-graph-analysis-module)
   - [3.5 Utilities](#35-utilities)

4. [Metrics Reference](#4-metrics-reference)
   - [4.1 Global Metrics (CSV)](#41-global-metrics-csv)
   - [4.2 Vessel Metrics (HDF5)](#42-vessel-metrics-hdf5)
   - [4.3 Junction Metrics (HDF5)](#43-junction-metrics-hdf5)

5. [Appendix](#5-appendix)
   - [5.1 Mathematical Formulas](#51-mathematical-formulas)
   - [5.2 Troubleshooting](#52-troubleshooting)
   - [5.3 Contributing](#53-contributing)

---

# 1. Introduction & Background

## 1.1 What is VascuMap3D?

VascuMap3D is an end-to-end computational pipeline for analyzing 3D vascular networks from microscopy images. It solves a common challenge in biological imaging: extracting quantitative morphometric data from complex, branching vessel structures.

**Target users:**
- **Biologists** studying angiogenesis, vasculogenesis, or tumor vasculature
- **Biomedical engineers** developing tissue engineering constructs
- **Computational scientists** building on or extending the pipeline

## 1.2 Pipeline Overview

The pipeline processes raw microscopy images through four main stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VascuMap3D Pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1: IMAGE TRANSLATION                                                 │
│   ──────────────────────────                                                 │
│   Input:  Brightfield Z-stack (.lif file)                                   │
│   Output: Fluorescence-like volume                                          │
│   Method: 3D Pix2Pix GAN                                                    │
│                                                                              │
│   Why? Brightfield images have low contrast. We translate them to           │
│   fluorescence-like images where vessels are clearly visible.               │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   STAGE 2: SEGMENTATION                                                      │
│   ─────────────────────                                                      │
│   Input:  Fluorescence-like volume                                          │
│   Output: Binary vessel mask                                                │
│   Method: 2D U-Net with 2.5D inference (3-plane averaging)                  │
│                                                                              │
│   Why? 2D models are faster and have better pretrained weights.             │
│   We run inference on axial, coronal, and sagittal planes, then average.    │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   STAGE 3: SKELETONIZATION                                                   │
│   ────────────────────────                                                   │
│   Input:  Binary vessel mask                                                │
│   Output: 1-pixel-wide skeleton                                             │
│   Method: Morphological thinning OR mesh contraction                        │
│                                                                              │
│   Why? We need to reduce vessels to centerlines to extract topology.        │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   STAGE 4: GRAPH ANALYSIS                                                    │
│   ───────────────────────                                                    │
│   Input:  Skeleton image                                                    │
│   Output: NetworkX graph + CSV/HDF5 metrics                                 │
│   Method: sknw graph extraction + custom metric computation                 │
│                                                                              │
│   Why? Graphs let us compute topological metrics: branching, tortuosity,    │
│   vessel lengths, fractal dimension, etc.                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Data flow summary:**
```
.lif file → Brightfield Z-stack → Pseudo-fluorescence → Binary mask → Skeleton → Graph → Metrics
```

## 1.3 Key Concepts

### Voxel Size and Isotropy

Raw microscopy data has **anisotropic voxels** (Z spacing ≠ XY spacing). This causes problems for skeletonization and metric computation. VascuMap3D handles this with two-stage rescaling:

```
Original acquisition:     Variable (depends on microscope settings)
        ↓ Stage 1
Reference voxel size:     5 × 2 × 2 µm (Z × Y × X) - Standardized
        ↓ Stage 2  
Isotropic for analysis:   2 × 2 × 2 µm (Z × Y × X) - Same in all dimensions
```

**Why 5×2×2 µm as reference?**
- Z = 5 µm: Typical Z-step for confocal/brightfield
- Y = X = 2 µm: Common lateral resolution at 10-20× magnification

**Why make it isotropic?**
- Skeletonization assumes equal spacing in all dimensions
- Length/tortuosity metrics need consistent units
- Cross-sectional area requires isotropic sampling

### Graph Representation

Vascular networks are naturally represented as graphs:
- **Nodes** = Junction points (where vessels meet) + Endpoints (sprout tips)
- **Edges** = Vessel segments between nodes
- **Edge attributes** = Length, diameter, tortuosity, orientation

### Key Terminology

| Term | Definition |
|------|------------|
| **Vessel segment** | A continuous tube between two junction points |
| **Junction** | Where vessels meet or bifurcate (node degree > 2) |
| **Endpoint/Sprout tip** | Terminal point of a vessel (node degree = 1) |
| **Sprout** | Vessel segment connected to an endpoint |
| **Branch** | Vessel segment between two junctions |
| **Tortuosity** | Path length / straight-line distance (≥1, higher = more curved) |
| **EDT** | Euclidean Distance Transform - distance to nearest boundary |

## 1.4 System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.9 | 3.10+ |
| GPU | 8GB VRAM | 16GB+ VRAM |
| RAM | 16GB | 32GB+ |
| CUDA | 11.x | 12.x |

**Installation:**
```bash
git clone <repository_url>
cd VascuMap3D
pip install -r requirements.txt
```

---

# 2. Detailed Workflow

## 2.1 Training

> **Note for beginners:** This section explains how to train the deep learning models. If you're new to deep learning, don't worry - we'll explain the key concepts along the way. Training requires a GPU with at least 8GB of memory.

### 2.1.1 Image Translation (3D Pix2Pix)

#### What is Image Translation?

Image translation is like teaching a computer to "translate" one type of image into another - similar to how Google Translate converts English to French. In our case:

- **Input**: Brightfield microscopy (low contrast, vessels hard to see)
- **Output**: Fluorescence-like image (high contrast, vessels clearly visible)

The model learns this translation by looking at many paired examples: "here's a brightfield image, and here's what the same sample looks like in fluorescence."

#### How Does Pix2Pix Work?

Pix2Pix is a type of **Generative Adversarial Network (GAN)** - a clever training setup with two neural networks competing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Pix2Pix Architecture                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   GENERATOR (The Artist)                                                     │
│   ──────────────────────                                                     │
│   • Takes brightfield image as input                                        │
│   • Tries to create a convincing fluorescence-like image                    │
│   • Structure: 3D U-Net (encodes image, then decodes to output)            │
│   • Goal: Fool the discriminator AND match the real fluorescence           │
│                                                                              │
│                              ↓                                               │
│                                                                              │
│   DISCRIMINATOR (The Critic)                                                 │
│   ──────────────────────────                                                 │
│   • Sees pairs: (brightfield + generated) vs (brightfield + real)          │
│   • Tries to tell which fluorescence image is real vs fake                 │
│   • Structure: 3D PatchGAN (looks at small patches, not whole image)       │
│   • Goal: Correctly identify fake images                                   │
│                                                                              │
│   TRAINING LOOP                                                              │
│   ─────────────                                                              │
│   1. Generator creates fake fluorescence from brightfield                   │
│   2. Discriminator tries to spot the fake                                   │
│   3. Generator improves to fool discriminator better                        │
│   4. Discriminator improves to catch fakes better                           │
│   5. Repeat until generator makes realistic images                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Why 3D?

Unlike standard Pix2Pix which works on 2D images, we use **3D convolutions** because:
- Vessels are 3D structures - context from adjacent Z-slices helps
- Maintains spatial coherence across the Z-stack
- Better handles vessels that are tilted or curved through the volume

#### Loss Functions Explained

The generator is trained with multiple loss functions working together:

| Loss | What it Does | Why it Matters |
|------|-------------|----------------|
| **Adversarial (BCE)** | Encourages realistic-looking output | Makes images that "look right" |
| **L1 (Mean Absolute Error)** | Penalizes pixel-by-pixel differences | Ensures correct brightness/structure |
| **MSE (Mean Squared Error)** | Like L1 but penalizes large errors more | Prevents major mistakes |

Combined: `Total Loss = Adversarial + 100×L1 + 100×MSE`

The factor of 100 (λ) gives more weight to matching the actual image than just fooling the discriminator.

#### Input Data Format

You need **paired** 3D volumes - each brightfield volume must have a corresponding fluorescence volume:

```
/data/
├── input/                          # Brightfield volumes
│   ├── sample_001.nii.gz
│   ├── sample_002.nii.gz
│   └── ...
└── target/                         # Fluorescence volumes (ground truth)
    ├── sample_001.nii.gz           # MUST match input filenames exactly
    ├── sample_002.nii.gz
    └── ...
```

**Volume requirements:**

| Property | Requirement | Why |
|----------|-------------|-----|
| Format | NIfTI (`.nii.gz`) | Standard medical imaging format |
| Dimensions | 3D (Z, Y, X) | Typically 16-32 × 256 × 256 voxels |
| Values | Normalized to [0, 1] | Neural networks work best with small numbers |
| Data type | Float32 | Sufficient precision, reasonable memory |
| Voxel size | Should match | Input and target must have same voxel dimensions |

**Preparing your data:**

```python
import nibabel as nib
import numpy as np

# Load your 3D volume
volume = ...  # Your data loading

# Normalize to [0, 1]
volume = (volume - volume.min()) / (volume.max() - volume.min())

# Save as NIfTI
nifti_img = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
nib.save(nifti_img, 'sample_001.nii.gz')
```

#### Training Command

```bash
python -m vascumap.models.image_translation.train \
    --input_path /path/to/input \
    --target_path /path/to/target \
    --model_path /path/to/save \
    --batch_size 4 \
    --epochs 200
```

**Key parameters:**

| Parameter | Default | What it Does | Recommendations |
|-----------|---------|--------------|-----------------|
| `batch_size` | 4 | Samples per training step | Lower if out of memory |
| `epochs` | 200 | Full passes through dataset | More epochs = longer training |
| `generator_lr` | 1e-3 | Generator learning rate | Don't change unless needed |
| `discriminator_lr` | 1e-6 | Discriminator learning rate | Much lower than generator |

**Key functions:** [`ImageDataset3D`](#imagedataset3d), [`Generator`](#generator), [`Discriminator`](#discriminator), [`Pix2Pix`](#pix2pix)

#### Monitoring Training

Launch TensorBoard to visualize training progress:

```bash
tensorboard --logdir /path/to/save
```

Then open `http://localhost:6006` in your browser.

**Metrics to watch:**

| Metric | What it Means | Good Values | Concerning Signs |
|--------|---------------|-------------|------------------|
| `train_g_loss` | Generator total loss | Decreasing over time | Stuck high or increasing |
| `train_d_loss` | Discriminator loss | Stable around 0.5-1.0 | Very low (<0.1) or very high (>2) |
| `val_g_psnr` | Image quality (Peak Signal-to-Noise) | > 20 dB | < 15 dB |
| `val_g_ssim` | Structural similarity | > 0.5 | < 0.3 |
| `val_g_accuracy` | How often generator fools discriminator | ~50% | 0% or 100% |
| `val_d_accuracy` | Discriminator correctness | ~50% | 0% or 100% |

**What "50% discriminator accuracy" means:**
- If the discriminator is too good (accuracy > 90%), the generator can't learn
- If the discriminator is too bad (accuracy < 10%), the generator has no guidance
- 50% means they're evenly matched - ideal for learning

**Common problems and solutions:**

| Problem | Symptom | Solution |
|---------|---------|----------|
| Mode collapse | Generator produces same output for all inputs | Lower learning rates, add dropout |
| Discriminator wins | D accuracy > 90%, G loss not decreasing | Reduce discriminator_lr |
| Generator wins | D accuracy < 10%, images look blurry | Increase discriminator_lr |
| Out of memory | CUDA out of memory error | Reduce batch_size |

#### When is Training Done?

Training is complete when:
1. **val_g_psnr** stabilizes above 20 dB
2. **val_g_ssim** stabilizes above 0.5
3. Visual inspection shows good quality images

The best model checkpoint is automatically saved based on validation SSIM.

---

### 2.1.2 Segmentation (2D U-Net)

#### What is Segmentation?

Segmentation is labeling each pixel in an image. For vessel segmentation:
- **Input**: Grayscale microscopy image
- **Output**: Binary mask where white = vessel, black = background

#### Why 2D Instead of 3D?

We use a 2D model because:

| Advantage | Explanation |
|-----------|-------------|
| **Pretrained weights** | 2D models have weights trained on millions of images (ImageNet) |
| **Lower memory** | 3D models need 10-100× more GPU memory |
| **More data** | Can extract many 2D slices from each 3D volume |
| **Proven performance** | 2D + 2.5D inference works as well as 3D for vessels |

To get 3D consistency, we use **2.5D inference** (see [Section 2.2.4](#224-segmentation-inference-25d)).

#### Understanding U-Net Architecture

U-Net has two parts: an **encoder** (compresses the image) and a **decoder** (expands back to full resolution):

```
U-Net Architecture
                                                                      
  Input (256×256)                                         Output (256×256)
       │                                                       ▲
       ▼                                                       │
  ┌─────────┐                                             ┌─────────┐
  │ Encoder │ ────────── Skip Connection ─────────────────│ Decoder │
  │ Block 1 │                                             │ Block 1 │
  └────┬────┘                                             └────▲────┘
       │ (Downsample)                               (Upsample) │
       ▼                                                       │
  ┌─────────┐                                             ┌─────────┐
  │ Encoder │ ────────── Skip Connection ─────────────────│ Decoder │
  │ Block 2 │                                             │ Block 2 │
  └────┬────┘                                             └────▲────┘
       │                                                       │
       ▼                                                       │
       ... (more layers) ...                                   ...
       │                                                       │
       ▼                                                       │
  ┌─────────────────────────────────────────────────────────────┐
  │                      Bottleneck                              │
  │              (Smallest, most abstract features)              │
  └─────────────────────────────────────────────────────────────┘

Skip Connections: Pass high-resolution details directly to decoder
Bottleneck: Learns "what is a vessel" at abstract level
```

**The encoder-decoder design:**
- **Encoder**: Progressively shrinks image, learning "what" is present
- **Decoder**: Progressively expands, learning "where" things are
- **Skip connections**: Copy fine details from encoder to decoder (critical for precise boundaries)

#### Choosing an Encoder

The encoder is the "backbone" that extracts features. We use pretrained encoders from `segmentation_models_pytorch`:

| Encoder Family | Examples | Characteristics | Recommendation |
|----------------|----------|-----------------|----------------|
| **MiT (SegFormer)** | `mit_b0` to `mit_b5` | Transformer-based, best accuracy | **Recommended** |
| ResNet | `resnet18`, `resnet34`, `resnet50`, `resnet101` | Classic CNN, fast | Good baseline |
| EfficientNet | `efficientnet-b0` to `efficientnet-b7` | Efficient, good accuracy/speed | Good alternative |
| SE-ResNet | `se_resnet50`, `se_resnext50_32x4d` | ResNet with attention | Slightly better than ResNet |

**MiT-B5 (our recommendation):**
- Based on SegFormer transformer architecture
- Excellent at capturing long-range dependencies (important for connected vessels)
- `b0` to `b5` = increasing model size (b5 is largest/most accurate)

#### Choosing a Model Architecture

The "model" is the overall structure (U-Net is most common):

| Model | Description | When to Use |
|-------|-------------|-------------|
| **Unet** | Classic encoder-decoder with skip connections | Default choice, works for most cases |
| **Unet++** | Nested skip connections | Slightly better for small structures |
| **DeepLabV3+** | Atrous convolutions for multi-scale | Good for varying vessel sizes |
| **FPN** | Feature Pyramid Network | Good for multi-scale detection |
| **MAnet** | Multi-scale attention | Variable feature scales |

**For vessel segmentation, we recommend: `Unet` + `mit_b5`**

#### Loss Functions Explained

We use two losses that complement each other:

| Loss | Formula | What it Does | Why Needed |
|------|---------|--------------|------------|
| **Dice Loss** | `1 - 2×(pred∩true)/(pred+true)` | Measures overlap between prediction and ground truth | Handles class imbalance (few vessel pixels vs many background) |
| **BCE Loss** | `-[y×log(p) + (1-y)×log(1-p)]` | Per-pixel classification loss | Provides stable gradients for learning |

**Why both?** Dice alone can be unstable early in training; BCE alone struggles with class imbalance. Together they work well.

#### Input Data Format

You need paired images and masks:

```
/data/
├── images/                         # Grayscale microscopy crops
│   ├── crop_001.tif
│   ├── crop_002.tif
│   └── ...
└── masks/                          # Binary ground truth masks
    ├── crop_001.tif                # MUST have same filename as image
    ├── crop_002.tif
    └── ...
```

**Image requirements:**

| Property | Requirement | Why |
|----------|-------------|-----|
| Format | TIFF (`.tif`) | Preserves quality, no compression artifacts |
| Dimensions | 256 × 256 pixels | Balance of detail and memory usage |
| Channels | 1 (grayscale) | Microscopy is typically grayscale |
| Values | Float32, range [0, 1] | Normalized for neural network |
| Preprocessing | Contrast-adjusted | Standardizes intensity range |

**Mask requirements:**

| Property | Requirement | Why |
|----------|-------------|-----|
| Format | Same as images | Consistency |
| Dimensions | Same as corresponding image | Must align pixel-by-pixel |
| Values | Binary: 0.0 or 1.0 | Background vs vessel |
| Data type | Float32 | Matches image format |

#### Preparing Training Data

**Step 1: Extract 2D crops from your 3D volumes**

```python
import tifffile
import numpy as np
from vascumap.pipeline.utils import contrast, scale

# Load your 3D stack
stack = tifffile.imread('volume.tif')  # Shape: (Z, Y, X)
mask_3d = tifffile.imread('mask.tif')

# Extract 256×256 crops from each Z-slice
crop_size = 256
for z in range(stack.shape[0]):
    for i, (y, x) in enumerate(get_crop_positions(stack.shape[1:], crop_size)):
        # Extract image crop
        img_crop = stack[z, y:y+crop_size, x:x+crop_size]
        
        # Preprocess: contrast adjustment + normalization
        img_crop = contrast(img_crop, 1, 99)  # Clip to 1st-99th percentile
        img_crop = scale(img_crop)             # Normalize to [0, 1]
        
        # Extract corresponding mask crop
        mask_crop = mask_3d[z, y:y+crop_size, x:x+crop_size]
        mask_crop = (mask_crop > 0).astype(np.float32)  # Ensure binary
        
        # Save
        tifffile.imwrite(f'images/z{z:03d}_crop{i:03d}.tif', img_crop.astype(np.float32))
        tifffile.imwrite(f'masks/z{z:03d}_crop{i:03d}.tif', mask_crop)
```

**Step 2: Verify your data**

```python
# Check a few samples
img = tifffile.imread('images/z000_crop000.tif')
mask = tifffile.imread('masks/z000_crop000.tif')

print(f"Image shape: {img.shape}")        # Should be (256, 256)
print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")  # Should be ~[0, 1]
print(f"Mask unique values: {np.unique(mask)}")  # Should be [0, 1]
```

**Key functions:** [`contrast`](#contrast), [`scale`](#scale)

#### Handling Grayscale with Pretrained Weights

Most pretrained encoders expect 3-channel RGB images. Our microscopy is 1-channel grayscale. The code automatically adapts the first layer:

```python
# Original first layer: Conv2d(3, 64, kernel_size=7)
#   - Expects 3 input channels (R, G, B)
#   - Each channel has its own weights

# Adapted layer: Conv2d(1, 64, kernel_size=7)
#   - Accepts 1 input channel (grayscale)
#   - Weights = sum of original RGB weights

# This is done automatically by adapt_input_model()
```

**Why summing works:** The RGB weights learned complementary features. Summing them creates a reasonable grayscale feature extractor.

**Key function:** [`adapt_input_model`](#adapt_input_model)

#### Training Command

```bash
python -m vascumap.models.segmentation.training \
    --images_dir_path /path/to/images \
    --masks_dir_path /path/to/masks \
    --model_dir_path /path/to/save \
    --model_architecture Unet \
    --encoder_architecture mit_b5 \
    --input_channels 1 \
    --weights imagenet \
    --epochs 200 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --fp16
```

**All parameters explained:**

| Parameter | Default | What it Does | Recommendations |
|-----------|---------|--------------|-----------------|
| `--images_dir_path` | Required | Folder with training images | - |
| `--masks_dir_path` | Required | Folder with training masks | - |
| `--model_dir_path` | Required | Where to save checkpoints | - |
| `--model_architecture` | `Unet` | Model type (see table above) | Start with `Unet` |
| `--encoder_architecture` | `mit_b5` | Encoder backbone | `mit_b5` for best accuracy |
| `--input_channels` | `1` | Number of image channels | `1` for grayscale |
| `--weights` | `imagenet` | Pretrained weights | Always use `imagenet` |
| `--epochs` | `200` | Training iterations | 100-300 depending on dataset |
| `--batch_size` | `16` | Samples per step | Reduce if out of memory |
| `--learning_rate` | `1e-3` | Step size for optimization | Don't change unless needed |
| `--fp16` | `False` | Use half precision | Add flag to save memory |

**Key functions:** [`SegmentationModule`](#segmentationmodule), [`build_model`](#build_model), [`load_data`](#load_data)

#### Monitoring Training

The training logs show these metrics:

| Metric | What it Means | Good Values | Warning Signs |
|--------|---------------|-------------|---------------|
| `train_loss` | Combined Dice+BCE on training data | Decreasing | Stuck or increasing |
| `val_loss` | Loss on validation data | Decreasing, close to train_loss | Much higher than train_loss (overfitting) |
| `train_dice` | Overlap between prediction and truth | Increasing toward 1.0 | Stuck below 0.5 |
| `val_dice` | Dice on validation data | > 0.7 | < 0.5 |
| `train_iou` | Intersection over Union | Increasing toward 1.0 | Stuck low |
| `val_iou` | IoU on validation data | > 0.6 | < 0.4 |

**Understanding Dice and IoU:**

```
Dice = 2 × (Overlap) / (Predicted + Truth)
IoU  = Overlap / (Predicted ∪ Truth)

Example:
  Predicted:  ████░░░░     (4 pixels)
  Truth:      ░░████░░     (4 pixels)
  Overlap:    ░░██░░░░     (2 pixels)
  
  Dice = 2×2 / (4+4) = 0.5
  IoU  = 2 / (4+4-2) = 0.33
```

**Learning rate schedule:**

The learning rate is automatically reduced halfway through training:
- Epochs 1-100: LR = 0.001
- Epochs 101-200: LR = 0.0005

This helps fine-tune the model after initial learning.

**When is training done?**

1. `val_dice` > 0.7 and `val_iou` > 0.6
2. Validation loss has stopped improving for 20+ epochs
3. Visual inspection shows good segmentations

Best checkpoint is automatically saved based on lowest `val_loss`.

---

## 2.2 Inference Pipeline

### 2.2.1 Data Loading & Voxel Normalization

The pipeline reads Leica LIF files and normalizes voxel sizes:

```python
from vascumap.pipeline.data import get_stack_from_lif

stack, voxel_size, [nz, ny, nx] = get_stack_from_lif(
    lif_file='sample.lif',
    stack_index=0,
    ref_voxel_size=[5, 2, 2]  # Target: 5×2×2 µm
)
# Returns: (Z, Y, X, C) array, rescaled to reference voxel size
```

**What happens internally:**
1. Read voxel size from LIF metadata
2. Calculate rescale factors: `original_voxel / ref_voxel`
3. Resample volume using cubic interpolation
4. Validate Z-range ≥ 160 µm

**Key function:** [`get_stack_from_lif`](#get_stack_from_lif)

---

### 2.2.2 Preprocessing & Cropping

The cropping pipeline solves all three challenges through four stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CROPPING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   STAGE 1: AUTO-FOCUS                   Output: Height map + focused image  │
│   ─────────────────────                                                      │
│   Find the best Z for each XY position                                      │
│                                                                              │
│                   ↓                                                          │
│                                                                              │
│   STAGE 2: EDGE DETECTION               Output: Edge map                    │
│   ─────────────────────                                                      │
│   Find sample boundaries in the focused image                               │
│                                                                              │
│                   ↓                                                          │
│                                                                              │
│   STAGE 3: TEMPLATE REGISTRATION        Output: Transformation parameters  │
│   ────────────────────────────                                               │
│   Match a reference template to the detected edges                          │
│                                                                              │
│                   ↓                                                          │
│                                                                              │
│   STAGE 4: GEOMETRIC CROPPING           Output: Cropped stack               │
│   ─────────────────────────                                                  │
│   Extract the registered ROI in both Z and XY                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

#### Stage 1: Auto-Focus (Finding the Curved Surface)

**The Problem:**
The sample surface is curved. If we just take a single Z-slice, some areas would be in focus and others blurry. We need to find the "surface" of the sample.

**The Solution:**
Create a **height map** - for each (X, Y) position, store which Z-slice is in focus.

**Algorithm Walkthrough:**

```
Step 1: Sample sparse grid points
─────────────────────────────────
We don't check every pixel (too slow). Instead, we sample a grid:

Full image (1000×1000 pixels):
┌────────────────────────────────────┐
│  ·     ·     ·     ·     ·     ·   │
│                                     │
│  ·     ·     ·     ·     ·     ·   │    · = sampling point
│                                     │        (n_sampling × n_sampling)
│  ·     ·     ·     ·     ·     ·   │
│                                     │
│  ·     ·     ·     ·     ·     ·   │
└────────────────────────────────────┘

Step 2: At each point, find the best Z
────────────────────────────────────────
For each sampling point, we look at a patch around it across all Z-slices:

Z=0:  ░░░░░    Z=1:  ░░▒░░    Z=2:  ░▒▓▒░    Z=3:  ░░▒░░    Z=4:  ░░░░░
      ░░░░░          ░░▒░░          ▒▓█▓▒          ░░▒░░          ░░░░░
      ░░░░░          ░░▒░░          ░▒▓▒░          ░░▒░░          ░░░░░
      Blurry         Sharper        SHARPEST       Sharper        Blurry

Focus measure = std(sobel(patch))
  - Sobel detects edges
  - Sharp images have more edges → higher std
  - Select Z with maximum focus measure

Step 3: Fit a plane to the sparse points
──────────────────────────────────────────
We have (x, y, z_best) for each sampling point. We fit a plane:

    Z = a·X + b·Y + c

This smooths out noise and gives us a continuous surface.

Step 4: Create dense height map
────────────────────────────────
Apply the plane equation to every pixel to get the full height map.
```

**Code:**

```python
from vascumap.pipeline.preprocessing import auto_focus

focused_image, height_map = auto_focus(
    stack_bf,           # 3D brightfield stack: (Z, Y, X)
    crop_window=50,     # Pixels to exclude from border
    n_sampling=20,      # Grid will be 20×20 = 400 points
    window_size=50      # Each patch is 100×100 pixels (±50)
)

# focused_image: 2D image showing the in-focus surface
# height_map: 2D array of Z-indices (same shape as focused_image)
```

**Parameters:**

| Parameter | What it Does | Typical Value | Notes |
|-----------|--------------|---------------|-------|
| `crop_window` | Excludes image borders | 50-200 | Borders often have artifacts |
| `n_sampling` | Grid density | 5-20 | More = slower but more accurate |
| `window_size` | Patch size for focus | 50-100 | Larger = more robust to noise |

**Key function:** [`auto_focus`](#auto_focus)

---

#### Stage 2: Edge Detection (Finding Sample Boundaries)

**The Problem:**
We need to find where the sample is in the image so we can align our template to it.

**The Solution:**
Use Sobel edge detection to highlight the sample boundaries.

**How Sobel Works:**

```
Sobel detects intensity changes (edges) in an image:

Original:                    After Sobel:
┌────────────────────┐       ┌────────────────────┐
│████████░░░░░░░░░░░░│       │░░░░░░░█░░░░░░░░░░░░│  ← Edge detected
│████████░░░░░░░░░░░░│       │░░░░░░░█░░░░░░░░░░░░│    at boundary
│████████░░░░░░░░░░░░│  →    │░░░░░░░█░░░░░░░░░░░░│
│████████░░░░░░░░░░░░│       │░░░░░░░█░░░░░░░░░░░░│
│░░░░░░░░░░░░░░░░░░░░│       │░░░░░░░░░░░░░░░░░░░░│
└────────────────────┘       └────────────────────┘
```

**Processing Steps:**

```python
# 1. Downsample for speed (10% of original size)
im_r = rescale(focused_image, 0.1)

# 2. Compute Sobel gradients in both directions
edges_x = sobel_h(im_r)  # Horizontal edges
edges_y = sobel_v(im_r)  # Vertical edges

# 3. Combine into gradient magnitude
edges = sqrt(edges_x² + edges_y²)

# 4. Smooth to reduce noise
edges = gaussian(edges, sigma=2.5)

# 5. Threshold high values (removes noise)
edges = clip(edges, 0, 0.15)

# 6. Resize back to original
edges = resize(edges, original_shape)
```

---

#### Stage 3: Template Registration (Aligning to a Reference)

**The Problem:**
We have edges showing the sample boundary. We need to figure out exactly where to crop. This is solved by aligning a **pre-defined template** to the detected edges.

**What is a Template?**

A template consists of two files:

```
crop_profiles/
├── {profile}_fiducials.npy    # Points that should align with edges
└── {profile}_rect.npy         # The ROI we want to extract
```

**Fiducials:** Points along the sample boundary that we expect to see in every image.
**Rectangle:** The four corners of the region we want to extract.


**The Registration Process:**

We find the transformation (scale, rotation, translation) that best aligns the template fiducials with the detected edges.

```
Transformation Parameters:
  - scale: How much to resize the template (usually ~1.0)
  - theta: Rotation angle in radians
  - tx: Translation in X
  - ty: Translation in Y

Mathematical formulation:
  
  transformed_point = scale × (original_point × rotation_matrix) + translation
  
  where rotation_matrix = [cos(θ)  -sin(θ)]
                          [sin(θ)   cos(θ)]
```

**Optimization Strategy:**

Finding the best transformation is an optimization problem. We use a two-stage approach:

```
Stage 1: BASIN-HOPPING (Global Search)
────────────────────────────────────────
Problem: The objective function has many local minima.
Solution: Try many random starting points, do local optimization from each.


Stage 2: POWELL (Local Refinement)
──────────────────────────────────
Take the best result from basin-hopping and refine it further.
Powell method: Optimizes one parameter at a time, then repeats.
```

**The Objective Function:**

```python
def objective_function(params, template_coords, edge_image):
    """
    Score how well the transformed template matches the edges.
    
    Lower score = better match (we minimize this).
    """
    # Transform template points using current params
    transformed = transform_coords(template_coords, params)
    
    # Check how many points are inside the image
    valid_mask = points_inside_image(transformed)
    valid_ratio = sum(valid_mask) / len(template_coords)
    
    # Penalize heavily if too many points are outside
    if valid_ratio < 0.9:
        return large_penalty  # Bad transformation
    
    # Sample edge values at transformed points
    edge_responses = edge_image[transformed_y, transformed_x]
    
    # Good alignment = high edge values = low cost
    return -mean(edge_responses)
```

**Key functions:** [`register_high_level`](#register_high_level), [`transform_coords`](#transform_coords), [`objective_function`](#objective_function), [`register_neck_coords_advanced`](#register_neck_coords_advanced)

---

#### Stage 4: Geometric Cropping (Extracting the ROI)

**The Problem:**
We now know where the ROI is (transformed rectangle coordinates). We need to extract it from the 3D stack.

**Two Sub-Problems:**

1. **Z-Cropping:** Extract a slab of the right thickness, following the curved surface
2. **XY-Cropping:** Extract the (possibly rotated) quadrilateral and warp it to a rectangle

**Z-Cropping - Following the Surface:**

The height map from Stage 1 tells us where the surface is. We extract a slab centered on this surface:


```python
# For each (y, x), we extract Z slices from:
#   grid[y, x] - z_range/2  to  grid[y, x] + z_range/2
# 
# This creates a "slab" that follows the curved surface
```

**XY-Cropping - Warping the Quadrilateral:**

The ROI is defined by 4 corners which may form a non-axis-aligned quadrilateral.


```python
from vascumap.pipeline.preprocessing import crop_stack

cropped = crop_stack(
    stack,                      # Input: (Z, Y, X, C)
    crop_profile='dorota',      # Template name
    bf_index=0,                 # Which channel to use for registration
    mode='3D',                  # '3D' extracts a slab; '2D' extracts single slice
    z_range=200,                # Thickness in µm
    ref_voxel_size=[5, 2, 2],   # Voxel size for Z-range conversion
    lif_path_root='/output',    # Save debug images here
    filename='sample1'          # Prefix for debug images
)
```

**Key functions:** [`crop_stack`](#crop_stack), [`crop_stack_z`](#crop_stack_z), [`apply_affine_crop_simple`](#apply_affine_crop_simple)

---

#### Creating Custom Crop Profiles

To use the cropping pipeline with your own sample type, you need to create a template.

**Step 1: Get a Representative Image**

```python
from vascumap.pipeline.preprocessing import auto_focus

# Load a typical sample
stack = ...  # Your 3D stack

# Get a focused 2D image
focused_image, _ = auto_focus(stack[..., 0])  # Use brightfield channel
```

**Step 2: Annotate Fiducial Points**

Using a tool like Napari, FIJI, or matplotlib, manually click points along the sample boundary:

```python
import matplotlib.pyplot as plt
import numpy as np

# Display the image
fig, ax = plt.subplots()
ax.imshow(focused_image, cmap='gray')

# Collect clicks
fiducials = []
def onclick(event):
    fiducials.append([event.ydata, event.xdata])  # Note: (y, x) order
    ax.plot(event.xdata, event.ydata, 'r.', markersize=5)
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# After clicking, convert to array
fiducials = np.array(fiducials)  # Shape: (N, 2) with columns [y, x]
```

**Tips for good fiducials:**
- Place points along strong edges (sample boundary)
- Space them evenly around the boundary
- Use 50-200 points for good coverage
- Avoid points on features that vary between samples

**Step 3: Define the ROI Rectangle**

Define the 4 corners of your region of interest (clockwise from top-left):

```python
# Corners in (y, x) format
rect = np.array([
    [y_top_left,     x_top_left],
    [y_top_right,    x_top_right],
    [y_bottom_right, x_bottom_right],
    [y_bottom_left,  x_bottom_left],
])
```

**Step 4: Save the Profile**

```python
np.save('crop_profiles/myprofile_fiducials.npy', fiducials)
np.save('crop_profiles/myprofile_rect.npy', rect)
```

**Step 5: Test the Profile**

```python
from vascumap.pipeline.preprocessing import crop_stack

# Test on a few samples
cropped = crop_stack(
    test_stack,
    crop_profile='myprofile',
    bf_index=0,
    mode='3D',
    z_range=200,
    lif_path_root='./debug',  # Check the debug images!
    filename='test'
)
```

Check the saved `*_crop_overview.png` to see if registration worked correctly.

**Common Issues and Solutions:**

| Problem | Symptom | Solution |
|---------|---------|----------|
| Template too large | Points go outside image | Scale down template coordinates |
| Not enough fiducials | Poor alignment | Add more fiducial points |
| Fiducials on variable features | Inconsistent alignment | Move fiducials to stable boundaries |
| Optimization stuck | Bad crop location | Try different initial parameters |

---

### 2.2.3 Image Translation Inference

Run the Pix2Pix model to translate brightfield to fluorescence-like:

```python
from vascumap.pipeline.prediction import load_pix2pix, predict_pix2pix

model = load_pix2pix('translation.ckpt')

pseudo_fluo = predict_pix2pix(
    model,
    stack_bf,           # (Z, Y, X)
    device='cuda',
    n_iter=1            # Monte Carlo averaging (for dropout uncertainty)
)
```

**Inference details:**
- Uses MONAI SlidingWindowInferer
- ROI size: 32×512×512, overlap: 75%/25%/25%
- Output clipped to [0, 1]

**Key functions:** [`load_pix2pix`](#load_pix2pix), [`predict_pix2pix`](#predict_pix2pix)

---

### 2.2.4 Segmentation Inference (2.5D)

Run the segmentation model using **2.5D inference** - predictions along three orthogonal planes, averaged:

```python
from vascumap.pipeline.prediction import load_segmentation_model, predict_mask_ortho

model = load_segmentation_model('segmentation.ckpt')

probability_map = predict_mask_ortho(
    model,
    vessel_iso,         # Isotropic volume (Z, Y, X)
    device='cuda'
)
```

**2.5D approach:**
1. **Axial** (XY slices): Primary view, full resolution
2. **Coronal** (XZ slices): Side view, 256×256 patches
3. **Sagittal** (YZ slices): Front view, 256×256 patches
4. **Average**: Final probability = mean of all three

**Post-processing:**
```python
from vascumap.pipeline.utils import process_vessel_mask

binary_mask = process_vessel_mask(probability_map, ortho=True)
# Applies: median filter (size=7) + hysteresis threshold (0.2, 0.5)
```

**Key functions:** [`load_segmentation_model`](#load_segmentation_model), [`predict_mask_ortho`](#predict_mask_ortho), [`predict_mask`](#predict_mask), [`process_vessel_mask`](#process_vessel_mask)

---

### Full Pipeline Command

```bash
python -m vascumap.pipeline.main \
    --lif_dir_path /path/to/lif/files \
    --model_p2p_path /path/to/translation.ckpt \
    --model_smp_path /path/to/segmentation.ckpt
```

**Output files (per sample):**
- `*_vessel_mask.tif`: Binary segmentation
- `*_overview.png`: Visual comparison
- `*_crop_overview.png`: Registration debug (if cropping)

**Key function:** [`run`](#run)

---

## 2.3 Graph Analysis

### 2.3.1 Skeletonization

Convert binary mask to 1-pixel-wide skeleton. Two methods available:

#### Method 1: Morphological Thinning

Fast, iterative erosion preserving topology:

```python
from vascumap.pipeline.skeletonization import skeletonize_3d_parallel

skeleton = skeletonize_3d_parallel(
    binary_mask,
    chunk_size=(128, 128, 128)
)
```

**Best for:** Dense networks (vessel coverage > 30%)

**Key function:** [`skeletonize_3d_parallel`](#skeletonize_3d_parallel)

#### Method 2: Mesh Contraction

GPU-accelerated Laplacian smoothing on mesh:

```python
from vascumap.pipeline.mesh_contraction import create_graph_contraction

graph, mesh, simple_mesh, contracted = create_graph_contraction(
    binary_edt,
    time_lim=300  # seconds
)
```

**Algorithm:**
1. Generate surface mesh (Marching Cubes)
2. Iteratively contract toward centerline
3. Extract skeleton from contracted mesh

**Best for:** Sparse networks, varying vessel diameters

**Key functions:** [`create_graph_contraction`](#create_graph_contraction), [`contract_gpu`](#contract_gpu)

---

### 2.3.2 Graph Construction

Convert skeleton to NetworkX graph:

```python
import sknw

# Build graph from skeleton
graph = sknw.build_sknw(skeleton_image)

# Nodes have 'pts' attribute (coordinates)
# Edges have 'pts' attribute (path coordinates)
```

**Post-processing:**
1. **Pruning**: Remove spurious branches based on length/EDT ratio
2. **Merge degree-2 nodes**: Simplify graph
3. **Border trimming**: Remove edges near image boundaries

```python
from vascumap.pipeline.skeletonization import prune_graph, finalize_graph

pruned = prune_graph(graph, area_3d, edt_cutoff=0.25, length_cutoff=25)
final = finalize_graph(pruned, binary_edt)
```

**Key functions:** [`prune_graph`](#prune_graph), [`finalize_graph`](#finalize_graph), [`collect_border_vicinity_edges`](#collect_border_vicinity_edges)

---

### 2.3.3 Metrics Computation

Compute morphometric features from the graph:

```python
from vascumap.pipeline.metrics import (
    compute_cross_sectional_areas,
    compute_vessel_metrics,
    compute_junction_metrics,
    fractal_dimension_and_lacunarity,
    convex_hull_volume
)

# Cross-sectional areas at skeleton points
area_3d = compute_cross_sectional_areas(mask, skeleton, edt)

# Per-vessel metrics
vessel_df = compute_vessel_metrics(graph, area_3d, pd.DataFrame())

# Per-junction metrics
junction_df = compute_junction_metrics(graph, pd.DataFrame(), distance_threshold=250)

# Global metrics
fd, lacunarity = fractal_dimension_and_lacunarity(skeleton > 0)
explant_vol = convex_hull_volume(mask)
```

**Key functions:** [`compute_cross_sectional_areas`](#compute_cross_sectional_areas), [`compute_vessel_metrics`](#compute_vessel_metrics), [`compute_junction_metrics`](#compute_junction_metrics), [`fractal_dimension_and_lacunarity`](#fractal_dimension_and_lacunarity), [`convex_hull_volume`](#convex_hull_volume)

---

### 2.3.4 Output Files

Graph analysis produces three output files:

| File | Format | Content |
|------|--------|---------|
| `global_metrics_{method}.csv` | CSV | One row per sample, aggregate statistics |
| `{sample}_vessel_metrics_{method}.h5` | HDF5 | One row per vessel segment |
| `{sample}_junction_metrics_{method}.h5` | HDF5 | One row per node |

#### Reading Output Files

```python
import pandas as pd

# Global metrics
global_df = pd.read_csv('global_metrics_thinning.csv')

# Vessel metrics (requires: pip install tables)
vessel_df = pd.read_hdf('sample_vessel_metrics_thinning.h5', key='df')

# Junction metrics
junction_df = pd.read_hdf('sample_junction_metrics_thinning.h5', key='df')
```

#### Full Pipeline Command

```bash
python -m vascumap.pipeline.graph_main \
    --masks_dir_path /path/to/masks \
    --file_suffix _vessel_mask \
    --method thinning \
    --save_local_metrics \
    --graph_visualization
```

**Key function:** [`process_directory`](#process_directory)

---

# 3. Function Reference

This section documents all public functions organized by module. Each entry includes:
- **Purpose**: What the function does
- **Arguments**: Input parameters with types
- **Returns**: Output values
- **Source**: Link to source file

---

## 3.1 Image Translation Module

**Location:** [`vascumap/models/image_translation/`](vascumap/models/image_translation/)

### `ImageDataset3D`

PyTorch Dataset for paired 3D NIfTI volumes.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
dataset = ImageDataset3D(
    input_paths: List[str],      # Paths to input volumes
    target_paths: List[str],     # Paths to target volumes (optional)
    split: str = 'train',        # 'train', 'val', or 'test'
    transform: callable = None   # Custom TorchIO transform
)
```

**Returns:** `tuple(input_tensor, target_tensor)` or `input_tensor`

---

### `Generator`

3D U-Net generator based on MONAI.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
generator = Generator(
    dropout_p: float = 0.4    # Dropout probability
)
```

**Architecture:**
- Channels: (32, 64, 128, 256, 512)
- Strides: (1, 2, 2, 2, 1)
- Residual units: 3 per level

---

### `Discriminator`

3D PatchGAN discriminator.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
discriminator = Discriminator(
    dropout_p: float = 0.4    # Dropout probability
)
```

**Input:** Concatenated input+target (2 channels)

---

### `Pix2Pix`

PyTorch Lightning module for GAN training.

**Source:** [`utils.py`](vascumap/models/image_translation/utils.py)

```python
model = Pix2Pix(
    generator_dropout_p: float = 0.4,
    discriminator_dropout_p: float = 0.4,
    generator_lr: float = 1e-3,
    discriminator_lr: float = 1e-6,
    weight_decay: float = 1e-5,
    lr_scheduler_T_0: float = 5e3,
    lr_scheduler_T_mult: float = 2
)
```

**Loss functions:**
- Generator: `BCE + λ×L1 + λ×MSE` (λ=100)
- Discriminator: `BCE`

---

## 3.2 Segmentation Module

**Location:** [`vascumap/models/segmentation/`](vascumap/models/segmentation/)

### `SegmentationModule`

PyTorch Lightning module for 2D segmentation.

**Source:** [`model.py`](vascumap/models/segmentation/model.py)

```python
model = SegmentationModule(
    model_name: str,             # 'Unet', 'FPN', 'DeepLabV3+', etc.
    encoder_name: str,           # 'mit_b5', 'resnet50', etc.
    in_channels: int,            # Usually 1 for grayscale
    encoder_weights: str = 'imagenet',
    learning_rate: float = 1e-3,
    max_epochs: int = 200        # For scheduler calculation
)
```

**Training config:**
- Loss: Dice + BCE
- Optimizer: RAdam (weight_decay=1e-4)
- Scheduler: StepLR (halves at max_epochs/2)

---

### `build_model`

Factory function for segmentation models.

**Source:** [`model_utils.py`](vascumap/models/segmentation/model_utils.py)

```python
model = build_model(
    model_str: str,              # Architecture name
    encoder_str: str,            # Encoder backbone
    in_channels: int = 1,
    encoder_weights: str = None  # 'imagenet' or None
)
```

**Supported architectures:**
`Unet`, `Unet++`, `MAnet`, `Linknet`, `FPN`, `PSPNet`, `PAN`, `DeepLabV3`, `DeepLabV3+`

**Returns:** `torch.nn.Module`

---

### `adapt_input_model`

Adapts pretrained RGB weights for grayscale input.

**Source:** [`model_utils.py`](vascumap/models/segmentation/model_utils.py)

```python
model = adapt_input_model(
    model: nn.Module    # Model with 3-channel first layer
)
```

**Returns:** `nn.Module` with 1-channel first layer

---

### `load_data`

Create train/val data loaders.

**Source:** [`dataset.py`](vascumap/models/segmentation/dataset.py)

```python
loaders = load_data(
    images_path_str: str,        # Directory of images
    masks_path_str: str = None,  # Directory of masks (None for inference)
    batch_size: int = 16,
    seed: int = 0,
    test_size: float = 0.1,
    format: str = 'tif'
)
```

**Returns:** `OrderedDict` with 'train' and 'valid' DataLoaders

---

### `SegmentationDataset`

PyTorch Dataset for 2D image-mask pairs.

**Source:** [`dataset.py`](vascumap/models/segmentation/dataset.py)

```python
dataset = SegmentationDataset(
    images: List[Path],          # Image file paths
    masks: List[Path] = None,    # Mask file paths
    transforms: callable = None  # Albumentations composition
)
```

**Returns:** `dict` with 'image', 'mask', 'filename'

---

## 3.3 Pipeline Module

**Location:** [`vascumap/pipeline/`](vascumap/pipeline/)

### Data Loading

#### `get_stack_from_lif`

Extract and rescale 3D stack from LIF file.

**Source:** [`data.py`](vascumap/pipeline/data.py)

```python
stack, voxel_size, [nz, ny, nx] = get_stack_from_lif(
    lif_file: str,                       # Path to LIF file
    stack_index: int = 0,                # Image index in LIF
    ref_voxel_size: list = [5, 2, 2]     # Target voxel size [Z, Y, X] µm
)
```

**Returns:**
- `stack`: `np.ndarray` (Z, Y, X, C)
- `voxel_size`: `np.ndarray` [Z, Y, X] original
- `[nz, ny, nx]`: Original dimensions

---

### Preprocessing

#### `auto_focus`

Find optimal focal plane across XY.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
focus, grid = auto_focus(
    stack_bf: np.ndarray,        # 3D stack (Z, Y, X)
    crop_window: int = 50,       # Border margin
    n_sampling: int = 20,        # Grid density
    window_size: int = 50,       # Focus patch size
    viz: bool = False            # Show visualization
)
```

**Returns:**
- `focus`: 2D focused image (Y, X)
- `grid`: Height map of Z-indices (Y, X)

---

#### `crop_stack`

Complete cropping pipeline.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
cropped = crop_stack(
    stack: np.ndarray,           # Input (Z, Y, X, C)
    crop_profile: str,           # Profile name
    bf_index: int,               # Brightfield channel
    mode: str = '3D',            # '2D' or '3D'
    z_range: float = 200,        # Z-thickness in µm
    ref_voxel_size: list = [5, 2, 2],
    lif_path_root: str = None,   # Save debug images here
    filename: str = None
)
```

**Returns:** `np.ndarray` cropped stack

---

#### `crop_stack_z`

Crop Z-range around focal surface.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
cropped = crop_stack_z(
    stack: np.ndarray,           # Full stack
    grid: np.ndarray,            # Height map from auto_focus
    z_range: float,              # Thickness in µm
    ref_voxel_size: tuple        # Voxel size [Z, Y, X]
)
```

**Returns:** `np.ndarray` Z-cropped stack

---

#### `apply_affine_crop_simple`

Warp quadrilateral ROI to rectangle.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
warped = apply_affine_crop_simple(
    image: np.ndarray,           # 3D (Z,Y,X,C) or 2D (Y,X,C)
    registered_rect: np.ndarray, # 4×2 corners (y, x)
    order: int = 1               # Interpolation order
)
```

**Returns:** `np.ndarray` warped image

---

#### `transform_coords`

Apply affine transformation to coordinates.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
transformed = transform_coords(
    coords: np.ndarray,          # (N, 2) coordinates
    params: list                 # [scale, theta, tx, ty]
)
```

**Returns:** `np.ndarray` transformed coordinates

---

#### `register_high_level`

High-level registration wrapper.

**Source:** [`preprocessing.py`](vascumap/pipeline/preprocessing.py)

```python
registered_coords, optimal_params, best_score = register_high_level(
    edges: np.ndarray,           # Edge map
    neck_coords_all: np.ndarray  # Template coordinates
)
```

**Returns:** `tuple` (transformed_coords, params, score)

---

### Prediction

#### `load_pix2pix`

Load trained Pix2Pix model.

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
model = load_pix2pix(
    model_path: str,             # Checkpoint path
    device: str = 'cuda'
)
```

**Returns:** `Pix2Pix` in eval mode

---

#### `load_segmentation_model`

Load trained segmentation model.

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
model = load_segmentation_model(
    model_path: str,             # Checkpoint path (.pth or .ckpt)
    device: str = 'cuda'
)
```

**Returns:** `nn.Module` in eval mode

---

#### `predict_pix2pix`

Run translation inference with sliding window.

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
prediction = predict_pix2pix(
    model_p2p: nn.Module,
    stack_bf: np.ndarray,        # (Z, Y, X) brightfield
    device: str,
    n_iter: int = 1              # Monte Carlo averaging
)
```

**Returns:** `np.ndarray` pseudo-fluorescence (Z, Y, X)

---

#### `predict_mask`

Run 2D segmentation (axial slices only).

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
proba = predict_mask(
    model_smp: nn.Module,
    vessel_pred: np.ndarray,     # (Z, Y, X)
    device: str
)
```

**Returns:** `np.ndarray` probability map (Z, Y, X)

---

#### `predict_mask_ortho`

Run 2.5D segmentation (3-plane averaging).

**Source:** [`prediction.py`](vascumap/pipeline/prediction.py)

```python
proba = predict_mask_ortho(
    model_smp: nn.Module,
    vessel_pred_iso: np.ndarray, # Isotropic volume
    device: str
)
```

**Returns:** `np.ndarray` averaged probability map

---

### Main Pipeline

#### `run`

Run full inference pipeline.

**Source:** [`main.py`](vascumap/pipeline/main.py)

```python
run(
    lif_dir_path: str,           # Directory with .lif files
    ortho: bool = True,          # Use 2.5D inference
    bf_index: int = None,        # Brightfield channel (auto-detect if None)
    fluo_index: int = None,      # Fluorescence channel (skip translation if set)
    crop_profile: str = 'dorota',
    z_range: int = 200,
    save_all_stacks: bool = False,
    crop: bool = True,
    model_p2p_path: str = None,
    model_smp_path: str = None
)
```

---

## 3.4 Graph Analysis Module

### Skeletonization

#### `skeletonize_3d_parallel`

Parallel thinning using Dask.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
skeleton = skeletonize_3d_parallel(
    binary_volume: np.ndarray,
    chunk_size: tuple = (128, 128, 128),
    iter: int = 1
)
```

**Returns:** `np.ndarray` skeleton image

---

#### `prune_graph`

Remove spurious branches.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
pruned = prune_graph(
    graph: nx.Graph,
    area_3d: np.ndarray,         # Cross-sectional areas
    edt_cutoff: float = 0.25,    # EDT ratio threshold
    length_cutoff: int = 25      # Minimum branch length
)
```

**Returns:** `nx.Graph` pruned

---

#### `finalize_graph`

Clean graph: merge degree-2 nodes, clip coordinates.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
final = finalize_graph(
    G_repositioned: nx.Graph,
    binary_edt: np.ndarray
)
```

**Returns:** `nx.Graph` cleaned

---

#### `collect_border_vicinity_edges`

Remove edges near image borders.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
trimmed = collect_border_vicinity_edges(
    graph: nx.Graph,
    image_shape: tuple,
    vicinity_z: int = 1,
    vicinity_xy: int = 50
)
```

**Returns:** `nx.Graph` with border edges removed

---

#### `measure_edge_length`

Compute path length of edge coordinates.

**Source:** [`skeletonization.py`](vascumap/pipeline/skeletonization.py)

```python
length = measure_edge_length(
    coordinates: np.ndarray      # (N, 3) points
)
```

**Returns:** `float` total path length

---

### Mesh Contraction

#### `create_graph_contraction`

Full mesh contraction pipeline.

**Source:** [`mesh_contraction.py`](vascumap/pipeline/mesh_contraction.py)

```python
graph, mesh, simple_mesh, contracted = create_graph_contraction(
    binary_edt: np.ndarray,
    time_lim: int = 300          # Seconds
)
```

**Returns:** `tuple` (graph, original_mesh, simplified_mesh, contracted_mesh)

---

#### `contract_gpu`

GPU-accelerated Laplacian mesh contraction.

**Source:** [`mesh_contraction.py`](vascumap/pipeline/mesh_contraction.py)

```python
contracted = contract_gpu(
    mesh: trimesh.Trimesh,
    epsilon: float = 1e-6,       # Convergence threshold
    iter_lim: int = 100,
    time_lim: int = None,        # Seconds
    SL: float = 2,               # Contraction weight multiplier
    WH0: float = 1,              # Initial attraction weight
    operator: str = 'cotangent' # 'cotangent' or 'umbrella'
)
```

**Returns:** `trimesh.Trimesh` contracted

---

#### `reposition_graph_edges`

Move nodes toward vessel centerline using EDT gradient.

**Source:** [`mesh_contraction.py`](vascumap/pipeline/mesh_contraction.py)

```python
repositioned = reposition_graph_edges(
    graph: nx.Graph,
    binary_edt: np.ndarray,
    min_segment_length: float = 5.0,
    max_disp: float = 12.5,
    step_size: float = 1.0,
    num_iterations: int = 1000
)
```

**Returns:** `nx.Graph` repositioned

---

### Metrics

#### `compute_cross_sectional_areas`

Compute vessel area at skeleton points.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
area_3d = compute_cross_sectional_areas(
    mask: np.ndarray,            # Binary mask
    skeleton_image: np.ndarray,  # Skeleton
    binary_edt: np.ndarray       # EDT
)
```

**Formula:** `Area = π × EDT_2D(y,x) × EDT_3D(z,y,x)`

**Returns:** `np.ndarray` with areas at skeleton points

---

#### `compute_vessel_metrics`

Compute per-edge metrics.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
vessel_df = compute_vessel_metrics(
    graph: nx.Graph,
    area_image: np.ndarray,
    vessel_metrics_df: pd.DataFrame
)
```

**Returns:** `pd.DataFrame` with columns: `z, y, x, volume, length, shortest_path, tortuosity, is_sprout, mean_cs_area, median_cs_area, std_cs_area, node1_degree, node2_degree, orientation_z, orientation_y, orientation_x`

---

#### `compute_junction_metrics`

Compute per-node metrics.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
junction_df = compute_junction_metrics(
    graph: nx.Graph,
    junction_metrics_df: pd.DataFrame,
    distance_threshold: int = 50
)
```

**Returns:** `pd.DataFrame` with columns: `z, y, x, number_of_vessel_per_node, is_sprout_tip, is_junction, dist_nearest_junction, dist_nearest_endpoint, num_junction_neighbors, num_endpoint_neighbors`

---

#### `fractal_dimension_and_lacunarity`

Box-counting fractal analysis.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
fd, lacunarity = fractal_dimension_and_lacunarity(
    array: np.ndarray,           # Binary
    max_box_size: int = None,    # Auto from image
    min_box_size: int = 1,
    n_samples: int = 20
)
```

**Returns:** `tuple` (fractal_dimension, lacunarity)

---

#### `convex_hull_volume`

Compute convex hull volume.

**Source:** [`metrics.py`](vascumap/pipeline/metrics.py)

```python
volume = convex_hull_volume(
    binary_image: np.ndarray
)
```

**Returns:** `float` hull volume in voxels³

---

### Main Graph Pipeline

#### `process_directory`

Process all masks in a directory.

**Source:** [`graph_main.py`](vascumap/pipeline/graph_main.py)

```python
process_directory(
    masks_dir_path: str,
    suffix: str,                 # e.g., '_vessel_mask'
    save_local_metrics: bool,
    visualization: bool = True,
    skel_method: str = "thinning",
    contraction_timelim: int = 300
)
```

---

#### `construct_vessel_network`

Build graph from binary mask.

**Source:** [`graph_main.py`](vascumap/pipeline/graph_main.py)

```python
graph, area_image = construct_vessel_network(
    binary_image: np.ndarray,
    global_metrics_df: pd.DataFrame,
    skel_method: str = "auto",
    contraction_timelim: int = 300
)
```

**Returns:** `tuple` (NetworkX graph, area volume)

---

#### `calculate_graph_metrics`

Compute all metrics from graph.

**Source:** [`graph_main.py`](vascumap/pipeline/graph_main.py)

```python
global_df, junction_df, vessel_df = calculate_graph_metrics(
    graph: nx.Graph,
    sample_name: str,
    area_image: np.ndarray,
    global_metrics_df: pd.DataFrame
)
```

**Returns:** `tuple` of DataFrames

---

## 3.5 Utilities

**Location:** [`vascumap/pipeline/utils.py`](vascumap/pipeline/utils.py)

### `contrast`

Percentile contrast adjustment.

```python
adjusted = contrast(
    arr: np.ndarray,
    low: float,                  # Lower percentile (0-100)
    high: float                  # Upper percentile (0-100)
)
```

**Returns:** `np.ndarray` clipped to percentile range

---

### `scale`

Min-max normalization to [0, 1].

```python
scaled = scale(
    arr: np.ndarray
)
```

**Returns:** `np.ndarray` in [0, 1]

---

### `resize_dask`

Dask-based 3D rescaling.

```python
resized = resize_dask(
    stack: np.ndarray,
    rescale_factor: list         # [z, y, x] factors
)
```

**Returns:** `np.ndarray` resized

---

### `cupy_chunk_processing`

Apply GPU function to volume in chunks.

```python
result = cupy_chunk_processing(
    volume: np.ndarray,
    processing_func: callable,
    chunk_size: tuple = (64, 512, 512),
    overlap: tuple = (15, 15, 15),
    **kwargs                     # Passed to processing_func
)
```

**Returns:** `np.ndarray` processed

---

### `median_filter_3d_gpu`

GPU-accelerated 3D median filter.

```python
filtered = median_filter_3d_gpu(
    volume: np.ndarray,
    size: int = 3,
    chunk_size: tuple = (64, 64, 64)
)
```

**Returns:** `np.ndarray` filtered

---

### `process_vessel_mask`

Convert probability map to binary mask.

```python
mask = process_vessel_mask(
    vessel_proba: np.ndarray,    # Probabilities [0, 1]
    ortho: bool = False          # Apply extra filtering
)
```

**Processing:**
- If `ortho=True`: median filter (size=7) + threshold (0.2, 0.5)
- If `ortho=False`: threshold (0.1, 0.5)

**Returns:** `np.ndarray` binary mask

---

### `remove_false_positives`

Remove artifacts using Gaussian fitting.

```python
remove_false_positives(
    mask_p: str                  # Path to mask file
)
```

Modifies file in-place, saves backup as `*_backup.tif`.

---

# 4. Metrics Reference

## 4.1 Global Metrics (CSV)

Stored in `global_metrics_{method}.csv`, one row per sample.

### Volume & Coverage

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `sample` | - | Sample identifier | - |
| `explant_volume` | voxels³ | Convex hull volume | `ConvexHull(foreground).volume` |
| `image_volume` | voxels³ | Total image volume | `Z × Y × X` |
| `total_vessel_volume` | voxels³ | Foreground voxels | `sum(mask)` |
| `vessel_coverage_explant` | ratio | Vessel/explant | `vessel_volume / explant_volume` |
| `vessel_coverage_image` | ratio | Vessel/image | `vessel_volume / image_volume` |
| `skeletonization_method` | - | 'thinning' or 'contraction' | - |

### Network Topology

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `total_vessel_length` | voxels | Sum of edge lengths | `Σ length(edge)` |
| `total_number_of_sprouts` | count | Edges with degree-1 endpoint | Vessels connected to tips |
| `total_number_of_branches` | count | Edges between junctions | Both endpoints degree ≥ 2 |
| `total_number_of_junctions` | count | Nodes with degree ≥ 2 | Branch points |

### Fractal Analysis

| Column | Unit | Description | Formula |
|--------|------|-------------|---------|
| `fractal_dimension` | - | Box-counting dimension | `slope(log(N) vs log(1/ε))` |
| `lacunarity` | - | Heterogeneity measure | `mean(Var(H)/Mean(H)²)` |

**Interpretation:**
- Fractal dimension ≈ 1: Linear structure
- Fractal dimension ≈ 2: Area-filling
- Fractal dimension ≈ 3: Volume-filling
- Typical vessels: 1.3 - 2.5

### Aggregated Statistics

Pattern: `{statistic}_{type}_{metric}`

**Statistics:** `mean`, `std`, `median`  
**Types:** `branch` (both ends junction), `sprout` (one end tip), `sprout_and_branch` (all)  
**Vessel metrics:** `volume`, `length`, `shortest_path`, `tortuosity`

**Junction types:** `junction` (degree ≥ 2), `sprout_tip` (degree = 1), `junction_and_sprout_tip` (all)  
**Junction metrics:** `number_of_vessel_per_node`, `dist_nearest_junction`, `dist_nearest_endpoint`, `num_junction_neighbors`, `num_endpoint_neighbors`

---

## 4.2 Vessel Metrics (HDF5)

Stored in `{sample}_vessel_metrics_{method}.h5`, one row per edge.

| Column | Type | Unit | Description | Formula |
|--------|------|------|-------------|---------|
| `z`, `y`, `x` | array | voxels | Coordinates along vessel | Path points |
| `length` | float | voxels | Path length | `Σ ‖pᵢ₊₁ - pᵢ‖` |
| `shortest_path` | float | voxels | Endpoint distance | `‖p_last - p_first‖` |
| `tortuosity` | float | ratio | Curvature measure | `length / shortest_path` (clipped [0,5]) |
| `volume` | float | voxels³ | Segment volume | `Σ area(pᵢ)` |
| `mean_cs_area` | float | voxels² | Mean cross-section | `mean(π × r_3D × r_2D)` |
| `median_cs_area` | float | voxels² | Median cross-section | `median(areas)` |
| `std_cs_area` | float | voxels² | Area variability | `std(areas)` |
| `is_sprout` | bool | - | Connected to endpoint | `degree(n1)=1 OR degree(n2)=1` |
| `node1_degree` | int | count | First node degree | - |
| `node2_degree` | int | count | Second node degree | - |
| `orientation_z/y/x` | float | ratio | Direction vector | `(p_last - p_first) / ‖·‖` |
| `sample` | str | - | Sample identifier | - |

**Tortuosity interpretation:**
- 1.0: Perfectly straight
- \>1.5: Significantly curved
- Clipped to [0, 5] for robustness

---

## 4.3 Junction Metrics (HDF5)

Stored in `{sample}_junction_metrics_{method}.h5`, one row per node.

| Column | Type | Unit | Description | Formula |
|--------|------|------|-------------|---------|
| `z`, `y`, `x` | float | voxels | Node position | - |
| `number_of_vessel_per_node` | int | count | Node degree | Edges connected |
| `is_sprout_tip` | bool | - | Is endpoint | `degree = 1` |
| `is_junction` | bool | - | Is branch point | `degree ≥ 2` |
| `dist_nearest_junction` | float | voxels | To closest junction | `min(‖pos - pos_j‖)` |
| `dist_nearest_endpoint` | float | voxels | To closest endpoint | `min(‖pos - pos_e‖)` |
| `num_junction_neighbors` | int | count | Junctions within threshold | Within 250 voxels |
| `num_endpoint_neighbors` | int | count | Endpoints within threshold | Within 250 voxels |
| `sample` | str | - | Sample identifier | - |

**Degree interpretation:**
- Degree 1: Sprout tip (terminal)
- Degree 3: Typical bifurcation
- Degree ≥ 4: Complex junction

---

## Unit Conversions

All metrics are in **voxels**. To convert to physical units:

```python
iso_voxel_um = 2.0  # µm per voxel (after isotropic rescaling)

# Length
length_um = length_voxels * iso_voxel_um

# Area
area_um2 = area_voxels * (iso_voxel_um ** 2)

# Volume
volume_um3 = volume_voxels * (iso_voxel_um ** 3)
```

---

# 5. Appendix

## 5.1 Mathematical Formulas

### Cross-Sectional Area

Modeled as ellipse at each skeleton point:

```
Area(p) = π × r_major × r_minor
        = π × EDT_2D(y, x) × EDT_3D(z, y, x)

where:
  EDT_3D = distance to nearest boundary in 3D
  EDT_2D = distance in XY max-projection
```

### Tortuosity

```
τ = L_path / L_straight
  = Σ‖pᵢ₊₁ - pᵢ‖ / ‖p_last - p_first‖

Interpretation:
  τ = 1: Straight
  τ > 1: Curved
```

### Fractal Dimension (Box-Counting)

```
D = lim(ε→0) [log N(ε) / log(1/ε)]

Implementation:
  scales = {2^k : k = 1..log₂(min_dim)}
  N(ε) = count of non-empty boxes at scale ε
  D = slope of linear fit: log(N) vs log(1/ε)
```

### Lacunarity

```
λ(ε) = Var(H) / Mean(H)²

where H = histogram of points per box at scale ε

Interpretation:
  λ ≈ 0: Homogeneous
  λ > 1: Heterogeneous/clustered
```

---

## 5.2 Troubleshooting

### CUDA Out of Memory

**Solution:** Reduce chunk sizes:
```python
cupy_chunk_processing(volume, func, chunk_size=(32, 256, 256))
```

### Empty Graph Output

**Causes:** No foreground, over-pruning  
**Solutions:**
1. Check `np.sum(mask) > 0`
2. Increase `edt_cutoff` (e.g., 0.5)
3. Decrease `length_cutoff` (e.g., 10)

### Mesh Contraction Timeout

**Solutions:**
1. Increase `time_lim` (e.g., 600)
2. Use `skel_method='thinning'` for dense networks

### NaN in Metrics

| Metric | Cause | Solution |
|--------|-------|----------|
| `tortuosity` | `shortest_path ≈ 0` | Filter short segments |
| `orientation_*` | Zero-length vector | Remove degenerate edges |
| `fractal_dimension` | Empty skeleton | Check mask |
| `dist_nearest_*` | No nodes of type | Expected for some samples |

---

## 5.3 Contributing

### Code Style

- PEP 8 compliant
- Google-style docstrings
- Line length: 100 characters

```bash
# Format before committing
black vascumap/
isort vascumap/
```

### Pull Request Process

1. Fork and create feature branch
2. Make changes with clear commits
3. Update documentation if needed
4. Submit PR with description

---

*Last updated: December 2024*
