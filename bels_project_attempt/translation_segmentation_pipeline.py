from __future__ import annotations

from pathlib import Path
from typing import Tuple
import math
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from monai.inferers import SlidingWindowInferer, SliceInferer
from monai.networks.nets import UNet
from scipy.ndimage import zoom as ndi_zoom

try:
    import cupy as cp
    from cupyx.scipy import ndimage
except Exception:
    cp = None
    ndimage = None

try:
    import dask.array as da
    from skimage.transform import resize
except Exception:
    da = None
    resize = None

try:
    from skimage.filters import apply_hysteresis_threshold
except Exception:
    apply_hysteresis_threshold = None


def legacy_scale(arr: np.ndarray) -> np.ndarray:
    arr = np.array(arr, dtype=np.float32)
    den = float(arr.max() - arr.min())
    if den <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - arr.min()) / den


def legacy_resize_dask(stack: np.ndarray, rescale_factor) -> np.ndarray:
    stack = np.asarray(stack)
    rf = np.asarray(rescale_factor, dtype=float)

    if da is not None and resize is not None:
        stack_dask = da.from_array(stack, chunks="auto")

        def _resize_block(block):
            out_shape = (
                max(1, int(block.shape[0] * rf[0])),
                max(1, int(block.shape[1] * rf[1])),
                max(1, int(block.shape[2] * rf[2])),
            )
            return resize(block, out_shape, order=3)

        rescaled_stack = stack_dask.map_blocks(_resize_block, dtype=stack.dtype)
        return np.asarray(rescaled_stack.compute(), dtype=np.float32)

    old_shape = np.array(stack.shape[:3], dtype=float)
    new_shape = np.maximum(1, (old_shape * rf).astype(int))
    zoom_factors = tuple(float(n) / float(o) for n, o in zip(new_shape, old_shape))
    return ndi_zoom(stack, zoom=zoom_factors, order=3, prefilter=False).astype(np.float32)


class Generator(nn.Module):
    def __init__(self, dropout_p=0.4):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256, 512),
            strides=(1, 2, 2, 2, 1),
            num_res_units=3,
            dropout=dropout_p,
        )

    def forward(self, x):
        x = self.unet(x)
        x = F.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, dropout_p=0.4):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(2, 64, kernel_size=4, stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=4, stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2),
            nn.Conv3d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class Pix2Pix(nn.Module):
    def __init__(self, generator_dropout_p=0.4, discriminator_dropout_p=0.4):
        super().__init__()
        self.generator = Generator(dropout_p=generator_dropout_p)
        self.discriminator = Discriminator(dropout_p=discriminator_dropout_p)

    def forward(self, x):
        return self.generator(x)


def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError("Weight format not supported by conversion.")
        repeat = int(math.ceil(in_chans / 3))
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= (3 / float(in_chans))
    return conv_weight.to(conv_type)


def adapt_input_model(model):
    if hasattr(model, "encoder") and hasattr(model.encoder, "patch_embed1"):
        new_weights = adapt_input_conv(in_chans=1, conv_weight=model.encoder.patch_embed1.proj.weight)
        model.encoder.patch_embed1.proj = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        with torch.no_grad():
            model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)
    return model


def load_pix2pix(model_path, device="cuda"):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = Pix2Pix()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def load_segmentation_model(model_path, device="cuda"):
    model = smp.Unet(encoder_name="mit_b5", classes=1, in_channels=3, encoder_weights=None)
    model = adapt_input_model(model)

    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    return model.eval()


def predict_pix2pix(model_p2p, stack_bf, device, n_iter=1):
    input_p2p_tensor = torch.tensor(stack_bf[None, None, ...], device="cpu").float()

    inferer = SlidingWindowInferer(
        roi_size=(32, 512, 512),
        overlap=(0.75, 0.25, 0.25),
        sw_batch_size=1,
        sw_device=device,
        device="cpu",
        progress=True,
    )

    model_p2p.to(device)

    autocast_device = "cuda" if "cuda" in str(device).lower() else "cpu"
    with torch.no_grad(), torch.amp.autocast(device_type=autocast_device):
        reconstruction_mean = torch.stack(
            [inferer(inputs=input_p2p_tensor, network=model_p2p).to("cpu") for _ in range(max(1, int(n_iter)))]
        )
        reconstruction_mean = torch.clip(reconstruction_mean, 0, 1)
        reconstruction_mean = torch.mean(reconstruction_mean, dim=0)

    vessel_pred = reconstruction_mean.squeeze().cpu().numpy()

    model_p2p.to("cpu")
    del input_p2p_tensor, reconstruction_mean
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return vessel_pred


def predict_mask_ortho(model_smp, vessel_pred_iso, device):
    vessel_pred_tensor = torch.tensor(vessel_pred_iso[None, None, ...], device="cpu").float()
    tensor_shape = vessel_pred_tensor.shape

    model_smp.to(device)

    axial_inferer = SliceInferer(
        roi_size=(1024, 1024),
        sw_batch_size=4,
        progress=True,
        mode="gaussian",
        overlap=0.5,
        device="cpu",
        sw_device=device,
    )

    with torch.no_grad():
        output3D_axial = torch.sigmoid(axial_inferer(vessel_pred_tensor, model_smp))

    vessel_axial = output3D_axial.squeeze().to("cpu").numpy()

    if tensor_shape[2] < 256:
        pad_d = 256 - tensor_shape[2]
        pad_pre = pad_d // 2
        pad_post = pad_d - pad_pre
        vessel_pred_tensor = F.pad(vessel_pred_tensor, (0, 0, 0, 0, pad_pre, pad_post), mode="constant", value=-1)

    coronal_inferer = SliceInferer(
        roi_size=(256, 256),
        sw_batch_size=32,
        spatial_dim=1,
        progress=True,
        mode="gaussian",
        overlap=0.5,
        device="cpu",
        sw_device=device,
    )

    sagital_inferer = SliceInferer(
        roi_size=(256, 256),
        sw_batch_size=32,
        spatial_dim=2,
        progress=True,
        mode="gaussian",
        overlap=0.5,
        device="cpu",
        sw_device=device,
    )

    with torch.no_grad():
        output3D_coronal = torch.sigmoid(coronal_inferer(vessel_pred_tensor, model_smp))
        output3D_sagital = torch.sigmoid(sagital_inferer(vessel_pred_tensor, model_smp))

    if tensor_shape[2] < 256:
        pad_d = 256 - tensor_shape[2]
        pad_pre = pad_d // 2
        vessel_coronal = output3D_coronal.squeeze().to("cpu").numpy()[pad_pre : pad_pre + tensor_shape[2], :, :]
        vessel_sagital = output3D_sagital.squeeze().to("cpu").numpy()[pad_pre : pad_pre + tensor_shape[2], :, :]
    else:
        vessel_coronal = output3D_coronal.squeeze().to("cpu").numpy()
        vessel_sagital = output3D_sagital.squeeze().to("cpu").numpy()

    vessel_proba = np.mean(np.array([vessel_axial, vessel_coronal, vessel_sagital]), axis=0)

    model_smp.to("cpu")
    del vessel_pred_tensor, output3D_axial, output3D_coronal, output3D_sagital
    if "cuda" in str(device).lower() and torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return vessel_proba


def cupy_chunk_processing(volume, processing_func, chunk_size=(64, 512, 512), overlap=(15, 15, 15), *args, **kwargs):
    if cp is None or ndimage is None:
        raise RuntimeError("CuPy is required for legacy ortho mask processing")

    result = np.empty_like(volume)
    pool = cp.get_default_memory_pool()

    z_steps = range(0, volume.shape[0], chunk_size[0])
    for z in z_steps:
        for y in range(0, volume.shape[1], chunk_size[1]):
            for x in range(0, volume.shape[2], chunk_size[2]):
                z_start = max(0, z - overlap[0])
                z_end = min(volume.shape[0], z + chunk_size[0] + overlap[0])
                y_start = max(0, y - overlap[1])
                y_end = min(volume.shape[1], y + chunk_size[1] + overlap[1])
                x_start = max(0, x - overlap[2])
                x_end = min(volume.shape[2], x + chunk_size[2] + overlap[2])

                chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                chunk_gpu = cp.asarray(chunk)

                filtered_chunk = processing_func(chunk_gpu, *args, **kwargs)

                w_z_start = z - z_start
                w_z_end = w_z_start + chunk_size[0]
                w_y_start = y - y_start
                w_y_end = w_y_start + chunk_size[1]
                w_x_start = x - x_start
                w_x_end = w_x_start + chunk_size[2]

                valid_chunk = filtered_chunk[
                    w_z_start : min(w_z_end, filtered_chunk.shape[0]),
                    w_y_start : min(w_y_end, filtered_chunk.shape[1]),
                    w_x_start : min(w_x_end, filtered_chunk.shape[2]),
                ].get()

                result_z = min(z, result.shape[0])
                result_y = min(y, result.shape[1])
                result_x = min(x, result.shape[2])

                result[
                    result_z : result_z + valid_chunk.shape[0],
                    result_y : result_y + valid_chunk.shape[1],
                    result_x : result_x + valid_chunk.shape[2],
                ] = valid_chunk

                del chunk_gpu, filtered_chunk
                pool.free_all_blocks()

    return result


def median_filter_3d_gpu(volume, size=3, chunk_size=(64, 64, 64)):
    if isinstance(size, int):
        overlap = (size // 2, size // 2, size // 2)
    else:
        overlap = (size[0] // 2, size[1] // 2, size[2] // 2)
    return cupy_chunk_processing(volume, ndimage.median_filter, chunk_size=chunk_size, overlap=overlap, size=size)


def process_vessel_mask(vessel_proba, ortho=False):
    if apply_hysteresis_threshold is None:
        raise RuntimeError("scikit-image is required for hysteresis thresholding")
    if ortho:
        vessel_filtered = median_filter_3d_gpu(vessel_proba, size=7, chunk_size=(32, 1024, 1024))
        vessel_out = apply_hysteresis_threshold(vessel_filtered, 0.2, 0.5)
    else:
        vessel_out = apply_hysteresis_threshold(vessel_proba, 0.1, 0.5)
    return vessel_out


def run_translation_and_segmentation(
    cropped_z_stack: np.ndarray,
    src_voxel_um: Tuple[float, float, float],
    pix2pix_model_path: str,
    seg_model_path: str,
    device: str = "cuda",
    n_iter: int = 1,
    ref_voxel_um: Tuple[float, float, float] = (5.0, 2.0, 2.0),
    iso_voxel_um: Tuple[float, float, float] = (2.0, 2.0, 2.0),
    device_width_um: float = 30.0,
    use_gpu: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    del use_gpu

    def _crop_xy_equal_margin(arr: np.ndarray, crop_y_px: int, crop_x_px: int) -> np.ndarray:
        out = np.asarray(arr)
        if out.ndim < 3:
            return out
        y0 = int(max(0, crop_y_px))
        x0 = int(max(0, crop_x_px))
        y1 = int(out.shape[1] - y0)
        x1 = int(out.shape[2] - x0)
        if y1 <= y0 or x1 <= x0:
            raise RuntimeError(
                f"Device-width crop too large for shape {tuple(out.shape[:3])}: crop_y={crop_y_px}, crop_x={crop_x_px}."
            )
        slicer = [slice(None)] * out.ndim
        slicer[1] = slice(y0, y1)
        slicer[2] = slice(x0, x1)
        return out[tuple(slicer)]

    stack_input_raw = np.asarray(cropped_z_stack, dtype=np.float32)
    if stack_input_raw.ndim != 3:
        raise ValueError(f"Expected cropped_z_stack with shape (Z,Y,X), got {stack_input_raw.shape}")

    src = np.asarray(src_voxel_um, dtype=float)
    ref = np.asarray(ref_voxel_um, dtype=float)
    iso = np.asarray(iso_voxel_um, dtype=float)

    # Match legacy pipeline flow: convert to legacy reference voxel grid first
    stack_input_ref = legacy_resize_dask(stack_input_raw, src / ref)
    stack_input_ref = legacy_scale(stack_input_ref)

    model_p2p = load_pix2pix(str(pix2pix_model_path), device=device)
    vessel_pred_ref = predict_pix2pix(model_p2p, stack_input_ref, device=device, n_iter=max(1, int(n_iter)))

    # Match legacy isotropic conversion step
    vessel_pred_iso = legacy_resize_dask(vessel_pred_ref, ref / iso).astype(np.float32)

    model_smp = load_segmentation_model(str(seg_model_path), device=device)
    vessel_proba_iso = predict_mask_ortho(model_smp, vessel_pred_iso, device=device).astype(np.float32)
    vessel_mask_iso = process_vessel_mask(vessel_proba_iso, ortho=True).astype(np.uint8)

    width_um = float(device_width_um)
    if width_um < 0:
        raise ValueError(f"device_width_um must be >= 0, got {width_um}")

    y_um_iso = float(iso[1])
    x_um_iso = float(iso[2])
    if y_um_iso <= 0 or x_um_iso <= 0:
        raise ValueError(f"iso_voxel_um must have positive y/x spacing, got {iso_voxel_um}")

    crop_y = int(np.ceil(width_um / y_um_iso))
    crop_x = int(np.ceil(width_um / x_um_iso))

    translated_iso = _crop_xy_equal_margin(vessel_pred_iso, crop_y_px=crop_y, crop_x_px=crop_x).astype(np.float32)
    seg_prob_map = _crop_xy_equal_margin(vessel_proba_iso, crop_y_px=crop_y, crop_x_px=crop_x).astype(np.float32)
    seg_mask = _crop_xy_equal_margin(vessel_mask_iso, crop_y_px=crop_y, crop_x_px=crop_x).astype(np.uint8)

    return translated_iso, seg_prob_map, seg_mask


def default_checkpoints(repo_root: str | Path) -> Tuple[str, str]:
    root = Path(repo_root)
    pix2pix_ckpt = root / "luca_models" / "epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt"
    seg_ckpt = root / "luca_models" / "best_full.pth"
    return str(pix2pix_ckpt), str(seg_ckpt)
