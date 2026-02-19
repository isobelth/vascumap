from __future__ import annotations

from pathlib import Path
from typing import Tuple

import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import SlidingWindowInferer, SliceInferer
from monai.networks.nets import UNet
import segmentation_models_pytorch as smp
from scipy.ndimage import zoom as ndi_zoom
from skimage.filters import apply_hysteresis_threshold

try:
    import cupy as cp
    from cupyx.scipy import ndimage as cpx_ndimage
except Exception:
    cp = None
    cpx_ndimage = None


def image_resizer(
    stack_input: np.ndarray,
    src_voxel_um: Tuple[float, float, float],
    ref_voxel_um: Tuple[float, float, float],
    order: int = 3,
    use_gpu: bool = True,
) -> np.ndarray:
    stack = np.asarray(stack_input, dtype=np.float32)
    src = np.array(src_voxel_um, dtype=float)
    ref = np.array(ref_voxel_um, dtype=float)
    old_shape = np.array(stack.shape[:3], dtype=float)
    new_shape = np.maximum(1, np.rint(old_shape * (src / ref))).astype(int)
    zoom_factors = tuple(float(n) / float(o) for n, o in zip(new_shape, old_shape))

    if all(np.isclose(zf, 1.0) for zf in zoom_factors):
        return stack.astype(np.float32, copy=False)

    can_gpu = bool(use_gpu and (cp is not None) and (cpx_ndimage is not None))
    if can_gpu:
        out = cpx_ndimage.zoom(cp.asarray(stack), zoom=zoom_factors, order=order).get()
    else:
        out = ndi_zoom(stack, zoom=zoom_factors, order=order, prefilter=False)
    return out.astype(np.float32)


def pix2pix_translation(
    stack_input: np.ndarray,
    model_path: str,
    device: str = "cuda",
    n_iter: int = 1,
) -> np.ndarray:
    class LegacyGenerator(nn.Module):
        def __init__(self, dropout_p: float = 0.4):
            super().__init__()
            self.unet = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                channels=(32, 64, 128, 256, 512),
                strides=(1, 2, 2, 2, 1),
                num_res_units=3,
                dropout=float(dropout_p),
            )

        def forward(self, x):
            return F.relu(self.unet(x))

    checkpoint = torch.load(model_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    gen_items = [(k, v) for k, v in state_dict.items() if k.startswith("generator.")]

    if gen_items:
        gen_sd = {k[len("generator."):]: v for k, v in gen_items}
    elif "generator" in state_dict and isinstance(state_dict["generator"], dict):
        gen_sd = state_dict["generator"]
    else:
        raise RuntimeError("Could not find generator weights with prefix 'generator.'.")

    gen_sd = {(k[len("module."):] if k.startswith("module.") else k): v for k, v in gen_sd.items()}
    hp = checkpoint.get("hyper_parameters", {}) if isinstance(checkpoint, dict) else {}
    dropout_p = float(hp.get("generator_dropout_p", 0.4)) if isinstance(hp, dict) else 0.4

    model = LegacyGenerator(dropout_p=dropout_p)
    model.load_state_dict(gen_sd, strict=True)
    model = model.to(device).eval()

    stack = np.asarray(stack_input, dtype=np.float32)
    denom = float(stack.max() - stack.min())
    if denom > 0:
        stack = (stack - float(stack.min())) / denom
    else:
        stack = np.zeros_like(stack, dtype=np.float32)

    input_tensor = torch.tensor(stack[None, None, ...], device="cpu").float()
    inferer = SlidingWindowInferer(
        roi_size=(32, 512, 512),
        overlap=(0.75, 0.25, 0.25),
        sw_batch_size=1,
        sw_device=device,
        device="cpu",
        progress=True,
    )

    with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
        preds = torch.stack(
            [inferer(inputs=input_tensor, network=model).to("cpu") for _ in range(max(1, int(n_iter)))]
        )
        preds = torch.clip(preds, 0, 1)
        preds = torch.mean(preds, dim=0)

    out = preds.squeeze().cpu().numpy().astype(np.float32)
    model.to("cpu")
    del input_tensor, preds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return out


def u_net_segmentation(stack_input: np.ndarray, model_path: str, device: str = "cuda") -> Tuple[np.ndarray, np.ndarray]:
    def _adapt_input_conv(in_chans, conv_weight):
        conv_type = conv_weight.dtype
        conv_weight = conv_weight.float()
        o_ch, i_ch, k_h, k_w = conv_weight.shape
        if in_chans == 1:
            if i_ch > 3:
                conv_weight = conv_weight.reshape(o_ch, i_ch // 3, 3, k_h, k_w)
                conv_weight = conv_weight.sum(dim=2, keepdim=False)
            else:
                conv_weight = conv_weight.sum(dim=1, keepdim=True)
        elif in_chans != 3:
            repeat = int(np.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
        return conv_weight.to(conv_type)

    model = smp.Unet(encoder_name="mit_b5", classes=1, in_channels=3, encoder_weights=None)
    if hasattr(model, "encoder") and hasattr(model.encoder, "patch_embed1"):
        new_weights = _adapt_input_conv(in_chans=1, conv_weight=model.encoder.patch_embed1.proj.weight)
        model.encoder.patch_embed1.proj = torch.nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        with torch.no_grad():
            model.encoder.patch_embed1.proj.weight = torch.nn.parameter.Parameter(new_weights)

    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        model.load_state_dict({(k[6:] if k.startswith("model.") else k): v for k, v in state_dict.items()})
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device).eval()
    tensor = torch.tensor(np.asarray(stack_input, dtype=np.float32)[None, None, ...], device="cpu").float()
    tensor_shape = tensor.shape

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
        proba_axial = torch.sigmoid(axial_inferer(tensor, model)).squeeze().to("cpu").numpy()

    if tensor_shape[2] < 256:
        pad_d = 256 - tensor_shape[2]
        pad_pre = pad_d // 2
        pad_post = pad_d - pad_pre
        tensor = F.pad(tensor, (0, 0, 0, 0, pad_pre, pad_post), mode="constant", value=-1)

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
    sagittal_inferer = SliceInferer(
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
        proba_cor = torch.sigmoid(coronal_inferer(tensor, model)).squeeze().to("cpu").numpy()
        proba_sag = torch.sigmoid(sagittal_inferer(tensor, model)).squeeze().to("cpu").numpy()

    if tensor_shape[2] < 256:
        pad_d = 256 - tensor_shape[2]
        pad_pre = pad_d // 2
        proba_cor = proba_cor[pad_pre : pad_pre + tensor_shape[2], :, :]
        proba_sag = proba_sag[pad_pre : pad_pre + tensor_shape[2], :, :]

    prob_map = np.mean(np.array([proba_axial, proba_cor, proba_sag]), axis=0)

    if cpx_ndimage is not None and cp is not None:
        filt = cpx_ndimage.median_filter(cp.asarray(prob_map), size=7).get()
    else:
        from scipy.ndimage import median_filter

        filt = median_filter(prob_map, size=7)

    seg_mask = apply_hysteresis_threshold(filt, 0.2, 0.5).astype(np.uint8)

    model.to("cpu")
    del tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return prob_map.astype(np.float32), seg_mask


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

    stack_input_ref = image_resizer(
        stack_input=stack_input_raw,
        src_voxel_um=tuple(float(v) for v in src_voxel_um),
        ref_voxel_um=tuple(float(v) for v in ref_voxel_um),
        order=3,
        use_gpu=bool(use_gpu),
    )

    translated_ref = pix2pix_translation(
        stack_input=stack_input_ref,
        model_path=str(pix2pix_model_path),
        device=device,
        n_iter=int(max(1, n_iter)),
    )

    translated_iso = image_resizer(
        stack_input=translated_ref,
        src_voxel_um=tuple(float(v) for v in ref_voxel_um),
        ref_voxel_um=tuple(float(v) for v in iso_voxel_um),
        order=3,
        use_gpu=bool(use_gpu),
    ).astype(np.float32)

    seg_prob_map, seg_mask = u_net_segmentation(
        stack_input=translated_iso,
        model_path=str(seg_model_path),
        device=device,
    )

    try:
        width_um = float(device_width_um)
    except Exception as exc:
        raise ValueError(f"device_width_um must be numeric, got {device_width_um!r}") from exc

    if width_um < 0:
        raise ValueError(f"device_width_um must be >= 0, got {width_um}")

    _, y_um_iso, x_um_iso = (float(iso_voxel_um[0]), float(iso_voxel_um[1]), float(iso_voxel_um[2]))
    if y_um_iso <= 0 or x_um_iso <= 0:
        raise ValueError(f"iso_voxel_um must have positive y/x spacing, got {iso_voxel_um}")

    crop_y = int(np.rint(width_um / y_um_iso))
    crop_x = int(np.rint(width_um / x_um_iso))

    translated_iso = _crop_xy_equal_margin(translated_iso, crop_y_px=crop_y, crop_x_px=crop_x).astype(np.float32)
    seg_prob_map = _crop_xy_equal_margin(seg_prob_map, crop_y_px=crop_y, crop_x_px=crop_x).astype(np.float32)
    seg_mask = _crop_xy_equal_margin(seg_mask, crop_y_px=crop_y, crop_x_px=crop_x).astype(np.uint8)

    return translated_iso, np.asarray(seg_prob_map, dtype=np.float32), np.asarray(seg_mask, dtype=np.uint8)


def default_checkpoints(repo_root: str | Path) -> Tuple[str, str]:
    root = Path(repo_root)
    pix2pix_ckpt = root / "luca_models" / "epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt"
    seg_ckpt = root / "luca_models" / "best_full.pth"
    return str(pix2pix_ckpt), str(seg_ckpt)