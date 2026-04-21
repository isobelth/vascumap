import os
import sys
import ctypes
from pathlib import Path
from skimage.transform import resize
import numpy as np
from tqdm.auto import tqdm

# Make NVRTC/NVIDIA DLLs discoverable before importing CuPy.
if os.name == 'nt':
    torch_lib = Path(sys.prefix) / 'Lib' / 'site-packages' / 'torch' / 'lib'
    dll_dirs = [
        Path(sys.prefix) / 'Library' / 'bin',
        torch_lib,
    ]
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            os.add_dll_directory(str(dll_dir))

    if torch_lib.exists():
        os.environ['PATH'] = f"{torch_lib};" + os.environ.get('PATH', '')
        for name in ('nvrtc64_120_0.dll', 'nvrtc-builtins64_128.dll'):
            dll_path = torch_lib / name
            if dll_path.exists():
                ctypes.WinDLL(str(dll_path))

import cupy as cp
import pandas as pd


def compare_outputs(dir_a, dir_b, rtol=1e-4, atol=1e-6):
    """Compare analysis outputs between two pipeline runs.

    Checks ``*_analysis_metrics.csv`` numeric columns and key ``.npy``
    arrays (clean segmentation, skeleton, vessel mask).

    Parameters
    ----------
    dir_a, dir_b : str or Path
        Directories containing pipeline outputs to compare.
    rtol, atol : float
        Relative / absolute tolerance for numeric comparisons.

    Returns
    -------
    bool
        True if all checks pass.
    """
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    all_ok = True

    csv_a = list(dir_a.glob("*_analysis_metrics.csv"))
    csv_b = list(dir_b.glob("*_analysis_metrics.csv"))
    if csv_a and csv_b:
        df_a = pd.read_csv(csv_a[0]).select_dtypes(include="number")
        df_b = pd.read_csv(csv_b[0]).select_dtypes(include="number")
        for col in df_a.columns:
            if col not in df_b.columns:
                print(f"  [WARN] column '{col}' missing in B")
                continue
            va, vb = float(df_a[col].iloc[0]), float(df_b[col].iloc[0])
            if np.isnan(va) and np.isnan(vb):
                print(f"  [OK]   {col}: both NaN")
            elif abs(va - vb) > atol + rtol * max(abs(va), abs(vb), 1e-12):
                print(f"  [FAIL] {col}: A={va:.6g}  B={vb:.6g}  diff={abs(va - vb):.2e}")
                all_ok = False
            else:
                print(f"  [OK]   {col}: {va:.6g}")
    else:
        print("  [WARN] analysis_metrics.csv not found in one or both directories")
        all_ok = False

    for suffix in ("_clean_segmentation.npy", "_skeleton.npy", "_vessel_mask.npy"):
        files_a = list(dir_a.glob(f"*{suffix}"))
        files_b = list(dir_b.glob(f"*{suffix}"))
        if not files_a or not files_b:
            print(f"  [SKIP] {suffix} not found in one or both directories")
            continue
        arr_a = np.load(files_a[0])
        arr_b = np.load(files_b[0])
        if arr_a.shape != arr_b.shape:
            print(f"  [FAIL] {suffix}: shape {arr_a.shape} vs {arr_b.shape}")
            all_ok = False
            continue
        if not np.allclose(arr_a.astype(float), arr_b.astype(float), rtol=rtol, atol=atol):
            diff = np.abs(arr_a.astype(float) - arr_b.astype(float))
            print(f"  [FAIL] {suffix}: max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
            all_ok = False
        else:
            print(f"  [OK]   {suffix}: match (shape {arr_a.shape})")

    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return all_ok


def scale(arr):
    """
    Scale array values to range [0, 1].

    Args:
        arr (np.ndarray): Input array.

    Returns:
        np.ndarray: Scaled array.
    """
    arr = np.array(arr, dtype=np.float32)
    return (arr - arr.min()) / (arr.max() - arr.min())

def resize_dask(stack, rescale_factor):
    """
    Resize a 3D stack by the given per-axis scale factors.

    B4: replaced the previous map_blocks implementation which applied resize
    independently to each Dask chunk, producing incorrect boundary behaviour
    for multi-chunk volumes. skimage.transform.resize operates on the full
    array and is fast enough for the sizes used in this pipeline.

    Args:
        stack (np.ndarray): Input 3D stack.
        rescale_factor (list or tuple): Scaling factors for each dimension.

    Returns:
        np.ndarray: Resized stack, same dtype as input.
    """
    new_shape = tuple(max(1, int(round(stack.shape[i] * rescale_factor[i]))) for i in range(len(rescale_factor)))
    return resize(stack, new_shape, order=3, preserve_range=True, anti_aliasing=True).astype(stack.dtype)


def cupy_chunk_processing(volume, processing_func, chunk_size=(64, 512, 512),
                         overlap=(15, 15, 15), *args, **kwargs):
    """
    Apply a processing function to a 3D volume using GPU acceleration with chunking.

    Handles splitting the volume into chunks, processing each chunk on the GPU,
    and stitching the results back together, accounting for overlap to avoid
    boundary artifacts.

    Args:
        volume (np.ndarray): Input 3D volume.
        processing_func (callable): Function to apply to each chunk. Must accept
            a CuPy array as the first argument.
        chunk_size (tuple, optional): Size of chunks (Z, Y, X). Defaults to (64, 512, 512).
        overlap (tuple, optional): Overlap size (Z, Y, X). Defaults to (15, 15, 15).
        *args: Additional positional arguments passed to processing_func.
        **kwargs: Additional keyword arguments passed to processing_func.

    Returns:
        np.ndarray: Processed volume.
    """
    result = np.empty_like(volume)
    pool = cp.get_default_memory_pool()
    z_steps = range(0, volume.shape[0], chunk_size[0])

    # Pre-convert constant args/kwargs to CuPy once (avoids repeated CPU→GPU transfers)
    gpu_args = [cp.asarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
    gpu_kwargs = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}

    for z in tqdm(z_steps, desc="Processing chunks"):
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

                filtered_chunk = processing_func(chunk_gpu, *gpu_args, **gpu_kwargs)

                w_z_start = z - z_start
                w_z_end = w_z_start + chunk_size[0]
                w_y_start = y - y_start
                w_y_end = w_y_start + chunk_size[1]
                w_x_start = x - x_start
                w_x_end = w_x_start + chunk_size[2]

                valid_chunk = filtered_chunk[
                    w_z_start:min(w_z_end, filtered_chunk.shape[0]),
                    w_y_start:min(w_y_end, filtered_chunk.shape[1]),
                    w_x_start:min(w_x_end, filtered_chunk.shape[2]),
                ].get()

                result_z = min(z, result.shape[0])
                result_y = min(y, result.shape[1])
                result_x = min(x, result.shape[2])

                result[
                    result_z:result_z + valid_chunk.shape[0],
                    result_y:result_y + valid_chunk.shape[1],
                    result_x:result_x + valid_chunk.shape[2],
                ] = valid_chunk

                del chunk_gpu, filtered_chunk
                pool.free_all_blocks()

    return result