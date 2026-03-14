import os
import sys
import ctypes
from pathlib import Path
import dask.array as da
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
    Resize a 3D stack using Dask for memory efficiency.

    Args:
        stack (np.ndarray): Input 3D stack.
        rescale_factor (list or tuple): Scaling factors for each dimension.

    Returns:
        np.ndarray: Resized stack.
    """
    
    stack_dask = da.from_array(stack, chunks='auto')
    rescaled_stack = stack_dask.map_blocks(
        lambda block: resize(
            block, 
            (int(block.shape[0] * rescale_factor[0]),  
            int(block.shape[1] * rescale_factor[1]),
            int(block.shape[2] * rescale_factor[2]),
            ),
            order=3
        ),
        dtype=stack.dtype
    )
    return rescaled_stack.compute()


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

                def func_wrapper(gpu_chunk):
                    gpu_args = []
                    for arg in args:
                        if isinstance(arg, np.ndarray):
                            gpu_args.append(cp.asarray(arg))
                        else:
                            gpu_args.append(arg)

                    gpu_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, np.ndarray):
                            gpu_kwargs[k] = cp.asarray(v)
                        else:
                            gpu_kwargs[k] = v

                    return processing_func(gpu_chunk, *gpu_args, **gpu_kwargs)

                filtered_chunk = func_wrapper(chunk_gpu)

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