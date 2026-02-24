import dask.array as da
from skimage.transform import resize
import numpy as np

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