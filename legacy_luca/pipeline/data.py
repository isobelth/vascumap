import numpy as np
from readlif.reader import LifFile
from scipy.ndimage import gaussian_filter1d
from .utils import resize_dask

def get_stack_from_lif(lif_file, stack_index=0, ref_voxel_size=[5, 2, 2]):
    """
    Extract a 3D stack from a LIF file and resize it to a reference voxel size.

    Args:
        lif_file (str): Path to the LIF file.
        stack_index (int, optional): Index of the image within the LIF file. Defaults to 0.
        ref_voxel_size (list, optional): Target voxel size [z, y, x] in microns. Defaults to [5, 2, 2].

    Returns:
        tuple:
            - np.ndarray: Resized 3D stack (Z, Y, X, C).
            - np.ndarray: Original voxel size [z, y, x].
            - list: Original dimensions [nz, ny, nx].

    Raises:
        ValueError: If the Z-range of the stack is too thin (< 160um).
    """
    img = LifFile(lif_file).get_image(stack_index)
    nx, ny, nz = list(img.dims_n.values())
    nc = img.channels

    voxel_size = np.flip(1/np.array(img.scale)[:-1]) # Assuming X and Y are equal
    rescale_factor = voxel_size / ref_voxel_size

    if voxel_size[0] * nz < 160:
        raise ValueError(f"Z range is too thin, minimum is 160um, current is {voxel_size[0] * nz}um.")

    stack = np.zeros((nz, ny, nx, nc))
    for c in range(nc):
            for z in range(nz):
                z_im = np.array(img.get_frame(z=z, c=c))
                stack[z, :, :, c] = z_im
    
    return resize_dask(stack, rescale_factor), voxel_size, [nz, ny, nx]

def get_stack_from_lif_custom(lif_file, stack_index=0, ref_voxel_size=[5, 2, 2]):
    """
    Extract a cropped 3D stack from a LIF file centered on the signal in channel 2.

    This function identifies the Z-slice with the highest variation (focus) in 
    channel 2 and crops a range around it.

    Args:
        lif_file (str): Path to the LIF file.
        stack_index (int, optional): Index of the image. Defaults to 0.
        ref_voxel_size (list, optional): Target voxel size. Defaults to [5, 2, 2].

    Returns:
        tuple:
            - np.ndarray: Resized and cropped 3D stack.
            - np.ndarray: Original voxel size.
            - list: Original dimensions.
    """
    # Logic from original file if needed, currently seems redundant with get_stack_from_lif but has extra logic for cropping? 
    # The original had logic to find location of interest. keeping it.
    
    def smooth(y_values, strength):
        if not (0 <= strength <= 1): raise ValueError("Strength must be between 0.0 and 1.0")
        y = np.asarray(y_values)
        if y.ndim != 1: raise ValueError("Input must be 1-dimensional.")
        n = len(y)
        if n < 2 or strength == 0: return y.copy()
        sigma = max(0.1, strength * (n / 8.0)) 
        return gaussian_filter1d(y, sigma=sigma, mode='nearest')
    
    img = LifFile(lif_file).get_image(stack_index)
    nx, ny, nz = list(img.dims_n.values())
    nc = img.channels
    stack = np.zeros((nz, ny, nx, nc))
    for c in range(nc):
            for z in range(nz):
                z_im = np.array(img.get_frame(z=z, c=c))
                stack[z, :, :, c] = z_im

    y = np.std(np.std(stack[..., 2], axis=1), axis=1) # Hardcoded channel 2?

    loc_id = np.argmax(smooth(y, 0.2))
    low, high = np.clip(loc_id - 10, 0, nz), np.clip(loc_id +10, 0, nz)
    stack_crop = stack[low:high, :, :, :]

    voxel_size = np.flip(1/np.array(img.scale)[:-1])
    rescale_factor = voxel_size / ref_voxel_size

    return resize_dask(stack_crop, rescale_factor), voxel_size, [nz, ny, nx]
