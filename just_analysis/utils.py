import numpy as np
import cupy as cp
import tqdm
from cupyx.scipy import ndimage

def contrast(arr, low, high):
    """
    Adjust contrast of an array by clipping values between percentiles.

    Args:
        arr (np.ndarray): Input array.
        low (float): Lower percentile (0-100).
        high (float): Higher percentile (0-100).

    Returns:
        np.ndarray: Contrast-adjusted array.
    """
    return np.clip(arr, np.percentile(arr, low), np.percentile(arr, high))

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
    # Initialize output array
    result = np.empty_like(volume)
    
    # Create memory pool for explicit memory management
    pool = cp.get_default_memory_pool()
    
    # Process in chunks
    # Calculate total chunks for progress bar
    z_steps = range(0, volume.shape[0], chunk_size[0])
    
    for z in tqdm.tqdm(z_steps, desc="Processing chunks"):
        for y in range(0, volume.shape[1], chunk_size[1]):
            for x in range(0, volume.shape[2], chunk_size[2]):
                # Calculate chunk boundaries with overlap
                z_start = max(0, z - overlap[0])
                z_end = min(volume.shape[0], z + chunk_size[0] + overlap[0])
                
                y_start = max(0, y - overlap[1])
                y_end = min(volume.shape[1], y + chunk_size[1] + overlap[1])
                
                x_start = max(0, x - overlap[2])
                x_end = min(volume.shape[2], x + chunk_size[2] + overlap[2])
                
                # Extract chunk with overlap
                chunk = volume[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Process on GPU with dynamic arguments
                chunk_gpu = cp.asarray(chunk)
                
                # Create wrapper to handle GPU conversion of arguments
                def func_wrapper(gpu_chunk):
                    # Convert any CPU-based arguments to GPU arrays if they aren't already
                    # This part handles *args
                    gpu_args = []
                    for arg in args:
                        if isinstance(arg, np.ndarray):
                            gpu_args.append(cp.asarray(arg))
                        else:
                            gpu_args.append(arg)
                            
                    # This part handles **kwargs
                    gpu_kwargs = {}
                    for k, v in kwargs.items():
                        if isinstance(v, np.ndarray):
                            gpu_kwargs[k] = cp.asarray(v)
                        else:
                            gpu_kwargs[k] = v
                            
                    return processing_func(gpu_chunk, *gpu_args, **gpu_kwargs)
                
                filtered_chunk = func_wrapper(chunk_gpu)
                
                # Calculate write boundaries without overlap
                w_z_start = z - z_start
                w_z_end = w_z_start + chunk_size[0]
                
                w_y_start = y - y_start
                w_y_end = w_y_start + chunk_size[1]
                
                w_x_start = x - x_start
                w_x_end = w_x_start + chunk_size[2]
                
                # Copy valid region back to CPU
                # Need to be careful with shapes if filtered_chunk is smaller/larger (it shouldn't be for filters)
                
                # Handle indices relative to the filtered chunk
                # The filtered chunk corresponds to [z_start:z_end, ...].
                # We want the region [z:z+chunk_size, ...] which is offset by (z - z_start) inside the chunk
                
                valid_chunk = filtered_chunk[w_z_start:min(w_z_end, filtered_chunk.shape[0]), 
                                            w_y_start:min(w_y_end, filtered_chunk.shape[1]),
                                            w_x_start:min(w_x_end, filtered_chunk.shape[2])].get()
                
                # Write to output
                result_z = min(z, result.shape[0])
                result_y = min(y, result.shape[1])
                result_x = min(x, result.shape[2])
                
                result[result_z:result_z+valid_chunk.shape[0],
                    result_y:result_y+valid_chunk.shape[1],
                    result_x:result_x+valid_chunk.shape[2]] = valid_chunk
                
                # Explicit memory cleanup
                del chunk_gpu, filtered_chunk
                pool.free_all_blocks()
    
    return result

def median_filter_3d_gpu(volume, size=3, chunk_size=(64, 64, 64)):
    """
    Apply 3D median filter using GPU with chunked processing.

    Args:
        volume (np.ndarray): 3D input volume (Z, Y, X).
        size (int or tuple, optional): Median filter kernel size. Defaults to 3.
        chunk_size (tuple, optional): Chunk size. Defaults to (64, 64, 64).

    Returns:
        np.ndarray: Filtered volume with same shape as input.
    """
    return cupy_chunk_processing(
        volume, 
        ndimage.median_filter, 
        chunk_size=chunk_size, 
        overlap=(size//2, size//2, size//2) if isinstance(size, int) else (size[0]//2, size[1]//2, size[2]//2),
        size=size
    )

def resize_dask(stack, rescale_factor):
    """
    Resize a 3D stack using Dask for memory efficiency.

    Args:
        stack (np.ndarray): Input 3D stack.
        rescale_factor (list or tuple): Scaling factors for each dimension.

    Returns:
        np.ndarray: Resized stack.
    """
    import dask.array as da
    from skimage.transform import resize
    
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

def process_vessel_mask(vessel_proba, ortho=False):
    """
    Process vessel probability map into a binary mask.

    Applies median filtering (if ortho is True) and hysteresis thresholding.

    Args:
        vessel_proba (np.ndarray): Vessel probability map (values 0-1).
        ortho (bool, optional): Whether orthogonal plane predictions were used. 
            If True, applies extra filtering. Defaults to False.

    Returns:
        np.ndarray: Binary vessel mask.
    """
    from skimage.filters import apply_hysteresis_threshold
    if ortho:
        vessel_filtered = median_filter_3d_gpu(vessel_proba, size=7, chunk_size=(32, 1024, 1024))
        vessel_out = apply_hysteresis_threshold(vessel_filtered, 0.2, 0.5)
    else:
        vessel_out = apply_hysteresis_threshold(vessel_proba, 0.1, 0.5)
    return vessel_out

def remove_false_positives(mask_p):
    """
    Remove false positive detections in Z-direction based on Gaussian fitting of vessel area profile.

    Loads a mask, computes vessel area per slice, fits a Gaussian, and removes 
    regions that deviate significantly or are far from the peak.

    Args:
        mask_p (str): Path to the mask file. The file is overwritten with the corrected mask.
            Original mask is saved as backup.
    """
    import tifffile as tif
    from scipy.signal import find_peaks
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt
    import os

    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(-0.5 * ((x - mu) / sigma)**2)

    mask = tif.imread(mask_p)
    y = np.sum(np.sum(mask, axis=1), axis=1)
    x = np.arange(len(y))
    
    if len(y) == 0: return

    peaks = find_peaks(y, width=10)[0]
    if len(peaks) == 0: return
        
    center = peaks[np.argmin(np.abs(peaks - (len(y) // 2)))]
    width = len(y) // 3
    mask_proj = (x >= (center - width)) & (x <= (center + width))

    x_clipped = x[mask_proj]
    y_clipped = y[mask_proj]
    
    p0 = [np.max(y_clipped), center, 10]
    
    try:
        popt, pcov = curve_fit(gaussian, x_clipped, y_clipped, p0=p0, maxfev=800)
        amp, mu, sigma = popt
        fit = gaussian(x, amp, mu, sigma)
        t = mu-2*sigma
        
        diff = y - fit
        t_diff = 1e5
        ind_t = np.where(diff >= t_diff)[0]

        if (len(ind_t) > 0) and (True in list(ind_t < t)):
            dist_t = ind_t - t
            cutoff_ind = np.argmax(dist_t[dist_t <=0])
            cutoff = ind_t[dist_t <=0][cutoff_ind]
            
            mask_clipped = np.copy(mask)
            mask_clipped[:np.uint8(cutoff), ...] = 0
            
            os.rename(mask_p, mask_p.replace('mask', 'mask_backup'))
            tif.imwrite(mask_p, mask_clipped.astype(np.float32))
        else: 
            mask_clipped = mask
            cutoff = 0

        plt.figure(figsize=(30,15))
        plt.subplot(241)
        plt.plot(x, y, label='Vessel area along Z dim')
        plt.plot(x, fit, label='gaussian fit')
        plt.axvline(cutoff, label='Cutoff')
        plt.axvline(t, linestyle='--', label='Cutoff limit')
        plt.legend(fontsize=15)
        plt.title('Raw detection counts', fontsize=40)
        
        plt.subplot(245)
        plt.plot(x, np.sum(np.sum(mask_clipped, axis=1), axis=1), label='Vessel area along Z dim')
        plt.title('Corrected detection counts', fontsize=40)

        plt.subplot(142)
        plt.imshow(np.sum(mask, axis=0), cmap='magma')
        plt.title('Raw mask', fontsize=40)

        plt.subplot(143)
        plt.imshow(np.sum(mask_clipped, axis=0), cmap='magma')
        plt.title('Corrected mask', fontsize=40)

        if cutoff > 0:
             plt.subplot(144)
             plt.imshow(np.sum(mask[:np.uint8(cutoff), ...], axis=0), cmap='magma')
             plt.title('Residual', fontsize=40)

        plt.tight_layout()
        plt.savefig(mask_p.replace('mask.tif', 'mask_correction.png'), dpi=100)
        plt.close('all')

    except RuntimeError:
        print('Gaussian fit failed')
