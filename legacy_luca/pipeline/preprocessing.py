import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.optimize import curve_fit, basinhopping, minimize
from skimage.filters import sobel, sobel_h, sobel_v, gaussian
from skimage.transform import estimate_transform, resize, warp, rescale
from .utils import contrast, scale
import os

def auto_focus(stack_bf, crop_window=50, n_sampling=20, window_size=50, viz=False, stack_chan=None): 
    """
    Performs auto-focus operation on a given image stack.

    Finds the optimal focal plane for each pixel position in the XY plane by fitting a plane 
    in the Z (focal) direction to sampled focus points.

    Args:
        stack_bf (np.ndarray): The input 3D stack of images in ZXY format.
        crop_window (int, optional): Border margin to ignore. Defaults to 50.
        n_sampling (int, optional): Number of sampling points in XY plane. Defaults to 20.
        window_size (int, optional): Size of the window used for focus measure. Defaults to 50.
        viz (bool, optional): If True, displays the focused image and grid. Defaults to False.
        stack_chan (np.ndarray or list, optional): Additional stack(s) to apply the same focusing. Defaults to None.

    Returns:
        tuple or np.ndarray:
            - focused_image (np.ndarray): Auto-focused image in XY plane.
            - focused_channels (np.ndarray or list, optional): Focused additional channels (if stack_chan provided).
            - grid (np.ndarray): The height map (Z-indices) of the focal plane.
    """

    def find_focus(patch): 
        """
        Determines the optimal focus plane for an image patch using Sobel derivative.
        """
        focus_list = []
        for p in patch:
            sobel_im = sobel(p)
            focus_list.append(np.std(np.sqrt(sobel_im**2)))
        return np.where(focus_list == np.max(focus_list))[0][0]

    def my_fun0(x, y, stack):
        patch = stack[:, x-window_size:x+window_size, y-window_size:y+window_size]
        return find_focus(patch)

    def plane_func(X, a, b, c):
        x, y = X
        return a * x + b * y + c

    nz, nx, ny = stack_bf.shape

    x_sampling = np.linspace(window_size+1+crop_window, nx-1-crop_window-window_size, n_sampling).astype('int')
    y_sampling = np.linspace(window_size+1+crop_window, ny-1-crop_window-window_size, n_sampling).astype('int')

    xs, ys = np.meshgrid(x_sampling, y_sampling)
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    results = Parallel(n_jobs=20, prefer="threads")(delayed(my_fun0)(x, y, stack_bf) for x, y in zip(xs.flatten(), ys.flatten()))

    xyz = np.vstack((xs.flatten(), ys.flatten(), results))
    coefficients, _ = curve_fit(plane_func, xyz[:2], xyz[2])
    a, b, c = coefficients
    
    grid_1D = np.round(plane_func([xx.flatten(), yy.flatten()], a, b, c)).astype(int)
    grid = np.reshape(grid_1D, (ny, nx)).T
    grid = np.clip(grid, 0, nz-1) 

    focus = stack_bf[grid, np.arange(grid.shape[0])[:, None], np.arange(grid.shape[1])]

    if viz: 
        plt.figure()
        ax = plt.subplot(121)
        plt.imshow(focus, cmap='gray')
        [plt.axvline(x=i, c=[1,1,1,0.2]) for i in np.arange(0, focus.shape[1], window_size)]
        [plt.axhline(y=i, c=[1,1,1,0.2]) for i in np.arange(0, focus.shape[0], window_size)]

        plt.subplot(122, sharex=ax, sharey=ax)
        plt.imshow(grid, cmap='gray')
        plt.scatter(ys.ravel(), xs.ravel(), 100, results, cmap='jet')

    if stack_chan is not None: 
        if len(np.shape(stack_chan)) > 3:
            focus_chan = [sc[grid, np.arange(grid.shape[0])[:, None], np.arange(grid.shape[1])] for sc in stack_chan]
        else:
            focus_chan = stack_chan[grid, np.arange(grid.shape[0])[:, None], np.arange(grid.shape[1])]
        return focus, focus_chan, grid
    else: 
        return focus, grid

def transform_coords(coords, params):
    """
    Apply affine transformation (scale, rotation, translation) to coordinates.

    Args:
        coords (np.ndarray): Input coordinates (N, 2).
        params (list): Transformation parameters [scale, rotation_angle, tx, ty].

    Returns:
        np.ndarray: Transformed coordinates.
    """
    scale, theta, tx, ty = params
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta), np.cos(theta)]])
    transformed = scale * (coords @ rot_matrix) + np.array([tx, ty])
    return transformed

def objective_function(params, coords, canny_image):
    """
    Objective function for registration optimization.

    Computes a score based on how well the transformed coordinates align with 
    edges in the canny_image.

    Args:
        params (list): Transformation parameters.
        coords (np.ndarray): Source coordinates.
        canny_image (np.ndarray): Target edge map.

    Returns:
        float: Negative score (for minimization).
    """
    transformed_coords = transform_coords(coords, params)
    h, w = canny_image.shape
    
    valid_mask = ((transformed_coords[:, 0] >= 0) & 
                 (transformed_coords[:, 0] < h) & 
                 (transformed_coords[:, 1] >= 0) & 
                 (transformed_coords[:, 1] < w))
    
    valid_ratio = np.sum(valid_mask) / len(coords)
    
    if valid_ratio < 0.9:
        return 1e6 * (0.9 - valid_ratio) 
    
    weights = np.zeros(len(coords))
    responses = np.zeros(len(coords))
    
    valid_coords = transformed_coords[valid_mask]
    y_coords = np.clip(valid_coords[:, 0].round().astype(int), 0, h-1)
    x_coords = np.clip(valid_coords[:, 1].round().astype(int), 0, w-1)
    
    valid_responses = canny_image[y_coords, x_coords]
    
    weights[valid_mask] = 1.0
    responses[valid_mask] = valid_responses
    
    weights[valid_mask] = np.where(valid_responses > 0, 1.1, 1.0)
    
    weighted_avg = np.sum(responses * weights) / np.sum(weights) if np.sum(weights) > 0 else 0
    
    penalty_factor = 1.0 - (valid_ratio - 0.9) 
    
    return -(weighted_avg * valid_ratio) * (1 + penalty_factor)

def register_neck_coords_advanced(neck_coords, canny_image, initial_params, method='Powell'):
    """
    Register neck coordinates to image edges using advanced optimization.

    Uses basin hopping (global search) followed by local minimization to find
    optimal transformation parameters.

    Args:
        neck_coords (np.ndarray): Source coordinates of the neck/template.
        canny_image (np.ndarray): Edge map of the target image.
        initial_params (list): Initial guess for [scale, rotation, tx, ty].
        method (str, optional): Minimization method. Defaults to 'Powell'.

    Returns:
        tuple:
            - transformed_coords (np.ndarray): Registered coordinates.
            - optimal_params (np.ndarray): Optimal transformation parameters.
            - best_score (float): Best objective function value.
    """
    h, _ = canny_image.shape
    bounds = [(0.999, 1.001), (-1.5, 1.5), (-500, 500), (-500, 500)]

    n_random_starts = 3
    random_starts = []
    random_starts.append(initial_params)
    
    scale_variations = [0.5, 0.75, 1.5, 2]
    rotation_variations = [-0.5, -0.25, 0.25, 0.5]
    tx_variations = [-100, -50, 50, 100]
    ty_variations = [-100, -50, 50, 100]
    
    for s in scale_variations:
        for r in rotation_variations:
            for tx in tx_variations:
                for ty in ty_variations:
                    if len(random_starts) < n_random_starts:
                        random_starts.append([
                            initial_params[0] * s,
                            initial_params[1] + r,
                            initial_params[2] + tx,
                            initial_params[3] + ty
                        ])
    
    def take_step(x):
        scale_step = np.random.uniform(-0.1, 0.1)
        rotation_step = np.random.uniform(-0.1, 0.1)
        tx_step = np.random.uniform(-20, 20)
        ty_step = np.random.uniform(-20, 20)
        return x + np.array([scale_step, rotation_step, tx_step, ty_step])

    def accept_test(f_new, x_new, f_old, x_old):
        for i, (lower, upper) in enumerate(bounds):
            if x_new[i] < lower or x_new[i] > upper:
                return False
        return True

    best_result = None
    best_score = float('inf')
    
    for start_params in random_starts:
        try:
            basin_result = basinhopping(
                objective_function,
                start_params,
                niter=50, 
                T=1.0, 
                stepsize=1.0,
                take_step=take_step,
                accept_test=accept_test,
                minimizer_kwargs={
                    'args': (neck_coords, canny_image),
                    'bounds': bounds,
                    'method': method,
                    'options': {'maxiter': 1000, 'disp': False}
                }
            )
            
            result = minimize(
                objective_function,
                basin_result.x,
                args=(neck_coords, canny_image),
                bounds=bounds,
                method=method,
                options={'maxiter': 2000, 'disp': False}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result

        except Exception as e:
            print(f"Optimization failed for starting point {start_params}: {str(e)}")
            continue
    
    if best_result is None:
        return None, None, None # fixed return count
        
    optimal_params = best_result.x
    transformed_coords = transform_coords(neck_coords, optimal_params)
    best_score = best_result.fun
    
    return transformed_coords, optimal_params, best_score

def register_high_level(edges, neck_coords_all):
    """
    High-level wrapper for registration.

    Initializes parameters based on centering the template on the image 
    before running advanced registration.

    Args:
        edges (np.ndarray): Edge map of the target image.
        neck_coords_all (np.ndarray): Template coordinates.

    Returns:
        tuple: Results from register_neck_coords_advanced.
    """
    h, w = edges.shape
    x_min, x_max = neck_coords_all[:, 1].min(), neck_coords_all[:, 1].max()
    y_min, y_max = neck_coords_all[:, 0].min(), neck_coords_all[:, 0].max()
    
    current_x_center = (x_min + x_max) / 2
    target_x_center = w / 2
    
    current_y_center = (y_min + y_max) / 2
    target_y_center = h / 2
    
    tx = target_x_center - current_x_center
    ty = target_y_center - current_y_center
    
    initial_params = [1.0, 0.0, ty, tx]

    registered_coords, optimal_params, best_score = register_neck_coords_advanced(
        neck_coords_all, 
        edges, 
        initial_params,
        method='Powell'
    )

    return registered_coords, optimal_params, best_score

def apply_affine_crop_simple(image: np.ndarray, registered_rect: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Apply affine transformation to crop a quadrilateral region defined by registered_rect.

    The region is warped into a rectangular output image.

    Args:
        image (np.ndarray): Input image (2D or 3D).
        registered_rect (np.ndarray): 4x2 array of (y, x) corners of the quadrilateral.
        order (int, optional): Interpolation order. Defaults to 1.

    Returns:
        np.ndarray: Cropped and warped image.
    """
    if not (image.ndim == 3 or image.ndim == 4):
        raise ValueError("Image must be 2D (Y,X,C) or 3D (Z,Y,X,C)")
    if not (isinstance(registered_rect, np.ndarray) and registered_rect.ndim == 2 and registered_rect.shape == (4, 2)):
        raise ValueError("registered_rect must be a 4x2 NumPy array of (y,x) coordinates.")

    is_3d = image.ndim == 4
    num_channels = image.shape[-1] 

    v0, v1, v2, v3 = registered_rect
    L01 = np.linalg.norm(v1 - v0)
    L12 = np.linalg.norm(v2 - v1)
    L23 = np.linalg.norm(v3 - v2)
    L30 = np.linalg.norm(v0 - v3)

    W_out = int(np.round((L01 + L23) / 2))
    H_out = int(np.round((L12 + L30) / 2))

    if W_out <= 0 or H_out <= 0:
        empty_dtype = np.float32 
        if is_3d:
            return np.zeros((image.shape[0], 0, 0, num_channels), dtype=empty_dtype)
        else:
            return np.zeros((0, 0, num_channels), dtype=empty_dtype)

    src_xy = registered_rect[:, ::-1].astype(np.float64)  
    dst_xy = np.array([
        [0, 0],
        [W_out - 1, 0],
        [W_out - 1, H_out - 1],
        [0, H_out - 1]
    ], dtype=np.float64)

    tform = estimate_transform('affine', src_xy, dst_xy)

    if tform is None:
        print("Warning: Affine transform estimation failed. Returning empty image.")
        empty_dtype = np.float32
        if is_3d:
            return np.zeros((image.shape[0], H_out, W_out, num_channels), dtype=empty_dtype)
        else:
            return np.zeros((H_out, W_out, num_channels), dtype=empty_dtype)

    output_yx_shape = (H_out, W_out)
    warp_dtype = np.float32 

    if is_3d: 
        warped_image_shape = (image.shape[0], *output_yx_shape, num_channels)
        warped_image = np.zeros(warped_image_shape, dtype=warp_dtype)
        
        for z_idx in range(image.shape[0]):
            slice_yxc = image[z_idx, ...] 
            
            warped_slice_yxc = warp(slice_yxc,
                                    tform.inverse,
                                    output_shape=output_yx_shape,
                                    order=order,
                                    preserve_range=True,
                                    mode='constant', cval=0)
            
            if warped_slice_yxc.ndim == 2 and num_channels == 1:
                 warped_slice_yxc = np.expand_dims(warped_slice_yxc, axis=-1)
            
            warped_image[z_idx, ...] = warped_slice_yxc
    else:
        warped_image = warp(image,
                            tform.inverse,
                            output_shape=output_yx_shape,
                            order=order,
                            preserve_range=True,
                            mode='constant', cval=0)

        if warped_image.ndim == 2 and num_channels == 1:
            warped_image = np.expand_dims(warped_image, axis=-1)
            
    return warped_image

def crop_stack_z(stack: np.ndarray, grid: np.ndarray, z_range: float, ref_voxel_size: tuple) -> np.ndarray:
    """
    Crop a stack in Z based on a height map (grid).

    Extracts a slab of thickness z_range centered around the grid surface.

    Args:
        stack (np.ndarray): Input 3D stack.
        grid (np.ndarray): Height map (Y, X) of Z-indices.
        z_range (float): Thickness of the slab in microns.
        ref_voxel_size (tuple): Voxel size [z, y, x].

    Returns:
        np.ndarray: Z-cropped stack.
    """
    z_max       = stack.shape[0]
    z_range_px  = int(round(z_range / ref_voxel_size[0]))

    if z_range_px > z_max:
        z_max_um = z_max * ref_voxel_size[0]
        if z_max < 32: 
            raise ValueError(f"Z range is too thin, minimum is 160um, current is {z_max_um}um.")
        z_range_px = z_max
        print(f"Z range ({z_range}um) is larger than the whole stack, setting to stack size of {z_max_um}um.")

    half_win = z_range_px // 2
    low  = grid - half_win
    high = low  + (z_range_px - 1)        

    shift = np.where(low  < 0,           -low,              0)
    shift = np.where(high >= z_max, (z_max - 1) - high,    shift)

    deltas  = np.arange(z_range_px) - half_win
    z_idx   = grid[None, :, :] + shift[None, :, :] + deltas[:, None, None]
    z_idx   = np.clip(z_idx, 0, z_max - 1)

    y_idx, x_idx = np.indices(grid.shape)
    y_idx = np.broadcast_to(y_idx, z_idx.shape)
    x_idx = np.broadcast_to(x_idx, z_idx.shape)

    z_cropped_stack = stack[z_idx, y_idx, x_idx]
    return z_cropped_stack

def crop_stack(stack, crop_profile, bf_index, mode='3D', z_range=200, ref_voxel_size=[5, 2, 2], lif_path_root=None, filename=None):
    """
    Perform full stack cropping: Auto-focus, registration to template, and geometric cropping.

    Args:
        stack (np.ndarray): Input stack.
        crop_profile (str): Name of the crop profile to load template from.
        bf_index (int): Index of the brightfield channel for focusing/registration.
        mode (str, optional): '2D' or '3D' cropping. Defaults to '3D'.
        z_range (float, optional): Z-range for 3D cropping. Defaults to 200.
        ref_voxel_size (list, optional): Voxel size. Defaults to [5, 2, 2].
        lif_path_root (str, optional): Path root for saving debug images.
        filename (str, optional): Filename for saving debug images.

    Returns:
        np.ndarray: The final cropped stack.
    """
    nz, ny, nx, nc = stack.shape
    crop_window = 0.20*nx
    if crop_window < 200:
        crop_window = 200

    focus, grid = auto_focus(
        stack_bf=stack[..., bf_index], 
        viz=False,
        crop_window=crop_window, 
        n_sampling=5, 
        window_size=400
        )

    # Note: paths to profiles are hardcoded in original, this needs to be parameterized or relative
    # Assuming default location or passed as arg would be better.
    # For now, I will warn or use a placeholder. The original used C:\ANALYSIS\crop_profiles...
    # I'll use the current working directory or a config.
    
    profile_path = os.path.join("crop_profiles", f"{crop_profile}_rect.npy")
    fiducials_path = os.path.join("crop_profiles", f"{crop_profile}_fiducials.npy")
    
    # If not found, maybe check C:\ANALYSIS\crop_profiles as fallback to keep backward compatibility if user has that setup
    if not os.path.exists(profile_path):
         if os.path.exists(f'C:\\ANALYSIS\\crop_profiles\\{crop_profile}_rect.npy'):
             profile_path = f'C:\\ANALYSIS\\crop_profiles\\{crop_profile}_rect.npy'
             fiducials_path = f'C:\\ANALYSIS\\crop_profiles\\{crop_profile}_fiducials.npy'
         else:
             raise FileNotFoundError(f"Crop profile files not found for {crop_profile}. Expected at {profile_path} or C:\\ANALYSIS\\crop_profiles")

    rect_coords = np.load(profile_path)
    neck_coords_all = np.load(fiducials_path)

    im = scale(contrast(focus, 0.1, 99.9))
    im_r = rescale(im, 0.1)

    edges_x = sobel_h(im_r)
    edges_y = sobel_v(im_r)
    edges = np.sqrt(edges_x**2 + edges_y**2)

    edges = edges[3:-3, 3:-3]
    edges = np.pad(edges, ((100, 100), (0, 0)), mode='edge')

    edges = gaussian(edges, sigma=2.5)
    edges = edges[97:-97, :]
    edges = np.clip(edges, 0, 0.15)
    edges = resize(edges, (im.shape[0], im.shape[1]))

    registered_coords, optimal_params, best_score = register_high_level(edges, neck_coords_all)
    registered_rect = transform_coords(rect_coords, optimal_params)

    # Visualize results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(edges, cmap='gray')
    plt.scatter(neck_coords_all[:, 1], neck_coords_all[:, 0], color=[1,0,0], label='Reference', s=3)
    plt.scatter(registered_coords[:, 1], registered_coords[:, 0], color=[0,1,0], label='Registered', s=3)
    plt.plot(registered_rect[:, 1], registered_rect[:, 0], color=[0,1,0])
    plt.plot(registered_rect[[0,-1], 1], registered_rect[[0,-1], 0], color=[0,1,0])
    plt.title('Blurred Sobel')
    plt.legend()

    plt.subplot(122)
    plt.imshow(im, cmap='gray')
    plt.scatter(registered_coords[:, 1], registered_coords[:, 0], color=[0,1,0], label='Registered', s=3)
    plt.plot(registered_rect[:, 1], registered_rect[:, 0], color=[0,1,0])
    plt.plot(registered_rect[[0,-1], 1], registered_rect[[0,-1], 0], color=[0,1,0])
    plt.title('Input image')
    plt.legend()

    if lif_path_root and filename:
        plt.savefig(os.path.join(lif_path_root, filename + '_crop_overview.png'), dpi=200)
    plt.close('all')

    registered_rect = registered_rect.astype(int)

    if mode == '2D':
        focused_slice = stack[grid, np.arange(grid.shape[0])[:, None], np.arange(grid.shape[1]), ...]     
        output = apply_affine_crop_simple(focused_slice, registered_rect, order=3)

    elif mode == '3D':
        z_cropped_stack = crop_stack_z(
            stack=stack,
            grid=grid,
            z_range=z_range,
            ref_voxel_size=ref_voxel_size
        )
        output = apply_affine_crop_simple(z_cropped_stack, registered_rect, order=3)

    return output
