import os
import argparse
import glob
import numpy as np
import tifffile as tif
import matplotlib.pyplot as plt
from pathlib import Path
from aicsimageio import AICSImage

from .data import get_stack_from_lif
from .preprocessing import crop_stack
from .prediction import load_pix2pix, load_segmentation_model, predict_pix2pix, predict_mask, predict_mask_ortho
from .utils import contrast, scale, resize_dask, process_vessel_mask, remove_false_positives

def run(lif_dir_path, ortho=True, bf_index=None, fluo_index=None, crop_profile='dorota', z_range=200, save_all_stacks=False, crop=True, model_p2p_path=None, model_smp_path=None):
    """
    Run the main inference pipeline on a directory of LIF files.

    Steps:
    1. Load LIF file and extract 3D stack.
    2. (Optional) Crop and focus the stack using a reference profile.
    3. (Optional) Translate Brightfield to Fluorescence using Pix2Pix.
    4. Segment the fluorescence (or predicted) volume using a 2D/2.5D U-Net.
    5. Save results (masks, probability maps, visualizations).

    Args:
        lif_dir_path (str): Directory containing .lif files.
        ortho (bool, optional): Use orthogonal plane inference averaging. Defaults to True.
        bf_index (int, optional): Index of Brightfield channel. Defaults to None (auto-detect).
        fluo_index (int, optional): Index of Fluorescence channel. Defaults to None (auto-detect).
        crop_profile (str, optional): Name of crop profile for registration. Defaults to 'dorota'.
        z_range (int, optional): Z-range for cropping. Defaults to 200.
        save_all_stacks (bool, optional): Save intermediate stacks (BF, Pred, Proba). Defaults to False.
        crop (bool, optional): Whether to apply cropping. Defaults to True.
        model_p2p_path (str, optional): Path to Pix2Pix checkpoint. Defaults to None.
        model_smp_path (str, optional): Path to Segmentation checkpoint. Defaults to None.
    """

    lif_files = glob.glob(os.path.join(lif_dir_path, '*.lif'))
    
    if not lif_files:
        print(f"No .lif files found in {lif_dir_path}")
        return

    # Validate model paths
    if model_p2p_path is None and fluo_index is None:
        raise ValueError("model_p2p_path is required when fluo_index is not provided (need Pix2Pix for BF->Fluo translation)")
    if model_smp_path is None:
        raise ValueError("model_smp_path is required for segmentation")

    device = 'cuda' if np.any([os.environ.get('CUDA_VISIBLE_DEVICES'), True]) else 'cpu' # Simple check, torch.cuda.is_available() used inside
    
    try:
        model_p2p = load_pix2pix(model_p2p_path).to('cpu')
        model_smp = load_segmentation_model(model_smp_path).to('cpu')
        print('Models loaded')
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    ref_voxel_size = [5, 2, 2]
    z_anisotropy = ref_voxel_size[0] / np.mean([ref_voxel_size[1], ref_voxel_size[2]])

    for lif_file in lif_files:

        n_images = 0
        try:
             n_images = AICSImage(lif_file).scenes
             # readlif might be better for count if AICSImage loads all
             from readlif.reader import LifFile
             n_images = LifFile(lif_file).num_images
        except Exception as e:
            print(f"Error reading file {lif_file}: {e}")
            continue

        for stack_index in range(n_images):
            try:
                from readlif.reader import LifFile # Import here to ensure fresh handle if needed
                reader = LifFile(lif_file)
                img_obj = reader.get_image(stack_index)
                filename = Path(reader.filename).stem + '_' + img_obj.name
                filename = filename.replace(' ', '_').replace('/', '_')
                
                dims = list(img_obj.dims_n.values())

                # if filename exists, skip loop
                if os.path.exists(os.path.join(lif_dir_path, filename + '_vessel_mask.tif')):
                    print(f'{filename} already processed. Skipped')
                    continue

                if len(dims) != 3:
                    print(f'{filename} is not a 3D stack. Skipped')
                    continue

                print(f'Analyzing {filename} ...')

                if bf_index is None: 
                    img_aics = AICSImage(lif_file)
                    channel_names = img_aics.channel_names
                    # Heuristic to find BF channel
                    candidates = [i for i, name in enumerate(channel_names) if 'BF' in name or 'Gray' in name]
                    current_bf_index = candidates[0] if candidates else 0 # Default to 0 if not found
                else:
                    current_bf_index = bf_index

                stack, voxel_size, [nz, ny, nx] = get_stack_from_lif(lif_file, stack_index=stack_index, ref_voxel_size=ref_voxel_size)

                if crop == True:
                    try:
                        stack = crop_stack(stack, crop_profile, current_bf_index, mode='3D', z_range=z_range, ref_voxel_size=ref_voxel_size, lif_path_root=lif_dir_path, filename=filename)
                    except Exception as e:
                        print(f"Cropping failed for {filename}: {e}. Proceeding without crop or skipping.")
                        # Decide whether to skip or continue. Assuming skip if crop was requested but failed.
                        continue

                stack_bf = scale(stack[..., current_bf_index]) 
                vessel_bf_iso = resize_dask(stack_bf, [z_anisotropy, 1, 1])
                
                if fluo_index is not None:
                    stack_fluo = scale(stack[..., int(stack.shape[-1]-current_bf_index-1)]) # Logic from original: assumes 2 channels and takes the other?
                    vessel_fluo_iso = resize_dask(stack_fluo , [z_anisotropy, 1, 1])
                    vessel_pred_iso = np.copy(vessel_fluo_iso)

                else:
                    if stack.shape[-1]>1:
                        # Use the other channel as fluo ground truth visualization if available
                        other_idx = 1 if current_bf_index == 0 else 0
                        if other_idx < stack.shape[-1]:
                             stack_fluo = scale(stack[..., other_idx])
                             vessel_fluo_iso = resize_dask(stack_fluo , [z_anisotropy, 1, 1])
                    
                    vessel_pred = predict_pix2pix(model_p2p, stack_bf, device, n_iter=1)
                    vessel_pred_iso = resize_dask(vessel_pred, [z_anisotropy, 1, 1])

                if ortho:
                    vessel_proba_iso = predict_mask_ortho(model_smp, vessel_pred_iso, device)
                    vessel_mask_iso = process_vessel_mask(vessel_proba_iso, ortho)

                else:
                    vessel_pred_clip = scale(np.clip(vessel_pred, 0.05, 1))
                    vessel_proba = predict_mask(model_smp, vessel_pred_clip, device)
                    vessel_mask = process_vessel_mask(vessel_proba, ortho).astype(np.float32)
                    
                    if save_all_stacks:
                        vessel_pred_iso = resize_dask(vessel_pred_clip, [z_anisotropy, 1, 1])
                    vessel_proba_iso = resize_dask(vessel_proba, [z_anisotropy, 1, 1])
                    vessel_mask_iso = resize_dask(vessel_mask, [z_anisotropy, 1, 1]) 

                # Visualization
                plt.figure(figsize=(30,15))

                ax = plt.subplot(131)
                if 'vessel_fluo_iso' in locals():
                    plt.imshow(contrast(np.sum(vessel_fluo_iso, axis=0), 0.1, 99.9), cmap='magma')
                else:
                    plt.imshow(contrast(np.sum(vessel_bf_iso, axis=0), 0.1, 99.9), cmap='gray') # Fallback

                plt.subplot(132, sharex=ax, sharey=ax)
                plt.imshow(np.sum(vessel_proba_iso, axis=0), cmap='magma')
                plt.subplot(133, sharex=ax, sharey=ax)
                plt.imshow(np.sum(vessel_mask_iso, axis=0), cmap='magma')
                plt.tight_layout()
                plt.suptitle(filename, fontsize=40)

                plt.savefig(os.path.join(lif_dir_path, filename + '_overview.png'), dpi=200)
                plt.close('all')

                if save_all_stacks:
                    if 'vessel_fluo_iso' in locals(): 
                        tif.imwrite(os.path.join(lif_dir_path, filename + '_vessel_fluo.tif'), vessel_fluo_iso.astype(np.float32))
                    tif.imwrite(os.path.join(lif_dir_path, filename + '_vessel_bf.tif'), vessel_bf_iso.astype(np.float32))
                    tif.imwrite(os.path.join(lif_dir_path, filename + '_vessel_pred.tif'), vessel_pred_iso.astype(np.float32))
                    tif.imwrite(os.path.join(lif_dir_path, filename + '_vessel_proba.tif'), vessel_proba_iso.astype(np.float32))

                tif.imwrite(os.path.join(lif_dir_path, filename + '_vessel_mask.tif'), vessel_mask_iso.astype(np.float32))

            except Exception as e:
                 print(f"Error processing stack {stack_index} in {lif_file}: {e}")
                 continue

        print('======================================================')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run Vascumap3D.")

    parser.add_argument("--lif_dir_path", type=str, required=True,
                        help="Path to the directory containing lif files.")
    parser.add_argument("--model_p2p_path", type=str, required=False, help="Path to Pix2Pix model checkpoint")
    parser.add_argument("--model_smp_path", type=str, required=False, help="Path to Segmentation model checkpoint")

    args = parser.parse_args()

    run(lif_dir_path=args.lif_dir_path, model_p2p_path=args.model_p2p_path, model_smp_path=args.model_smp_path)
