from typing import Literal, Dict, List
import argparse
import numpy as np
import napari
import math
from pathlib import Path
from liffile import LifFile
from gui_device_segmentation import DeviceSegmentationApp
import torch
from warnings import filterwarnings

# Relative imports

from models import Pix2Pix, load_segmentation_model, predict_mask_ortho, process_vessel_mask
from utils import scale, resize_dask
from skeletonisation import clean_and_analyse

filterwarnings('ignore')

class VascuMap:
    def __init__(
        self,
        pix2pix_model_path: str = r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt",
        unet_model_path: str = r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\best_full.pth",
        use_device_segmentation_app: bool = True,
        image_source_path: str | None = None,
        image_index: int = 0,
        device_width_um: float = 35.0,
        mask_central_region: bool = False,
    ) -> None:
        """Initialize the VascuMap workflow container.

        Optionally launches the napari-based device segmentation UI and stores
        the resulting cropped stack plus metadata when the viewer is closed.

        Args:
            use_device_segmentation_app: If ``True``, start
                :class:`DeviceSegmentationApp` and collect outputs. If ``False``,
                initialization for the non-GUI path is not yet implemented.

        Raises:
            NotImplementedError: If ``use_device_segmentation_app`` is ``False``.
        """
        
        self.model_p2p = Pix2Pix(model_path=pix2pix_model_path)
        self.model_unet = load_segmentation_model(unet_model_path)
        self.unet_model_path = unet_model_path
        self.app = None
        self.use_device_segmentation_app = bool(use_device_segmentation_app)
        self.cropped_stack = None
        self.device_width_um = None
        self.pixel_size_um = None
        self.z_votes = None
        self.image_name = None
        self.mask_central_region_enabled = False
        self.cropped_organoid_mask_xy = None
        self.initial_z_range = None
        self._pre_z_start_global = None
        self._pre_z_um = None
        
        if self.use_device_segmentation_app:
            self.app = DeviceSegmentationApp()
            napari.run()
            outputs = self.app.get_cropped_outputs()
            self.cropped_stack, self.device_width_um, self.pixel_size_um, self.z_votes, self.image_name = outputs[:5]
            if len(outputs) >= 7:
                self.mask_central_region_enabled = bool(outputs[5])
                self.cropped_organoid_mask_xy = outputs[6]
        else:
            if image_source_path is None:
                raise ValueError("When use_device_segmentation_app is False, image_source_path is required and must point to a .tif/.tiff/.lif file.")

            src = Path(image_source_path)
            if (not src.exists()) or src.suffix.lower() not in (".tif", ".tiff", ".lif"):
                raise ValueError("image_source_path must exist and be a .tif/.tiff/.lif file.")

            self.app = DeviceSegmentationApp(enable_gui=False)
            outputs = self.app.run_automatic(
                image_source=src,
                image_index=int(image_index),
                device_width_um=float(device_width_um),
                mask_central_region=bool(mask_central_region),
            )
            self.cropped_stack, self.device_width_um, self.pixel_size_um, self.z_votes, self.image_name = outputs[:5]
            if len(outputs) >= 7:
                self.mask_central_region_enabled = bool(outputs[5])
                self.cropped_organoid_mask_xy = outputs[6]
       
    def preprocess(self, cropped_stack: np.ndarray = None, pixel_size_um: Dict = None, z_votes: List = None, min_span_um: float = 160.0
    ) -> np.array:
        """Select a focus-informed z-range and crop the stack along z.

        The function builds a normalized z-vote map, finds the best contiguous
        span with at least one vote, expands that span (downward first) to meet
        a minimum physical z-coverage, and returns both the selected votes and
        the cropped z-slice of the input stack.

        Args:
            cropped_stack: Input cropped image stack with z as the first axis.
            pixel_size_um: Pixel-size metadata dictionary. Must include
                ``"z_um"`` as a positive value in micrometers.
            z_votes: Vote counts by z-plane. Keys can be integers or strings
                like ``"z12"``.
            min_span_um: Minimum z-span to enforce, in micrometers.

        Returns:
            np.array: A tuple containing:
                1) Dict of votes within the selected z-range, preserving key
                format style from input.
                2) Cropped stack slice over the selected z-range.

        Raises:
            ValueError: If ``pixel_size_um["z_um"]`` is missing, non-finite, or
                not positive.
        """

        if not cropped_stack:
            cropped_stack = self.cropped_stack
            
        if not pixel_size_um:
            pixel_size_um = self.pixel_size_um
            
        if not z_votes:
            z_votes = self.z_votes
        print(f"Initial z votes {self.z_votes}")

        z_step = float(pixel_size_um["z_um"])
        if not np.isfinite(z_step) or z_step <= 0:
            raise ValueError("z_step_um must be positive (um).")

        vote_map: dict[int, int] = {}
        for key, value in z_votes.items():
            if isinstance(key, (int, np.integer)):
                zi = int(key)
            else:
                txt = str(key).strip().lower()
                if txt.startswith("z"):
                    txt = txt[1:]
                zi = int(txt)
            vote_map[zi] = int(value)

        if len(vote_map) == 0:
            return None

        z_min_all, z_max_all = int(min(vote_map.keys())), int(max(vote_map.keys()))
        votes = {z: int(vote_map.get(z, 0)) for z in range(z_min_all, z_max_all + 1)}
        
        min_slices = max(1, math.ceil(float(min_span_um) / z_step))
        zs = sorted(votes)
        mask = [votes[z] >= 1 for z in zs]

        best = (None, None, 0)
        start = None

        for i, m in enumerate(mask):
            if m and start is None:
                start = i
            elif (not m) and (start is not None):
                if i - start > best[2]:
                    best = (zs[start], zs[i - 1], i - start)
                start = None

        if start is not None:
            if len(mask) - start > best[2]:
                best = (zs[start], zs[-1], len(mask) - start)

        z_start, z_end, length = best
        if z_start is None or z_end is None:
            return None

        if length < min_slices:
            needed = min_slices - length
            z_min, z_max = zs[0], zs[-1]

            extend_down = min(needed, z_start - z_min)
            z_start -= extend_down
            needed -= extend_down

            if needed > 0:
                extend_up = min(needed, z_max - z_end)
                z_end += extend_up

        final_length = z_end - z_start + 1

        self.initial_z_range, self.cropped_stack = (
            {int(i): int(votes.get(i, 0)) for i in range(int(z_start), int(z_end) + 1)},
            cropped_stack[int(z_start):int(z_end) + 1, :, :]
        )
        self._pre_z_start_global = int(z_start)
        self._pre_z_um = float(pixel_size_um["z_um"])
        self.cropped_stack = resize_dask(self.cropped_stack, [pixel_size_um["z_um"] / 5.0, pixel_size_um["y_um"] / 2.0, pixel_size_um["x_um"] / 2.0])
        
        print(f"First cropping to z: {self.initial_z_range}")
        print(f"Stack width {(z_step * final_length)}")

    def model_inference(self, cropped_stack: np.ndarray = None, device: Literal['cuda', 'cpu'] = 'cuda', 
    ) -> None:
        """
        """
        
        if not cropped_stack:
            cropped_stack = self.cropped_stack

        if self.mask_central_region_enabled and self.cropped_organoid_mask_xy is not None:
            mask_xy = np.asarray(self.cropped_organoid_mask_xy, dtype=np.float32)
            if mask_xy.ndim == 2 and mask_xy.size > 0:
                target_h = int(cropped_stack.shape[1])
                target_w = int(cropped_stack.shape[2])
                src_h = int(mask_xy.shape[0])
                src_w = int(mask_xy.shape[1])
                if src_h > 0 and src_w > 0:
                    scale_h = float(target_h) / float(src_h)
                    scale_w = float(target_w) / float(src_w)
                    resized_mask = resize_dask(mask_xy[None, :, :], [1.0, scale_h, scale_w])[0]
                    resized_mask = np.asarray(resized_mask > 0.5, dtype=bool)

                    if resized_mask.shape[0] < target_h:
                        pad_h = target_h - resized_mask.shape[0]
                        resized_mask = np.pad(resized_mask, ((0, pad_h), (0, 0)), mode="constant", constant_values=False)
                    if resized_mask.shape[1] < target_w:
                        pad_w = target_w - resized_mask.shape[1]
                        resized_mask = np.pad(resized_mask, ((0, 0), (0, pad_w)), mode="constant", constant_values=False)
                    resized_mask = resized_mask[:target_h, :target_w]

                    if np.any(resized_mask):
                        fill_source = np.asarray(cropped_stack)
                        valid_region = ~resized_mask
                        for zi in range(fill_source.shape[0]):
                            slice_data = fill_source[zi]
                            if np.any(valid_region):
                                fill_value = float(np.mean(slice_data[valid_region]))
                            else:
                                fill_value = float(np.mean(slice_data))
                            slice_data[resized_mask] = fill_value

        stack_bf = scale(cropped_stack)
        vessel_pred = self.model_p2p.predict(stack_bf, device, n_iter=1)
        self.vessel_pred_iso = resize_dask(vessel_pred, [2.5, 1, 1])

        self.vessel_proba_iso = predict_mask_ortho(self.model_unet, self.vessel_pred_iso, device)
        self.vessel_mask_iso = process_vessel_mask(self.vessel_proba_iso, True) 

        
    def postprocess(self, 
    ) -> None:
        """
        """
        if self.initial_z_range is None or self._pre_z_start_global is None or self._pre_z_um is None:
            return

        final_z_um = 2.0
        zs = sorted(int(z) for z in self.initial_z_range.keys())
        strong_mask = [int(self.initial_z_range[z]) > 2 for z in zs]

        best_start = None
        best_end = None
        best_len = 0
        run_start = None

        for i, is_strong in enumerate(strong_mask):
            if is_strong and run_start is None:
                run_start = i
            elif (not is_strong) and (run_start is not None):
                run_len = i - run_start
                if run_len > best_len:
                    best_len = run_len
                    best_start = run_start
                    best_end = i - 1
                run_start = None

        if run_start is not None:
            run_len = len(strong_mask) - run_start
            if run_len > best_len:
                best_len = run_len
                best_start = run_start
                best_end = len(strong_mask) - 1

        if best_start is None or best_end is None:
            return

        global_z0 = zs[best_start]
        global_z1 = zs[best_end]
        print(f"strong contiguous vote planes {global_z0}-{global_z1}")

        rel_z0 = global_z0 - int(self._pre_z_start_global)
        rel_z1 = global_z1 - int(self._pre_z_start_global)

        um_start = float(rel_z0) * float(self._pre_z_um)
        um_stop = float(rel_z1 + 1) * float(self._pre_z_um)

        z_start_final = int(np.floor(um_start / final_z_um))
        z_stop_final = int(np.ceil(um_stop / final_z_um))

        z_start_final = max(0, z_start_final)
        z_stop_final = min(self.vessel_pred_iso.shape[0], z_stop_final)
        if z_stop_final <= z_start_final:
            return

        run_suffix =  int(np.random.randint(1, 10000))
        name_prefix = self.image_name if self.image_name else "image"
        np.save(f"{name_prefix}_vessel_proba_iso_{run_suffix}.npy", self.vessel_proba_iso)
        np.save(f"{name_prefix}_vessel_pred_iso_{run_suffix}.npy", self.vessel_pred_iso)
        np.save(f"{name_prefix}_vessel_mask_iso_{run_suffix}.npy", self.vessel_mask_iso)
        np.save(f"{name_prefix}_cropped_stack_{run_suffix}.npy", self.cropped_stack)
        current_x_pixels = self.vessel_proba_iso.shape[1]
        current_y_pixels = self.vessel_proba_iso.shape[2]
        pixels_to_remove = int(self.device_width_um)################updated harsher cropping here
        self.vessel_pred_iso = self.vessel_pred_iso[z_start_final:z_stop_final, pixels_to_remove:current_x_pixels-pixels_to_remove, pixels_to_remove:current_y_pixels-pixels_to_remove]
        self.vessel_proba_iso = self.vessel_proba_iso[z_start_final:z_stop_final, pixels_to_remove:current_x_pixels-pixels_to_remove, pixels_to_remove:current_y_pixels-pixels_to_remove]
        self.vessel_mask_iso = self.vessel_mask_iso[z_start_final:z_stop_final, pixels_to_remove:current_x_pixels-pixels_to_remove, pixels_to_remove:current_y_pixels-pixels_to_remove]
        np.save(f"{name_prefix}_vessel_mask_iso_{run_suffix}_cropped.npy", self.vessel_mask_iso)        

        if self.use_device_segmentation_app:
            viewer = napari.Viewer()
            viewer.add_image(self.vessel_proba_iso)
            viewer.add_image(self.vessel_pred_iso)
            viewer.add_labels(self.vessel_mask_iso)
            napari.run()
        
        
    def skeletonisation_and_analysis(
        self,
        voxel_size_um=(2.0, 2.0, 2.0),
        junction_distance_mode='skeleton',
    ) -> None:
        """
        Run skeletonisation and vascular-network analysis on the final mask.
        """
        if self.vessel_mask_iso is None:
            print("No vessel_mask_iso available – skipping analysis.")
            return

        print("Running skeletonisation and analysis...")
        # Build exclusion mask matched to vessel_mask_iso shape if organoid masking was used
        exclusion_mask_xy = None
        if self.mask_central_region_enabled and self.cropped_organoid_mask_xy is not None:
            mask_xy = np.asarray(self.cropped_organoid_mask_xy, dtype=np.float32)
            if mask_xy.ndim == 2 and mask_xy.size > 0:
                target_h = int(self.vessel_mask_iso.shape[1])
                target_w = int(self.vessel_mask_iso.shape[2])
                src_h, src_w = int(mask_xy.shape[0]), int(mask_xy.shape[1])
                if src_h > 0 and src_w > 0:
                    # Resize to pre-trim resolution (same XY as model output)
                    pre_trim_h = target_h + 2 * int(self.device_width_um / 2)
                    pre_trim_w = target_w + 2 * int(self.device_width_um / 2)
                    scale_h = float(pre_trim_h) / float(src_h)
                    scale_w = float(pre_trim_w) / float(src_w)
                    resized = resize_dask(mask_xy[None, :, :], [1.0, scale_h, scale_w])[0]
                    resized = np.asarray(resized > 0.5, dtype=bool)
                    # Apply same device-width trim as postprocess
                    pixels_to_remove = int(self.device_width_um / 2)
                    resized = resized[pixels_to_remove:pixels_to_remove + target_h,
                                      pixels_to_remove:pixels_to_remove + target_w]
                    # Pad/clip to exact target shape
                    if resized.shape[0] < target_h:
                        resized = np.pad(resized, ((0, target_h - resized.shape[0]), (0, 0)),
                                         mode="constant", constant_values=False)
                    if resized.shape[1] < target_w:
                        resized = np.pad(resized, ((0, 0), (0, target_w - resized.shape[1])),
                                         mode="constant", constant_values=False)
                    resized = resized[:target_h, :target_w]
                    if np.any(resized):
                        exclusion_mask_xy = resized
                        print(f"Organoid exclusion mask applied: {int(np.count_nonzero(exclusion_mask_xy))} pixels excluded per z-slice")

        self.analysis_results = clean_and_analyse(
            self.vessel_mask_iso,
            voxel_size_um=voxel_size_um,
            junction_distance_mode=junction_distance_mode,
            exclusion_mask_xy=exclusion_mask_xy,
        )

        name_prefix = self.image_name if self.image_name else "image"
        metrics_df = self.analysis_results['global_metrics_df']
        csv_path = f"{name_prefix}_analysis_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        print(f"Analysis metrics saved to {csv_path}")
        print(metrics_df.to_string(index=False))
        
    def pipeline(
        self,
    ) -> None:
        """
        """

        run_suffix = int(np.random.randint(1, 10000))
        name_prefix = self.image_name if self.image_name else "image"

        if self.app is not None:
            self.app.save_overlay_and_slice_tifs(name_prefix=name_prefix, run_suffix=run_suffix)

        self.preprocess()
        self.model_inference(device="cuda")
        self.postprocess()
        self.skeletonisation_and_analysis()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VascuMap in GUI or no-GUI mode.")
    parser.add_argument("--no-gui", action="store_true", help="Run in no-GUI batch mode.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory with .lif/.tif/.tiff files (required with --no-gui).",
    )
    args = parser.parse_args()

    if not args.no_gui:
        vascumap = VascuMap(use_device_segmentation_app=True)
        vascumap.pipeline()
    else:
        def _should_mask_from_name(p: Path) -> bool:
            return "marina" in p.name.lower()

        if args.image_dir is None:
            raise ValueError("--image-dir is required when --no-gui is set.")

        source_dir = Path(args.image_dir)
        if (not source_dir.exists()) or (not source_dir.is_dir()):
            raise ValueError(f"--image-dir must be an existing directory: {source_dir}")

        image_paths = sorted(
            [p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in (".lif", ".tif", ".tiff")]
        )

        if len(image_paths) == 0:
            raise FileNotFoundError(f"No .lif/.tif/.tiff files found in: {source_dir}")

        print(f"Found {len(image_paths)} files. Running non-GUI pipeline...")
        failures = []
        total_runs = 0
        successes = 0

        for i, image_path in enumerate(image_paths, start=1):
            print(f"[{i}/{len(image_paths)}] Processing: {image_path.name}")
            mask_central_region = _should_mask_from_name(image_path)
            print(f"  -> mask_central_region={mask_central_region} (filename contains 'marina': {mask_central_region})")

            if image_path.suffix.lower() == ".lif":
                try:
                    with LifFile(image_path) as lif:
                        n_images = len(lif.images)
                except Exception as exc:
                    failures.append((str(image_path), f"Could not inspect .lif images: {exc}"))
                    print(f"[FAILED] {image_path.name}: Could not inspect .lif images: {exc}")
                    continue

                for idx in range(n_images):
                    total_runs += 1
                    print(f"  -> LIF image index {idx + 1}/{n_images}")
                    try:
                        vascumap = VascuMap(
                            use_device_segmentation_app=False,
                            image_source_path=str(image_path),
                            image_index=idx,
                            mask_central_region=mask_central_region,
                        )
                        vascumap.image_name = f"{image_path.stem}_img{idx}_{vascumap.image_name if vascumap.image_name else 'image'}"
                        vascumap.pipeline()
                        successes += 1
                    except Exception as exc:
                        failures.append((f"{image_path} (image_index={idx})", str(exc)))
                        print(f"[FAILED] {image_path.name} (image_index={idx}): {exc}")
            else:
                total_runs += 1
                try:
                    vascumap = VascuMap(
                        use_device_segmentation_app=False,
                        image_source_path=str(image_path),
                        mask_central_region=mask_central_region,
                    )
                    vascumap.image_name = f"{image_path.stem}_{vascumap.image_name if vascumap.image_name else 'image'}"
                    vascumap.pipeline()
                    successes += 1
                except Exception as exc:
                    failures.append((str(image_path), str(exc)))
                    print(f"[FAILED] {image_path.name}: {exc}")

        print(f"Batch complete. Total runs: {total_runs}, Success: {successes}, Failed: {len(failures)}")
        if failures:
            print("Failed files:")
            for fp, msg in failures:
                print(f"- {fp} -> {msg}")
