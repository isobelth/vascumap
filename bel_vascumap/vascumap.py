from typing import Literal, Dict, List
import numpy as np
import pandas as pd
import math
from pathlib import Path
from device_segmentation import DeviceSegmentationApp
from warnings import filterwarnings
from models import Pix2Pix, load_segmentation_model, predict_mask_ortho, process_vessel_mask
from utils import scale, resize_dask
from skeletonisation import clean_and_analyse, trim_segmentation, generate_skeleton_overview_plot, build_internal_pore_label_volumes, graph2image
import pickle

filterwarnings('ignore')

class VascuMap:
    def __init__(
        self,
        pix2pix_model_path: str = r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt",
        unet_model_path: str = r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\best_full.pth",
        curated_outputs: dict | None = None,
        image_source_path: str | None = None,
        image_index: int = 0,
        device_width_um: float = 35.0,
        hough_line_length: int = 400,
        mask_central_region = False,
        brightfield_channel: int = 0,
        model_p2p=None,
        model_unet=None,
        failure_output_dir: str | None = None,
    ) -> None:
        """Initialize the VascuMap workflow container.

        There are two modes of operation:

        1. **Curated** — pass ``curated_outputs`` (a dict produced by
           :class:`CurationApp`) to skip device segmentation entirely.
        2. **Headless automatic** — pass ``image_source_path`` with a
           ``.tif/.tiff/.lif`` file to run device segmentation automatically.

        For interactive GUI curation, use :class:`gui_region_detection.CurationApp`
        to curate images first, then feed the results here via
        ``curated_outputs``.

        Args:
            curated_outputs: Dict from ``CuratedJob.finalised_outputs`` —
                when provided, all device-segmentation steps are skipped.
            image_source_path: Path to the source image (headless mode).
            model_p2p: Pre-loaded Pix2Pix model. If ``None``, loaded from
                ``pix2pix_model_path``.
            model_unet: Pre-loaded UNet segmentation model. If ``None``, loaded
                from ``unet_model_path``.
        """
        
        self.model_p2p = model_p2p if model_p2p is not None else Pix2Pix(model_path=pix2pix_model_path)
        self.model_unet = model_unet if model_unet is not None else load_segmentation_model(unet_model_path)
        self.unet_model_path = unet_model_path
        self.app = None
        self.cropped_stack = None
        self.device_width_um = None
        self.pixel_size_um = None
        self.z_votes = None
        self.image_name = None
        self.mask_central_region_enabled = False
        self.cropped_organoid_mask_xy = None
        self.initial_z_range = None
        self.pre_z_start_global = None
        self.pre_z_um = None
        self.z_start_final = None
        self.z_stop_final = None
        self.pixels_to_remove = None
        self.exclusion_mask_xy_aligned = None
        self.resized_organoid_mask_pre_trim = None
        self.image_source_path = image_source_path
        self.image_index = int(image_index) if image_index is not None else 0
        self.hough_line_length = int(hough_line_length)
        
        if curated_outputs is not None:
            # Pre-curated path: populate fields directly from CurationApp output
            self.cropped_stack = curated_outputs["cropped_stack"]
            self.device_width_um = curated_outputs["device_width_um"]
            self.pixel_size_um = curated_outputs["pixel_size_um"]
            self.z_votes = curated_outputs["z_votes"]
            self.image_name = curated_outputs["image_name"]
            self.mask_central_region_enabled = bool(curated_outputs.get("mask_central_region_enabled", False))
            self.cropped_organoid_mask_xy = curated_outputs.get("cropped_organoid_mask_xy")
            self.image_source_path = curated_outputs.get("source_path", image_source_path)
            self.image_index = int(curated_outputs.get("image_index", image_index))
        else:
            if image_source_path is None:
                raise ValueError("Either curated_outputs or image_source_path must be provided.")

            src = Path(image_source_path)
            if (not src.exists()) or src.suffix.lower() not in (".tif", ".tiff", ".lif"):
                raise ValueError("image_source_path must exist and be a .tif/.tiff/.lif file.")

            self.app = DeviceSegmentationApp(enable_gui=False, line_length=self.hough_line_length)
            self.app.channel = int(brightfield_channel)
            outputs = self.app.run_automatic(
                image_source=src,
                image_index=int(image_index),
                device_width_um=float(device_width_um),
                mask_central_region=mask_central_region,
                failure_output_dir=Path(failure_output_dir) if failure_output_dir else None,
            )
            self.cropped_stack, self.device_width_um, self.pixel_size_um, self.z_votes, self.image_name = outputs[:5]
            if len(outputs) >= 7:
                self.mask_central_region_enabled = bool(outputs[5])
                self.cropped_organoid_mask_xy = outputs[6]
       
    def preprocess(self, cropped_stack: np.ndarray = None, pixel_size_um: Dict = None, z_votes: List = None, min_span_um: float = 160.0) -> np.array:
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

        if z_votes is None or len(z_votes) == 0:
            print("No z-vote data available – skipping preprocess.")
            return None

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

        self.initial_z_range = {int(i): int(votes.get(i, 0)) for i in range(int(z_start), int(z_end) + 1)}
        self.cropped_stack = cropped_stack[int(z_start):int(z_end) + 1, :, :]
        self.pre_z_start_global = int(z_start)
        self.pre_z_um = float(pixel_size_um["z_um"])
        self.cropped_stack = resize_dask(self.cropped_stack, [pixel_size_um["z_um"] / 5.0, pixel_size_um["y_um"] / 2.0, pixel_size_um["x_um"] / 2.0])
        
        print(f"First cropping to z: {self.initial_z_range}")
        print(f"Stack width {(z_step * final_length)}")

    def model_inference(self, cropped_stack: np.ndarray = None, device: Literal['cuda', 'cpu'] = 'cuda') -> None:
        """Run Pix2Pix vessel translation and UNet segmentation on the cropped stack."""
        
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

                    self.resized_organoid_mask_pre_trim = resized_mask

                    if np.any(resized_mask):
                        # A2: vectorised fill — compute per-slice means in one shot instead of a Python loop
                        fill_source = np.asarray(cropped_stack)
                        valid_region = ~resized_mask
                        if np.any(valid_region):
                            fill_values = np.mean(fill_source[:, valid_region], axis=1)  # (Z,)
                        else:
                            fill_values = np.mean(fill_source, axis=(1, 2))  # (Z,)
                        fill_source[:, resized_mask] = fill_values[:, None]

        stack_bf = scale(cropped_stack)
        vessel_pred = self.model_p2p.predict(stack_bf, device, n_iter=1)
        self.vessel_pred_iso = resize_dask(vessel_pred, [2.5, 1, 1])

        self.vessel_proba_iso = predict_mask_ortho(self.model_unet, self.vessel_pred_iso, device)
        self.vessel_mask_iso = process_vessel_mask(self.vessel_proba_iso, True) 

        
    def postprocess(self) -> None:
        """Map the best contiguous z-vote span to isotropic coordinates and crop all volumes."""
        if self.initial_z_range is None or self.pre_z_start_global is None or self.pre_z_um is None:
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

        rel_z0 = global_z0 - int(self.pre_z_start_global)
        rel_z1 = global_z1 - int(self.pre_z_start_global)

        um_start = float(rel_z0) * float(self.pre_z_um)
        um_stop = float(rel_z1 + 1) * float(self.pre_z_um)

        z_start_final = int(np.floor(um_start / final_z_um))
        z_stop_final = int(np.ceil(um_stop / final_z_um))

        z_start_final = max(0, z_start_final)
        z_stop_final = min(self.vessel_pred_iso.shape[0], z_stop_final)
        if z_stop_final <= z_start_final:
            return

        current_x_pixels = self.vessel_proba_iso.shape[1]
        current_y_pixels = self.vessel_proba_iso.shape[2]
        pixels_to_remove = int(self.device_width_um)  # harsher cropping
        self.z_start_final = z_start_final
        self.z_stop_final = z_stop_final
        self.pixels_to_remove = pixels_to_remove
        self.vessel_pred_iso = self.vessel_pred_iso[z_start_final:z_stop_final, pixels_to_remove:current_x_pixels-pixels_to_remove, pixels_to_remove:current_y_pixels-pixels_to_remove]
        self.vessel_proba_iso = self.vessel_proba_iso[z_start_final:z_stop_final, pixels_to_remove:current_x_pixels-pixels_to_remove, pixels_to_remove:current_y_pixels-pixels_to_remove]
        self.vessel_mask_iso = self.vessel_mask_iso[z_start_final:z_stop_final, pixels_to_remove:current_x_pixels-pixels_to_remove, pixels_to_remove:current_y_pixels-pixels_to_remove]
        
        
    def skeletonisation_and_analysis(self, voxel_size_um=(2.0, 2.0, 2.0), junction_distance_mode='skeleton') -> None:
        """Run skeletonisation and vascular-network analysis on the final vessel mask."""
        if self.vessel_mask_iso is None:
            print("No vessel_mask_iso available – skipping analysis.")
            return

        print("Running skeletonisation and analysis...")
        # Build exclusion mask matched to vessel_mask_iso shape if organoid masking was used
        exclusion_mask_xy = None
        if self.mask_central_region_enabled and self.cropped_organoid_mask_xy is not None:
            target_h = int(self.vessel_mask_iso.shape[1])
            target_w = int(self.vessel_mask_iso.shape[2])
            ptr = self.pixels_to_remove if self.pixels_to_remove is not None else int(self.device_width_um)

            if self.resized_organoid_mask_pre_trim is not None:
                resized = self.resized_organoid_mask_pre_trim[ptr:ptr + target_h, ptr:ptr + target_w]
            else:
                # Fallback: compute from scratch (e.g. if model_inference was skipped)
                mask_xy = np.asarray(self.cropped_organoid_mask_xy, dtype=np.float32)
                if mask_xy.ndim == 2 and mask_xy.size > 0:
                    src_h, src_w = int(mask_xy.shape[0]), int(mask_xy.shape[1])
                    if src_h > 0 and src_w > 0:
                        pre_trim_h = target_h + 2 * ptr
                        pre_trim_w = target_w + 2 * ptr
                        scale_h = float(pre_trim_h) / float(src_h)
                        scale_w = float(pre_trim_w) / float(src_w)
                        resized = resize_dask(mask_xy[None, :, :], [1.0, scale_h, scale_w])[0]
                        resized = np.asarray(resized > 0.5, dtype=bool)
                        resized = resized[ptr:ptr + target_h, ptr:ptr + target_w]
                    else:
                        resized = np.zeros((target_h, target_w), dtype=bool)
                else:
                    resized = np.zeros((target_h, target_w), dtype=bool)

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

        # Zero out reconstructed vasculature inside the organoid region
        if exclusion_mask_xy is not None:
            self.vessel_mask_iso[:, exclusion_mask_xy] = 0

        self.exclusion_mask_xy_aligned = exclusion_mask_xy
        self.analysis_results = clean_and_analyse(
            self.vessel_mask_iso,
            voxel_size_um=voxel_size_um,
            junction_distance_mode=junction_distance_mode,
            exclusion_mask_xy=exclusion_mask_xy,
        )
        print(self.analysis_results['global_metrics_df'].to_string(index=False))
        
    def pipeline(self, output_dir: str | Path | None = None, save_all_interim: bool = False) -> None:
        """Run the full VascuMap pipeline and save outputs.

        Args:
            output_dir: Directory for all saved files. If ``None``, saves to cwd.
            save_all_interim: When ``True``, also saves the extra volumes needed
                for the full napari visualisation (holes, pore labels, pore
                distances, full-graph skeleton, clean-graph skeleton, and graph
                node coordinates).
        """
        name_prefix = self.image_name if self.image_name else "image"
        out = Path(output_dir) if output_dir is not None else Path.cwd()
        out.mkdir(parents=True, exist_ok=True)

        # ── 2-D device overlay ────────────────────────────────────────────
        if self.app is not None:
            self.app.save_overlay_and_slice_tifs(
                name_prefix=name_prefix, run_suffix=0, output_dir=out,
            )

        # ── Organoid size guard ───────────────────────────────────────────
        if self.mask_central_region_enabled and self.cropped_organoid_mask_xy is not None:
            organoid_mask = np.asarray(self.cropped_organoid_mask_xy, dtype=bool)
            if organoid_mask.size > 0:
                organoid_fraction = np.count_nonzero(organoid_mask) / organoid_mask.size
                if organoid_fraction > 0.40:
                    print(
                        f"  ⚠ Skipping {name_prefix}: organoid covers "
                        f"{organoid_fraction:.1%} of the image (>40% threshold). "
                        f"Device segmentation outputs saved."
                    )
                    return

        # ── Stage 1: z-selection + isotropic resize ──────────────────────
        result = self.preprocess()
        if result is None and self.cropped_stack is None:
            print(f"  ⚠ Skipping {name_prefix}: no valid z-range / cropped stack.")
            return

        # ── Stage 2: Translation + segmentation ──────────────────────────
        self.model_inference(device="cuda")
        self.postprocess()

        if self.z_start_final is None:
            print(f"  ⚠ Skipping {name_prefix}: postprocess found no strong vote planes.")
            return

        # ── Trim over-segmented edge slices ──────────────────────────────
        orig_z = self.vessel_mask_iso.shape[0]
        trimmed, trim_start, trim_stop = trim_segmentation(self.vessel_mask_iso)
        if trimmed.shape[0] == 0:
            # Every z-slice exceeded the fill threshold — nothing left
            slice_fill = self.vessel_mask_iso.astype(bool).mean(axis=(1, 2))
            msg_lines = [
                f"No vasculature retained for '{name_prefix}'.",
                "",
                "All {0} z-slices were trimmed because their segmentation fill".format(orig_z),
                "fraction exceeded the 75% threshold (too much vasculature).",
                "",
                "Per-slice fill fractions:",
            ]
            for i, frac in enumerate(slice_fill):
                msg_lines.append(f"  slice {i:4d}: {frac:.4f}")
            msg_lines.append("")
            msg_lines.append("Possible causes:")
            msg_lines.append("  - The model over-segmented this image (e.g. noisy input).")
            msg_lines.append("  - The device region was not cropped tightly enough.")
            msg_lines.append("  - The z-range selection included out-of-focus planes.")
            debug_txt = out / f"{name_prefix}_trim_failure_debug.txt"
            debug_txt.write_text("\n".join(msg_lines), encoding="utf-8")
            print(f"  ⚠ Skipping {name_prefix}: all z-slices trimmed (too much vasculature). "
                  f"Debug info → {debug_txt.name}")

            # Save interim outputs even on trim failure so the user can
            # inspect the over-segmented result in napari.
            if save_all_interim:
                z0, z1 = self.z_start_final, self.z_stop_final
                ptr = self.pixels_to_remove
                if z0 is not None and z1 is not None and ptr is not None:
                    cropped_stack_iso = resize_dask(self.cropped_stack, [2.5, 1, 1])
                    H, W = cropped_stack_iso.shape[1], cropped_stack_iso.shape[2]
                    cropped_stack_aligned = cropped_stack_iso[z0:z1, ptr:H - ptr, ptr:W - ptr]
                    np.save(str(out / f"{name_prefix}_cropped_stack_aligned.npy"),
                            cropped_stack_aligned)
                np.save(str(out / f"{name_prefix}_vessel_translation_aligned.npy"),
                        self.vessel_pred_iso)
                np.save(str(out / f"{name_prefix}_vessel_mask.npy"),
                        self.vessel_mask_iso)
                print(f"  Saved interim outputs (pre-trim) for napari debugging")

            return
        if trim_start > 0 or trim_stop < orig_z:
            old_z0 = self.z_start_final
            self.vessel_mask_iso = trimmed
            self.vessel_pred_iso = self.vessel_pred_iso[trim_start:trim_stop]
            self.z_start_final = old_z0 + trim_start
            self.z_stop_final = old_z0 + trim_stop
            print(f"  Trimmed {trim_start} top / {orig_z - trim_stop} bottom over-segmented z-slices")

        # ── Stage 3: Skeletonisation + analysis ──────────────────────────
        self.skeletonisation_and_analysis()

        # ── Skeleton overview plot ────────────────────────────────────────
        app_debug = getattr(self.app, 'last_segment_debug', None) or {}
        generate_skeleton_overview_plot(
            self.vessel_mask_iso,
            self.analysis_results,
            title=name_prefix,
            save_path=str(out / f"{name_prefix}_skeleton_overview.png"),
            brightfield_stack=self.cropped_stack,
            organoid_mask_xy=self.cropped_organoid_mask_xy,
            brightfield_full=getattr(self.app, 'last_image', None),
            device_corners_xy=app_debug.get('final_corners'),
            organoid_mask_full_xy=getattr(self.app, 'last_organoid_region', None),
        )
        print(f"  Skeleton overview → {name_prefix}_skeleton_overview.png")

        # ── Metrics CSV (always saved) ────────────────────────────────────
        ar = self.analysis_results
        metrics_df = ar['global_metrics_df'].copy()
        # Prepend identification columns
        src = Path(self.image_source_path) if self.image_source_path else None
        metrics_df.insert(0, 'image_name', name_prefix)
        metrics_df.insert(1, 'source_file', src.name if src else '')
        metrics_df.insert(2, 'image_index', int(self.image_index))
        metrics_df.to_csv(str(out / f"{name_prefix}_analysis_metrics.csv"), index=False)
        print(f"  Metrics → {name_prefix}_analysis_metrics.csv")

        # ── All morphological parameters CSV (always saved) ──────────────
        all_params_df = ar.get('all_morphological_params_df', pd.DataFrame()).copy()
        if all_params_df is not None and not all_params_df.empty:
            all_params_df.insert(0, 'image_name', name_prefix)
            all_params_df.insert(1, 'source_file', src.name if src else '')
            all_params_df.insert(2, 'image_index', int(self.image_index))
        else:
            all_params_df = pd.DataFrame({'image_name': [name_prefix],
                                          'source_file': [src.name if src else ''],
                                          'image_index': [int(self.image_index)]})
        all_params_df.to_csv(str(out / f"{name_prefix}_all_morphological_params.csv"), index=False)
        print(f"  All params → {name_prefix}_all_morphological_params.csv")

        # ── Branch-level metrics CSV (always saved) ──────────────────────
        branch_df = ar.get('branch_metrics_df', pd.DataFrame()).copy()
        if branch_df is not None and not branch_df.empty:
            branch_df.insert(0, 'image_name', name_prefix)
            branch_df.insert(1, 'source_file', src.name if src else '')
            branch_df.insert(2, 'image_index', int(self.image_index))
            branch_df.to_csv(str(out / f"{name_prefix}_branch_metrics.csv"), index=False)
            print(f"  Branch metrics → {name_prefix}_branch_metrics.csv")
        else:
            # still write an empty schema-consistent file for downstream joins
            branch_df = branch_df if isinstance(branch_df, pd.DataFrame) else pd.DataFrame()
            branch_df.insert(0, 'image_name', name_prefix)
            branch_df.insert(1, 'source_file', src.name if src else '')
            branch_df.insert(2, 'image_index', int(self.image_index))
            branch_df.to_csv(str(out / f"{name_prefix}_branch_metrics.csv"), index=False)
            print(f"  Branch metrics → {name_prefix}_branch_metrics.csv (empty)")

        # ── Extra outputs for full napari visualisation ───────────────────
        if save_all_interim:

            # ── Aligned cropped stack (2 µm iso) ─────────────────────────
            z0, z1, ptr = self.z_start_final, self.z_stop_final, self.pixels_to_remove
            if z0 is not None and z1 is not None and ptr is not None:
                cropped_stack_iso = resize_dask(self.cropped_stack, [2.5, 1, 1])
                H, W = cropped_stack_iso.shape[1], cropped_stack_iso.shape[2]
                cropped_stack_aligned = cropped_stack_iso[z0:z1, ptr:H - ptr, ptr:W - ptr]
                np.save(str(out / f"{name_prefix}_cropped_stack_aligned.npy"), cropped_stack_aligned)
                print(f"  Aligned 3-D shape: {cropped_stack_aligned.shape}  (2 µm iso)")

            np.save(str(out / f"{name_prefix}_vessel_translation_aligned.npy"), self.vessel_pred_iso)
            np.save(str(out / f"{name_prefix}_clean_segmentation.npy"), ar["clean_segmentation"])
            np.save(str(out / f"{name_prefix}_skeleton.npy"), ar["skeleton_from_graph"])

            if self.exclusion_mask_xy_aligned is not None:
                np.save(str(out / f"{name_prefix}_organoid_mask.npy"),
                        self.exclusion_mask_xy_aligned.astype(np.uint8))

            seg = ar["clean_segmentation"].astype(bool)
            holes, hole_labels, hole_dist = build_internal_pore_label_volumes(
                seg, voxel_size_um=(2.0, 2.0, 2.0), max_pore_area_fraction_of_slice=0.10,
            )
            full_skel = graph2image(ar["graph"], self.vessel_mask_iso.shape).astype(np.int32)

            np.save(str(out / f"{name_prefix}_holes.npy"), holes)
            np.save(str(out / f"{name_prefix}_hole_labels_per_slice.npy"), hole_labels)
            np.save(str(out / f"{name_prefix}_hole_distance_per_slice_um.npy"), hole_dist)
            np.save(str(out / f"{name_prefix}_full_graph_skeleton.npy"), full_skel)
            np.save(str(out / f"{name_prefix}_vessel_mask.npy"), self.vessel_mask_iso)

            # Graph node coordinates (sprout vs junction) as .npz
            clean_graph = ar["clean_graph"]
            node_ids = list(clean_graph.nodes())
            if node_ids:
                pts = np.array([clean_graph.nodes[n]['pts'] for n in node_ids], dtype=float)
                is_sprout = np.array([bool(clean_graph.nodes[n].get('sprout', False)) for n in node_ids])
                np.savez(str(out / f"{name_prefix}_graph_nodes.npz"),
                         pts=pts, is_sprout=is_sprout)

            with open(str(out / f"{name_prefix}_clean_graph.pkl"), "wb") as f:
                pickle.dump(clean_graph, f)

            print(f"  Saved all interim outputs for napari visualisation")

        print(f"  ✓ Done: {name_prefix}")
