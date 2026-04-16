"""Multi-image curation GUI for VascuMap.

Phase A: discover jobs → auto-detect device ROI + organoid mask for all images
         → user navigates and curates in a single napari session.
Phase B: close viewer → return curated jobs for unattended batch pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import napari
import numpy as np
from magicgui import magicgui
from magicgui.widgets import Container, Label, ComboBox, PushButton
from skimage.filters import sobel

from device_segmentation import (
    DeviceSegmentationApp,
    read_voxel_size_um,
    um_to_xy_pixels,
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CuratedJob:
    """Per-image curation state."""
    source_path: Path
    image_index: int
    image_name: str
    organoid_mode: str          # "dark" / "light" / "off"
    device_width_um: float
    status: str = "pending"     # "pending" / "curated" / "skip" / "failed"
    error_msg: str = ""

    # Auto-detection results (filled during init scan)
    inner_corners: Optional[np.ndarray] = None      # (4, 2) xy
    organoid_mask_xy: Optional[np.ndarray] = None    # 2D bool, full-res focus plane
    pixel_size_um: Optional[dict] = None
    focus_plane: Optional[np.ndarray] = None         # 2D float32
    z_step_um: Optional[float] = None
    y_step_um: Optional[float] = None
    x_step_um: Optional[float] = None
    xy_step_um: Optional[float] = None

    # Internal: stored by DeviceSegmentationApp methods during detection
    _focus_zmap_full: Optional[np.ndarray] = field(default=None, repr=False)
    _focus_downsample: int = 4
    _focus_n_sampling: int = 10
    _focus_patch: int = 50
    _segment_debug: Optional[dict] = field(default=None, repr=False)


def infer_organoid_mode(source_path: Path, image_name: str) -> str:
    """Infer organoid mode from filename heuristic."""
    name = (source_path.name + " " + image_name).lower()
    if "marina" in name and "bead" in name:
        return "light"
    if "marina" in name:
        return "dark"
    return "off"


# ---------------------------------------------------------------------------
# CurationApp
# ---------------------------------------------------------------------------

class CurationApp:
    """Napari-based multi-image curation GUI.

    Usage::

        app = CurationApp(jobs, device_width_um=35.0)
        curated = app.run()   # blocks until viewer is closed
        # curated is a list of CuratedJob with status "curated" or "skip"
    """

    def __init__(
        self,
        jobs: list[tuple],
        device_width_um: float = 35.0,
        brightfield_channel: int = 0,
        default_organoid_mode: str = "infer",
    ):
        """
        Parameters
        ----------
        jobs : list of (source_path, image_index, mask_flag, image_name)
            Same format as ``discover_jobs()`` returns.
        device_width_um : float
            Default device border width in µm.
        brightfield_channel : int
            Channel index for multi-channel stacks.
        default_organoid_mode : str
            "infer" | "dark" | "light" | "off" — default per-job organoid mode.
        """
        self._default_device_width_um = float(device_width_um)
        self._brightfield_channel = int(brightfield_channel)
        self._default_organoid_mode = default_organoid_mode

        # Build CuratedJob list
        self.jobs: list[CuratedJob] = []
        for source_path, image_index, mask_flag, image_name in jobs:
            if default_organoid_mode == "infer":
                mode = infer_organoid_mode(Path(source_path), image_name)
            elif default_organoid_mode == "off":
                mode = "off"
            else:
                mode = str(default_organoid_mode)
            # Override with per-job mask_flag if provided
            if mask_flag is not False and mask_flag is not None:
                mode = str(mask_flag)
            elif mask_flag is False:
                mode = "off"

            self.jobs.append(CuratedJob(
                source_path=Path(source_path),
                image_index=int(image_index),
                image_name=str(image_name),
                organoid_mode=mode,
                device_width_um=float(device_width_um),
            ))

        self._current_idx: int = 0
        self._engine = DeviceSegmentationApp(enable_gui=False)
        self._engine.channel = self._brightfield_channel

        # Napari viewer + layers (created in run())
        self.viewer: Optional[napari.Viewer] = None
        self._image_layer = None
        self._organoid_labels_layer = None
        self._roi_layer = None
        self._roi_outer_layer = None
        self._syncing = False

    # ------------------------------------------------------------------
    # Auto-detection pass
    # ------------------------------------------------------------------

    def _auto_detect_all(self):
        """Run device + organoid auto-detection for every job."""
        print(f"Auto-detecting device regions for {len(self.jobs)} images...")
        for i, job in enumerate(self.jobs):
            print(f"  [{i+1}/{len(self.jobs)}] {job.image_name} ...", end=" ", flush=True)
            try:
                self._auto_detect_single(job)
                job.status = "curated"
                print("OK")
            except Exception as exc:
                job.status = "failed"
                job.error_msg = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {exc}")

    def _auto_detect_single(self, job: CuratedJob):
        """Run detection for one job using the DeviceSegmentationApp engine."""
        eng = self._engine

        # Load images into engine
        eng._list_images(job.source_path)

        # Find the label for this image_index
        selected_label = None
        for lbl, idx in eng._image_choice_map.items():
            if int(idx) == int(job.image_index):
                selected_label = lbl
                break
        if selected_label is None:
            raise ValueError(f"image_index {job.image_index} not found in {job.source_path}")

        # Run segmentation (no GUI)
        eng._segment_and_view(
            image_choice=selected_label,
            focus_downsample=4,
            focus_n_sampling=10,
            focus_patch=50,
            mask_central_region=(job.organoid_mode != "off"),
            clear_layers=True,
        )

        # Check success — if the engine couldn't find corners, cropped_xyz won't exist yet
        # but we can check inner corners from the debug dict
        debug = getattr(eng, '_last_segment_debug', None) or {}
        final_corners = debug.get('final_corners', None)
        if final_corners is None:
            msg = str(getattr(eng.images_output, "value", "Segmentation failed"))
            raise RuntimeError(msg)

        # Extract focus plane
        focus_plane = eng._last_image
        if focus_plane is not None:
            focus_plane = np.asarray(focus_plane, dtype=np.float32)
            if focus_plane.ndim == 3:
                focus_plane = np.mean(focus_plane, axis=-1).astype(np.float32)

        # Extract organoid mask
        organoid_mask = None
        if job.organoid_mode != "off":
            org_region = eng._last_organoid_region
            if org_region is not None:
                organoid_mask = np.asarray(org_region, dtype=bool)

        # Store voxel sizes
        z_um = eng._last_z_step_um
        y_um = eng._last_y_step_um
        x_um = eng._last_x_step_um
        xy_um = eng._last_xy_step_um

        # Populate job
        job.focus_plane = focus_plane
        job.inner_corners = np.asarray(final_corners, dtype=float)
        job.organoid_mask_xy = organoid_mask
        job.z_step_um = z_um
        job.y_step_um = y_um
        job.x_step_um = x_um
        job.xy_step_um = xy_um
        job.pixel_size_um = {"x_um": x_um, "y_um": y_um, "z_um": z_um, "xy_um": xy_um}
        job._focus_zmap_full = eng._last_focus_zmap_full
        job._focus_downsample = eng._last_focus_downsample
        job._focus_n_sampling = eng._last_focus_n_sampling
        job._focus_patch = eng._last_focus_patch
        job._segment_debug = debug

    def _re_detect_organoid(self, job: CuratedJob):
        """Re-run organoid detection with the current mode for one job."""
        if job.focus_plane is None:
            return
        if job.organoid_mode == "off":
            job.organoid_mask_xy = None
            return
        try:
            org_mask = self._engine._mask_out_organoid(
                job.focus_plane, mode=job.organoid_mode,
            )
            job.organoid_mask_xy = np.asarray(org_mask, dtype=bool) if org_mask is not None else None
        except Exception:
            job.organoid_mask_xy = None

    # ------------------------------------------------------------------
    # Viewer setup
    # ------------------------------------------------------------------

    def _build_viewer(self):
        """Create the napari viewer, layers, and widget panel."""
        self.viewer = napari.Viewer(title="VascuMap Curation")

        # Layers — initialised with blank data, populated on first _show_job()
        dummy = np.zeros((100, 100), dtype=np.float32)
        self._image_layer = self.viewer.add_image(dummy, name="focus_plane")
        self._organoid_labels_layer = self.viewer.add_labels(
            np.zeros((100, 100), dtype=np.int32), name="organoid_mask",
        )
        self._organoid_labels_layer.opacity = 0.4

        self._roi_layer = self.viewer.add_shapes(
            name="device_ROI", edge_color="red", face_color="transparent",
        )
        self._roi_layer.editable = True
        self._roi_outer_layer = self.viewer.add_shapes(
            name="device_outer", edge_color="yellow", face_color="transparent",
        )
        self._roi_outer_layer.editable = False

        # Connect ROI drag events
        try:
            self._roi_layer.events.data.connect(self._on_roi_changed)
        except Exception:
            pass

        # Build widget panel
        self._job_dropdown = ComboBox(
            label="Image",
            choices=self._job_choices(),
        )
        self._job_dropdown.changed.connect(self._on_dropdown_changed)

        self._status_label = Label(value="")

        self._organoid_mode_dropdown = ComboBox(
            label="Organoid mode",
            choices=["off", "dark", "light"],
        )
        self._organoid_mode_dropdown.changed.connect(self._on_organoid_mode_changed)

        self._device_width_widget = magicgui(
            self._on_device_width_changed,
            device_width_um={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0,
                             "step": 1.0, "value": self._default_device_width_um},
            auto_call=True,
            call_button=False,
        )

        self._btn_prev = PushButton(text="◀ Previous")
        self._btn_prev.clicked.connect(self._go_prev)
        self._btn_next = PushButton(text="Next ▶")
        self._btn_next.clicked.connect(self._go_next)

        self._btn_skip = PushButton(text="Mark as Skip")
        self._btn_skip.clicked.connect(self._mark_skip)

        self._btn_accept = PushButton(text="Accept ✓")
        self._btn_accept.clicked.connect(self._accept_current)

        self._btn_done = PushButton(text="Done — Start Batch")
        self._btn_done.clicked.connect(self._done_start_batch)

        self._progress_label = Label(value="")

        panel = Container(widgets=[
            self._job_dropdown,
            self._status_label,
            self._organoid_mode_dropdown,
            self._device_width_widget,
            self._btn_prev,
            self._btn_next,
            self._btn_skip,
            self._btn_accept,
            self._btn_done,
            self._progress_label,
        ])
        self.viewer.window.add_dock_widget(panel, area="right", name="Curation")

    def _job_choices(self) -> list[str]:
        return [f"{i}: {j.image_name}" for i, j in enumerate(self.jobs)]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _save_current_state(self):
        """Save napari layer state back to the current job."""
        if not self.jobs:
            return
        job = self.jobs[self._current_idx]

        # Save organoid mask from Labels layer
        if self._organoid_labels_layer is not None:
            labels_data = np.asarray(self._organoid_labels_layer.data)
            if job.focus_plane is not None and labels_data.shape == job.focus_plane.shape:
                job.organoid_mask_xy = labels_data > 0

        # Save device ROI corners from Shapes layer
        if self._roi_layer is not None and len(self._roi_layer.data) > 0:
            corners_yx = np.asarray(self._roi_layer.data[0])
            job.inner_corners = corners_yx[:, ::-1]  # yx → xy

    def _show_job(self, idx: int):
        """Load job at index into the napari viewer."""
        if idx < 0 or idx >= len(self.jobs):
            return
        self._current_idx = idx
        job = self.jobs[idx]

        # Update image layer
        if job.focus_plane is not None:
            self._image_layer.data = job.focus_plane
            self.viewer.reset_view()
        else:
            self._image_layer.data = np.zeros((100, 100), dtype=np.float32)

        # Update organoid labels layer
        if job.organoid_mask_xy is not None and job.focus_plane is not None:
            labels = np.zeros(job.focus_plane.shape[:2], dtype=np.int32)
            mask = np.asarray(job.organoid_mask_xy, dtype=bool)
            if mask.shape == labels.shape:
                labels[mask] = 1
            self._organoid_labels_layer.data = labels
        else:
            shape = job.focus_plane.shape[:2] if job.focus_plane is not None else (100, 100)
            self._organoid_labels_layer.data = np.zeros(shape, dtype=np.int32)

        # Update device ROI shapes
        if job.inner_corners is not None:
            corners_yx = job.inner_corners[:, ::-1]  # xy → yx
            self._roi_layer.data = [corners_yx]
            self._roi_layer.shape_type = ["polygon"]
        else:
            self._roi_layer.data = []

        # Update outer geometry
        self._update_outer_geometry(job)

        # Update widgets
        self._syncing = True
        try:
            self._job_dropdown.value = self._job_choices()[idx]
            self._organoid_mode_dropdown.value = job.organoid_mode if job.organoid_mode in ("off", "dark", "light") else "off"
            self._device_width_widget.device_width_um.value = job.device_width_um
        finally:
            self._syncing = False

        self._update_status()

    def _update_outer_geometry(self, job: CuratedJob):
        """Recompute and display the yellow outer geometry from current inner corners."""
        if job.inner_corners is None or job.x_step_um is None or job.y_step_um is None:
            self._roi_outer_layer.data = []
            return

        px_xy = um_to_xy_pixels(job.device_width_um, job.x_step_um, job.y_step_um)
        if px_xy is None:
            self._roi_outer_layer.data = []
            return

        expand_x_px, expand_y_px = px_xy
        expanded_xy = self._engine._expand_rectangle_corners(
            job.inner_corners, expand_x_px, expand_y_px,
        )
        if expanded_xy is None:
            self._roi_outer_layer.data = []
            return

        expanded_yx = expanded_xy[:, ::-1]
        self._roi_outer_layer.data = [expanded_yx]
        self._roi_outer_layer.shape_type = ["polygon"]

    def _update_status(self):
        """Update the status label and progress summary."""
        if not self.jobs:
            return
        job = self.jobs[self._current_idx]
        status_text = job.status.upper()
        if job.status == "failed":
            status_text += f": {job.error_msg}"
        self._status_label.value = f"Status: {status_text}"

        n_curated = sum(1 for j in self.jobs if j.status == "curated")
        n_skip = sum(1 for j in self.jobs if j.status == "skip")
        n_pending = sum(1 for j in self.jobs if j.status == "pending")
        n_failed = sum(1 for j in self.jobs if j.status == "failed")
        total = len(self.jobs)
        self._progress_label.value = (
            f"{n_curated}/{total} curated, {n_skip} skipped, "
            f"{n_pending} pending, {n_failed} failed"
        )

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _on_dropdown_changed(self, value):
        if self._syncing:
            return
        # Parse index from "0: name"
        try:
            idx = int(str(value).split(":")[0])
        except (ValueError, IndexError):
            return
        self._save_current_state()
        self._show_job(idx)

    def _on_organoid_mode_changed(self, value):
        if self._syncing:
            return
        job = self.jobs[self._current_idx]
        new_mode = str(value)
        if new_mode == job.organoid_mode:
            return
        job.organoid_mode = new_mode
        # Re-run auto-detection for organoid with new mode
        self._re_detect_organoid(job)
        # Update labels layer
        if job.organoid_mask_xy is not None and job.focus_plane is not None:
            labels = np.zeros(job.focus_plane.shape[:2], dtype=np.int32)
            mask = np.asarray(job.organoid_mask_xy, dtype=bool)
            if mask.shape == labels.shape:
                labels[mask] = 1
            self._organoid_labels_layer.data = labels
        else:
            shape = job.focus_plane.shape[:2] if job.focus_plane is not None else (100, 100)
            self._organoid_labels_layer.data = np.zeros(shape, dtype=np.int32)

    def _on_device_width_changed(self, device_width_um: float):
        if self._syncing:
            return
        job = self.jobs[self._current_idx]
        job.device_width_um = float(device_width_um)
        self._update_outer_geometry(job)

    def _on_roi_changed(self, event=None):
        """When the user drags device corners, update the outer geometry."""
        if self._syncing:
            return
        job = self.jobs[self._current_idx]
        if self._roi_layer is not None and len(self._roi_layer.data) > 0:
            corners_yx = np.asarray(self._roi_layer.data[0])
            job.inner_corners = corners_yx[:, ::-1]  # yx → xy
        self._update_outer_geometry(job)

    def _go_prev(self):
        if self._current_idx > 0:
            self._save_current_state()
            self._show_job(self._current_idx - 1)

    def _go_next(self):
        if self._current_idx < len(self.jobs) - 1:
            self._save_current_state()
            self._show_job(self._current_idx + 1)

    def _mark_skip(self):
        job = self.jobs[self._current_idx]
        job.status = "skip"
        self._update_status()
        # Auto-advance
        if self._current_idx < len(self.jobs) - 1:
            self._save_current_state()
            self._show_job(self._current_idx + 1)

    def _accept_current(self):
        self._save_current_state()
        job = self.jobs[self._current_idx]
        if job.inner_corners is None:
            self._status_label.value = "Status: Cannot accept — no device ROI detected"
            return
        job.status = "curated"
        self._update_status()
        # Auto-advance
        if self._current_idx < len(self.jobs) - 1:
            self._show_job(self._current_idx + 1)

    def _done_start_batch(self):
        """Validate all jobs and close viewer to start batch processing."""
        self._save_current_state()

        unresolved = [j for j in self.jobs if j.status not in ("curated", "skip")]
        if unresolved:
            names = ", ".join(j.image_name for j in unresolved[:5])
            extra = f" (and {len(unresolved)-5} more)" if len(unresolved) > 5 else ""
            self._status_label.value = (
                f"Cannot start: {len(unresolved)} images not curated/skipped: {names}{extra}"
            )
            return

        self.viewer.close()

    # ------------------------------------------------------------------
    # Finalise: produce cropped outputs for each curated job
    # ------------------------------------------------------------------

    def _finalise_jobs(self) -> list[CuratedJob]:
        """After viewer is closed, produce cropped outputs for curated jobs.

        Reloads 3D stacks from disk, applies device width expansion, perspective
        warp, and focus-patch voting.
        """
        curated = [j for j in self.jobs if j.status == "curated"]
        print(f"\nFinalising {len(curated)} curated jobs...")

        for i, job in enumerate(curated):
            print(f"  [{i+1}/{len(curated)}] {job.image_name} ...", end=" ", flush=True)
            try:
                self._finalise_single(job)
                print("OK")
            except Exception as exc:
                job.status = "failed"
                job.error_msg = f"Finalise failed: {exc}"
                print(f"FAILED: {exc}")

        return self.jobs

    def _finalise_single(self, job: CuratedJob):
        """Produce cropped outputs for one curated job."""
        eng = self._engine

        # Reload images into engine to get the 3D stack
        eng._list_images(job.source_path)
        selected_label = None
        for lbl, idx in eng._image_choice_map.items():
            if int(idx) == int(job.image_index):
                selected_label = lbl
                break
        if selected_label is None:
            raise ValueError(f"image_index {job.image_index} not found")

        # Load the stack — use _compute_focus_plane_from_stack to populate eng._last_stack
        source_is_lif = job.source_path.suffix.lower() == ".lif"
        if source_is_lif:
            eng._compute_focus_plane_from_stack(
                stack=None, downsample=4, n_sampling=10, patch=50,
                source_is_lif=True, image_index=job.image_index,
            )
        else:
            import tifffile
            arr = np.asarray(tifffile.imread(str(eng._image_paths[job.image_index])))
            if arr.ndim == 4:
                ch_candidates = [i for i in range(arr.ndim) if arr.shape[i] < 4]
                ch_axis = ch_candidates[0] if ch_candidates else int(np.argmin(arr.shape))
                ch_idx = min(eng.channel, arr.shape[ch_axis] - 1)
                arr = np.take(arr, ch_idx, axis=ch_axis)
            eng._last_stack = arr.astype(np.float32)
            z_um, y_um, x_um = read_voxel_size_um(eng._image_paths[job.image_index], source_is_lif=False)
            eng._set_last_voxel_steps(z_um, y_um, x_um)
            _, zmap, _, _ = eng._curved_plane_refocus(
                eng._last_stack, grid=10, patch=50,
            )
            eng._last_focus_zmap_full = np.asarray(zmap, dtype=np.int16)

        # We need the full-res stack
        stack = eng._last_stack
        if stack is None:
            raise RuntimeError("Could not load 3D stack")

        # Compute outer corners from inner corners + device width
        px_xy = um_to_xy_pixels(job.device_width_um, job.x_step_um, job.y_step_um)
        if px_xy is None:
            raise RuntimeError("Cannot convert device width to pixels — missing voxel size")
        expand_x_px, expand_y_px = px_xy
        outer_corners_xy = eng._expand_rectangle_corners(
            job.inner_corners, expand_x_px, expand_y_px,
        )
        if outer_corners_xy is None:
            raise RuntimeError("Could not expand rectangle corners")

        # Refocus stack
        scale = max(1, int(job._focus_downsample))
        if eng._last_focus_zmap_full is not None:
            refocused = eng._refocus_stack_around_plane(stack, eng._last_focus_zmap_full)
            if refocused is not None:
                stack = refocused

        # Crop stack with outer corners (full resolution)
        cropped_stack = eng._crop_rectified(stack, outer_corners_xy * float(scale))
        if cropped_stack is None:
            raise RuntimeError("Perspective warp failed for stack")

        # Crop inner ROI for focus votes
        inner_corners_full = job.inner_corners * float(scale)
        roi_stack = eng._crop_rectified(stack, inner_corners_full)

        # Compute focus votes
        vote_counts = None
        if roi_stack is not None:
            vote_counts = eng._compute_focus_patch_votes_for_stack(
                roi_stack.astype(np.float32),
                n_sampling=job._focus_n_sampling,
                patch=job._focus_patch,
            )

        # Crop organoid mask if present
        cropped_organoid_mask = None
        if job.organoid_mask_xy is not None and job.organoid_mode != "off":
            org_mask = np.asarray(job.organoid_mask_xy, dtype=bool)
            target_h, target_w = stack.shape[1], stack.shape[2]
            if scale > 1:
                org_mask = np.repeat(
                    np.repeat(org_mask.astype(np.uint8), scale, axis=0),
                    scale, axis=1,
                ).astype(bool)
            # Pad/clip to target
            if org_mask.shape[0] < target_h:
                org_mask = np.pad(org_mask, ((0, target_h - org_mask.shape[0]), (0, 0)),
                                  mode="constant", constant_values=False)
            if org_mask.shape[1] < target_w:
                org_mask = np.pad(org_mask, ((0, 0), (0, target_w - org_mask.shape[1])),
                                  mode="constant", constant_values=False)
            org_mask = org_mask[:target_h, :target_w]

            cropped_org = eng._crop_rectified_from_corners(
                org_mask.astype(np.float32),
                outer_corners_xy * float(scale),
            )
            if cropped_org is not None:
                cropped_organoid_mask = np.asarray(cropped_org > 0.5, dtype=bool)

        # Build z_votes dict
        z_votes = None
        if vote_counts is not None and len(vote_counts) > 0:
            z_votes = {int(i): int(c) for i, c in enumerate(vote_counts)}

        # Store finalised outputs on the job for downstream use
        job._finalised_outputs = {
            "cropped_stack": cropped_stack,
            "device_width_um": job.device_width_um,
            "pixel_size_um": job.pixel_size_um,
            "z_votes": z_votes,
            "image_name": job.image_name,
            "mask_central_region_enabled": job.organoid_mode != "off",
            "cropped_organoid_mask_xy": cropped_organoid_mask,
            "source_path": str(job.source_path),
            "image_index": job.image_index,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[CuratedJob]:
        """Launch curation GUI, block until done, return curated job list.

        Returns
        -------
        list[CuratedJob]
            All jobs with status "curated" have ``_finalised_outputs`` dict
            ready for VascuMap consumption. Jobs with status "skip" are included
            but have no outputs.
        """
        if not self.jobs:
            print("No jobs to curate.")
            return self.jobs

        # Phase A-1: auto-detect all
        self._auto_detect_all()

        # Phase A-2: build viewer and show first job
        self._build_viewer()
        self._show_job(0)

        # Block until viewer is closed
        napari.run()

        # Phase A-3: finalise curated jobs (reload stacks, crop, compute votes)
        self._finalise_jobs()

        return self.jobs

    # ------------------------------------------------------------------
    # Manifest I/O (optional persistence)
    # ------------------------------------------------------------------

    def save_manifest(self, path: Path):
        """Save curation state to JSON for session resumability.

        Organoid masks are saved as separate .npy files alongside the JSON.
        """
        manifest_path = Path(path)
        manifest_dir = manifest_path.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)

        entries = []
        for i, job in enumerate(self.jobs):
            entry = {
                "source_path": str(job.source_path),
                "image_index": job.image_index,
                "image_name": job.image_name,
                "organoid_mode": job.organoid_mode,
                "device_width_um": job.device_width_um,
                "status": job.status,
                "error_msg": job.error_msg,
                "inner_corners": job.inner_corners.tolist() if job.inner_corners is not None else None,
                "pixel_size_um": job.pixel_size_um,
                "z_step_um": job.z_step_um,
                "y_step_um": job.y_step_um,
                "x_step_um": job.x_step_um,
            }
            # Save organoid mask as .npy if manually edited
            if job.organoid_mask_xy is not None:
                mask_path = manifest_dir / f"organoid_mask_{i}_{job.image_name}.npy"
                np.save(str(mask_path), job.organoid_mask_xy.astype(np.uint8))
                entry["organoid_mask_npy"] = mask_path.name
            entries.append(entry)

        manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Manifest saved to {manifest_path}")

    @classmethod
    def load_manifest(cls, path: Path, device_width_um: float = 35.0) -> list[CuratedJob]:
        """Load curation state from a saved manifest JSON."""
        manifest_path = Path(path)
        manifest_dir = manifest_path.parent
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))

        jobs = []
        for i, entry in enumerate(entries):
            job = CuratedJob(
                source_path=Path(entry["source_path"]),
                image_index=entry["image_index"],
                image_name=entry["image_name"],
                organoid_mode=entry.get("organoid_mode", "off"),
                device_width_um=entry.get("device_width_um", device_width_um),
                status=entry.get("status", "pending"),
                error_msg=entry.get("error_msg", ""),
                pixel_size_um=entry.get("pixel_size_um"),
                z_step_um=entry.get("z_step_um"),
                y_step_um=entry.get("y_step_um"),
                x_step_um=entry.get("x_step_um"),
            )
            corners = entry.get("inner_corners")
            if corners is not None:
                job.inner_corners = np.asarray(corners, dtype=float)
            mask_npy = entry.get("organoid_mask_npy")
            if mask_npy is not None:
                mask_file = manifest_dir / mask_npy
                if mask_file.exists():
                    job.organoid_mask_xy = np.load(str(mask_file)).astype(bool)
            jobs.append(job)

        return jobs
