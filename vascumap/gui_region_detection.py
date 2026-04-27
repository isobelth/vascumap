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
import tifffile
from magicgui import magicgui
from magicgui.widgets import ComboBox, Container, Label, PushButton
from skimage.filters import sobel
from device_segmentation import DeviceSegmentationApp, read_voxel_size_um, um_to_xy_pixels


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CuratedJob:
    """Per-image curation state tracked through the napari GUI session."""

    source_path: Path
    image_index: int
    image_name: str
    organoid_mode: str          # "dark" / "light" / "off"
    device_width_um: float
    status: str = "pending"     # "pending" / "curated" / "skip" / "failed"
    error_msg: str = ""

    inner_corners: Optional[np.ndarray] = None       # (4, 2) xy
    organoid_mask_xy: Optional[np.ndarray] = None    # 2D bool, full-res focus plane
    pixel_size_um: Optional[dict] = None
    focus_plane: Optional[np.ndarray] = None
    z_step_um: Optional[float] = None
    y_step_um: Optional[float] = None
    x_step_um: Optional[float] = None
    xy_step_um: Optional[float] = None

    focus_zmap_full: Optional[np.ndarray] = field(default=None, repr=False)
    focus_downsample: int = 4
    focus_n_sampling: int = 10
    focus_patch: int = 50
    segment_debug: Optional[dict] = field(default=None, repr=False)


def infer_organoid_mode(source_path: Path, image_name: str) -> str:
    """Infer organoid mode from filename heuristic (marina/bead keywords)."""
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

    Detects device ROI + organoid mask for every job automatically, then
    lets the user navigate through each image in a single napari session to
    accept, edit, or skip results. Finalised outputs are returned to the
    batch pipeline via an optional callback.

    Usage::

        app = CurationApp(jobs, device_width_um=35.0)
        app.open()   # opens napari; blocks until viewer is closed via exec_()
    """

    def __init__(self, jobs: list[tuple], device_width_um: float = 35.0, brightfield_channel: int = 0, default_organoid_mode: str = "infer", on_done=None):
        """Initialise the curation app.

        Parameters
        ----------
        jobs:
            List of (source_path, image_index, mask_flag, image_name) tuples
            as returned by ``pipeline.discover_jobs()``.
        device_width_um:
            Default device border width in µm used for the outer geometry overlay.
        brightfield_channel:
            Channel index to use for multi-channel stacks.
        default_organoid_mode:
            ``"infer"`` | ``"dark"`` | ``"light"`` | ``"off"`` – starting
            organoid mode for each job (overridden by per-job mask_flag).
        on_done:
            Optional callable invoked with the job list after the user clicks
            *Done – Start Batch*.
        """
        self.default_device_width_um = float(device_width_um)
        self.brightfield_channel = int(brightfield_channel)
        self.default_organoid_mode = default_organoid_mode
        self.on_done = on_done

        self.jobs: list[CuratedJob] = []
        for source_path, image_index, mask_flag, image_name in jobs:
            if default_organoid_mode == "infer":
                mode = infer_organoid_mode(Path(source_path), image_name)
            elif default_organoid_mode == "off":
                mode = "off"
            else:
                mode = str(default_organoid_mode)
            if mask_flag is not False and mask_flag is not None:
                mode = str(mask_flag)
            elif mask_flag is False:
                mode = "off"
            self.jobs.append(CuratedJob(source_path=Path(source_path), image_index=int(image_index), image_name=str(image_name), organoid_mode=mode, device_width_um=float(device_width_um)))

        self.current_idx: int = 0
        self.engine = DeviceSegmentationApp(enable_gui=False)
        self.engine.channel = self.brightfield_channel

        self.viewer: Optional[napari.Viewer] = None
        self.image_layer = None
        self.organoid_labels_layer = None
        self.roi_layer = None
        self.roi_outer_layer = None
        self.syncing = False
        self.syncing_outer_geometry = False

    # ------------------------------------------------------------------
    # Auto-detection
    # ------------------------------------------------------------------

    def auto_detect_all(self):
        """Run device + organoid auto-detection for every job.

        Successful jobs are pre-accepted (status='curated'). Navigate in
        napari to review; press ``s`` to skip any you don't want, or edit
        the device ROI / organoid mask and press ``a`` to re-accept.
        """
        print(f"Auto-detecting device regions for {len(self.jobs)} images...")
        for i, job in enumerate(self.jobs):
            print(f"  [{i+1}/{len(self.jobs)}] {job.image_name} ...", end=" ", flush=True)
            try:
                self.auto_detect_single(job)
                job.status = "curated"
                print("OK")
            except Exception as exc:
                job.status = "failed"
                job.error_msg = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {exc}")

    def auto_detect_single(self, job: CuratedJob):
        """Run device segmentation and organoid detection for one job."""
        eng = self.engine
        eng.list_images(job.source_path)

        selected_label = None
        for label, idx in eng.image_choice_map.items():
            if int(idx) == int(job.image_index):
                selected_label = label
                break
        if selected_label is None:
            raise ValueError(f"image_index {job.image_index} not found in {job.source_path}")

        eng.segment_and_view(image_choice=selected_label, focus_downsample=4, focus_n_sampling=10, focus_patch=50, mask_central_region=(job.organoid_mode != "off"), clear_layers=True)

        debug = getattr(eng, "last_segment_debug", None) or {}
        final_corners = debug.get("final_corners", None)
        if final_corners is None:
            msg = str(getattr(eng.images_output, "value", "Segmentation failed"))
            raise RuntimeError(msg)

        focus_plane = None
        stack = getattr(eng, "last_stack", None)
        zmap = getattr(eng, "last_focus_zmap_full", None)
        ds = max(1, int(getattr(eng, "last_focus_downsample", 1)))
        if stack is not None and zmap is not None:
            try:
                arr = np.asarray(stack)
                zm = np.asarray(zmap, dtype=np.int32)
                if arr.ndim == 4:
                    arr = np.mean(arr, axis=-1)
                if arr.ndim == 3 and zm.shape == arr.shape[1:]:
                    zm_clip = np.clip(zm, 0, arr.shape[0] - 1).astype(np.int64)
                    refocused = np.take_along_axis(arr, zm_clip[None, :, :], axis=0)[0].astype(np.float32)
                    if ds > 1:
                        refocused = refocused[::ds, ::ds]
                    focus_plane = refocused
            except Exception as exc:
                print(f"[curation] refocused image build failed: {exc}")

        if focus_plane is None:
            focus_plane = eng.last_image
            if focus_plane is not None:
                focus_plane = np.asarray(focus_plane, dtype=np.float32)
                if focus_plane.ndim == 3:
                    focus_plane = np.mean(focus_plane, axis=-1).astype(np.float32)

        organoid_mask = None
        if job.organoid_mode != "off":
            org_region = eng.last_organoid_region
            if org_region is not None:
                organoid_mask = np.asarray(org_region, dtype=bool)

        job.focus_plane = focus_plane
        job.inner_corners = np.asarray(final_corners, dtype=float)
        job.organoid_mask_xy = organoid_mask
        job.z_step_um = eng.last_z_step_um
        job.y_step_um = eng.last_y_step_um
        job.x_step_um = eng.last_x_step_um
        job.xy_step_um = eng.last_xy_step_um
        job.pixel_size_um = {"x_um": eng.last_x_step_um, "y_um": eng.last_y_step_um, "z_um": eng.last_z_step_um, "xy_um": eng.last_xy_step_um}
        job.focus_zmap_full = eng.last_focus_zmap_full
        job.focus_downsample = eng.last_focus_downsample
        job.focus_n_sampling = eng.last_focus_n_sampling
        job.focus_patch = eng.last_focus_patch
        job.segment_debug = debug

    def re_detect_organoid(self, job: CuratedJob):
        """Re-run organoid detection with the current mode for one job."""
        if job.focus_plane is None:
            return
        if job.organoid_mode == "off":
            job.organoid_mask_xy = None
            return
        try:
            org_mask = self.engine.mask_out_organoid(job.focus_plane, mode=job.organoid_mode)
            job.organoid_mask_xy = np.asarray(org_mask, dtype=bool) if org_mask is not None else None
        except Exception:
            job.organoid_mask_xy = None

    # ------------------------------------------------------------------
    # Viewer setup
    # ------------------------------------------------------------------

    def build_viewer(self):
        """Create the napari viewer, initialise layers, and add the widget panel."""
        self.viewer = napari.Viewer(title="VascuMap Curation")

        dummy = np.zeros((100, 100), dtype=np.float32)
        self.image_layer = self.viewer.add_image(dummy, name="focus_plane")
        self.organoid_labels_layer = self.viewer.add_labels(np.zeros((100, 100), dtype=np.int32), name="organoid_mask")
        self.organoid_labels_layer.opacity = 0.4

        self.roi_layer = self.viewer.add_shapes(name="device_ROI", edge_color="red", face_color="transparent")
        self.roi_layer.editable = True
        try:
            self.roi_layer.mode = "select"
        except Exception:
            pass
        self.roi_outer_layer = self.viewer.add_shapes(name="device_outer", edge_color="yellow", face_color="transparent")
        self.roi_outer_layer.editable = False

        try:
            self.roi_layer.events.data.connect(self.on_roi_changed)
        except Exception:
            pass

        self.job_dropdown = ComboBox(label="Image", choices=self.job_choices())
        self.job_dropdown.changed.connect(self.on_dropdown_changed)
        self.status_label = Label(value="")

        self.organoid_mode_dropdown = ComboBox(label="Organoid mode", choices=["off", "dark", "light"])
        self.organoid_mode_dropdown.changed.connect(self.on_organoid_mode_changed)

        self.device_width_widget = magicgui(self.on_device_width_changed, device_width_um={"widget_type": "FloatSpinBox", "min": 0.0, "max": 1000.0, "step": 1.0, "value": self.default_device_width_um}, auto_call=True, call_button=False)

        self.btn_prev = PushButton(text="◀ Previous")
        self.btn_prev.clicked.connect(self.go_prev)
        self.btn_next = PushButton(text="Next ▶")
        self.btn_next.clicked.connect(self.go_next)
        self.btn_skip = PushButton(text="Mark as Skip")
        self.btn_skip.clicked.connect(self.mark_skip)
        self.btn_accept = PushButton(text="Accept ✓")
        self.btn_accept.clicked.connect(self.accept_current)
        self.btn_done = PushButton(text="Done — Start Batch")
        self.btn_done.clicked.connect(self.done_start_batch)
        self.progress_label = Label(value="")

        panel = Container(widgets=[self.job_dropdown, self.status_label, self.organoid_mode_dropdown, self.device_width_widget, self.btn_prev, self.btn_next, self.btn_skip, self.btn_accept, self.btn_done, self.progress_label])
        self.viewer.window.add_dock_widget(panel, area="right", name="Curation")

    def job_choices(self) -> list[str]:
        """Return display strings for the job dropdown (index: image_name)."""
        return [f"{i}: {j.image_name}" for i, j in enumerate(self.jobs)]

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def save_current_state(self):
        """Write back the current napari layer state to the active job."""
        if not self.jobs:
            return
        job = self.jobs[self.current_idx]
        if self.organoid_labels_layer is not None:
            labels_data = np.asarray(self.organoid_labels_layer.data)
            if job.focus_plane is not None and labels_data.shape == job.focus_plane.shape:
                job.organoid_mask_xy = labels_data > 0
        if self.roi_layer is not None and len(self.roi_layer.data) > 0:
            corners_yx = np.asarray(self.roi_layer.data[0])
            job.inner_corners = corners_yx[:, ::-1]  # yx → xy

    def show_job(self, idx: int):
        """Load the job at idx into the napari viewer layers."""
        if idx < 0 or idx >= len(self.jobs):
            return
        self.current_idx = idx
        job = self.jobs[idx]

        data = job.focus_plane if job.focus_plane is not None else np.zeros((100, 100), dtype=np.float32)
        try:
            arr = np.asarray(data)
            finite = arr[np.isfinite(arr)] if arr.size else arr
            print(f"[curation] focus_plane job={job.image_name} shape={arr.shape} dtype={arr.dtype} min={float(finite.min()) if finite.size else 'NA'} max={float(finite.max()) if finite.size else 'NA'}")
        except Exception:
            pass

        try:
            insert_idx = self.viewer.layers.index(self.image_layer)
            try:
                self.roi_layer.events.data.disconnect(self.on_roi_changed)
            except Exception:
                pass
            self.viewer.layers.remove(self.image_layer)
            self.image_layer = self.viewer.add_image(data, name="focus_plane", colormap="gray")
            try:
                self.viewer.layers.move(len(self.viewer.layers) - 1, insert_idx)
            except Exception:
                pass
            try:
                self.roi_layer.events.data.connect(self.on_roi_changed)
            except Exception:
                pass
        except Exception as exc:
            print(f"[curation] image layer refresh failed: {exc}")
            self.image_layer.data = data
        if job.focus_plane is not None:
            self.viewer.reset_view()

        if job.organoid_mask_xy is not None and job.focus_plane is not None:
            labels = np.zeros(job.focus_plane.shape[:2], dtype=np.int32)
            mask = np.asarray(job.organoid_mask_xy, dtype=bool)
            if mask.shape == labels.shape:
                labels[mask] = 1
            self.organoid_labels_layer.data = labels
        else:
            shape = job.focus_plane.shape[:2] if job.focus_plane is not None else (100, 100)
            self.organoid_labels_layer.data = np.zeros(shape, dtype=np.int32)

        self.syncing_outer_geometry = True
        try:
            if job.inner_corners is not None:
                corners_yx = job.inner_corners[:, ::-1]
                self.roi_layer.data = [corners_yx]
                self.roi_layer.shape_type = ["rectangle"]
                self.roi_layer.edge_color = "red"
                self.roi_layer.face_color = "transparent"
                self.roi_layer.editable = True
                try:
                    self.roi_layer.mode = "select"
                except Exception:
                    pass
            else:
                self.roi_layer.data = []
        finally:
            self.syncing_outer_geometry = False

        self.update_outer_geometry(job)

        self.syncing = True
        try:
            self.job_dropdown.value = self.job_choices()[idx]
            self.organoid_mode_dropdown.value = job.organoid_mode if job.organoid_mode in ("off", "dark", "light") else "off"
            self.device_width_widget.device_width_um.value = job.device_width_um
        finally:
            self.syncing = False

        self.update_status()

    def update_outer_geometry(self, job: CuratedJob):
        """Recompute and display the yellow outer geometry from the current inner corners."""
        if job.inner_corners is None or job.x_step_um is None or job.y_step_um is None:
            self.roi_outer_layer.data = []
            return
        px_xy = um_to_xy_pixels(job.device_width_um, job.x_step_um, job.y_step_um)
        if px_xy is None:
            self.roi_outer_layer.data = []
            return
        expand_x_px, expand_y_px = px_xy
        expanded_xy = self.engine.expand_rectangle_corners(job.inner_corners, expand_x_px, expand_y_px)
        if expanded_xy is None:
            self.roi_outer_layer.data = []
            return
        expanded_yx = expanded_xy[:, ::-1]
        self.syncing_outer_geometry = True
        try:
            self.roi_outer_layer.data = [expanded_yx]
            self.roi_outer_layer.shape_type = ["rectangle"]
            self.roi_outer_layer.edge_color = "yellow"
            self.roi_outer_layer.face_color = "transparent"
            self.roi_outer_layer.editable = False
        finally:
            self.syncing_outer_geometry = False

    def update_status(self):
        """Refresh the status label and progress summary widgets."""
        if not self.jobs:
            return
        job = self.jobs[self.current_idx]
        status_text = job.status.upper()
        if job.status == "failed":
            status_text += f": {job.error_msg}"
        self.status_label.value = f"Status: {status_text}"
        n_curated = sum(1 for j in self.jobs if j.status == "curated")
        n_skip = sum(1 for j in self.jobs if j.status == "skip")
        n_pending = sum(1 for j in self.jobs if j.status == "pending")
        n_failed = sum(1 for j in self.jobs if j.status == "failed")
        self.progress_label.value = f"{n_curated}/{len(self.jobs)} curated, {n_skip} skipped, {n_pending} pending, {n_failed} failed"

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def on_dropdown_changed(self, value):
        """Navigate to the job selected in the image dropdown."""
        if self.syncing:
            return
        try:
            idx = int(str(value).split(":")[0])
        except (ValueError, IndexError):
            return
        self.save_current_state()
        self.show_job(idx)

    def on_organoid_mode_changed(self, value):
        """Re-run organoid detection when the user changes the mode dropdown."""
        if self.syncing:
            return
        job = self.jobs[self.current_idx]
        new_mode = str(value)
        if new_mode == job.organoid_mode:
            return
        job.organoid_mode = new_mode
        self.re_detect_organoid(job)
        if job.organoid_mask_xy is not None and job.focus_plane is not None:
            labels = np.zeros(job.focus_plane.shape[:2], dtype=np.int32)
            mask = np.asarray(job.organoid_mask_xy, dtype=bool)
            if mask.shape == labels.shape:
                labels[mask] = 1
            self.organoid_labels_layer.data = labels
        else:
            shape = job.focus_plane.shape[:2] if job.focus_plane is not None else (100, 100)
            self.organoid_labels_layer.data = np.zeros(shape, dtype=np.int32)

    def on_device_width_changed(self, device_width_um: float):
        """Update the outer geometry overlay when the device width spinbox changes."""
        if self.syncing:
            return
        job = self.jobs[self.current_idx]
        job.device_width_um = float(device_width_um)
        self.update_outer_geometry(job)

    def on_roi_changed(self, event=None):
        """Sync the outer geometry overlay whenever the user drags the device ROI."""
        if self.syncing or self.syncing_outer_geometry:
            return
        job = self.jobs[self.current_idx]
        if self.roi_layer is not None and len(self.roi_layer.data) > 0:
            corners_yx = np.asarray(self.roi_layer.data[0])
            job.inner_corners = corners_yx[:, ::-1]
        self.update_outer_geometry(job)

    def go_prev(self):
        """Navigate to the previous job."""
        if self.current_idx > 0:
            self.save_current_state()
            self.show_job(self.current_idx - 1)

    def go_next(self):
        """Navigate to the next job."""
        if self.current_idx < len(self.jobs) - 1:
            self.save_current_state()
            self.show_job(self.current_idx + 1)

    def mark_skip(self):
        """Mark the current job as skipped and auto-advance to the next."""
        job = self.jobs[self.current_idx]
        job.status = "skip"
        self.update_status()
        if self.current_idx < len(self.jobs) - 1:
            self.save_current_state()
            self.show_job(self.current_idx + 1)

    def accept_current(self):
        """Accept the current job's curation and auto-advance to the next."""
        self.save_current_state()
        job = self.jobs[self.current_idx]
        if job.inner_corners is None:
            self.status_label.value = "Status: Cannot accept — no device ROI detected"
            return
        job.status = "curated"
        self.update_status()
        if self.current_idx < len(self.jobs) - 1:
            self.show_job(self.current_idx + 1)

    def done_start_batch(self):
        """Validate all jobs, finalise curated ones, and invoke the on_done callback."""
        self.save_current_state()
        unresolved = [j for j in self.jobs if j.status not in ("curated", "skip")]
        if unresolved:
            names = ", ".join(j.image_name for j in unresolved[:5])
            extra = f" (and {len(unresolved)-5} more)" if len(unresolved) > 5 else ""
            self.status_label.value = f"Cannot start: {len(unresolved)} images not curated/skipped: {names}{extra}"
            return

        self.status_label.value = "Finalising curated jobs... (see terminal output)"
        try:
            self.viewer.window.qt_window.repaint()
        except Exception:
            pass

        self.finalise_jobs()
        try:
            self.viewer.close()
        except Exception:
            pass

        if self.on_done is not None:
            try:
                self.on_done(self.jobs)
            except Exception as exc:
                print(f"[on_done callback failed] {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalise_jobs(self) -> list[CuratedJob]:
        """Produce pipeline-ready cropped outputs for every curated job."""
        curated = [j for j in self.jobs if j.status == "curated"]
        print(f"\nFinalising {len(curated)} curated jobs...")
        for i, job in enumerate(curated):
            print(f"  [{i+1}/{len(curated)}] {job.image_name} ...", end=" ", flush=True)
            try:
                self.finalise_single(job)
                print("OK")
            except Exception as exc:
                job.status = "failed"
                job.error_msg = f"Finalise failed: {exc}"
                print(f"FAILED: {exc}")
        return self.jobs

    def finalise_single(self, job: CuratedJob):
        """Reload the 3D stack, warp to ROI, and store finalised_outputs on the job."""
        eng = self.engine
        eng.list_images(job.source_path)
        selected_label = None
        for label, idx in eng.image_choice_map.items():
            if int(idx) == int(job.image_index):
                selected_label = label
                break
        if selected_label is None:
            raise ValueError(f"image_index {job.image_index} not found")

        source_is_lif = job.source_path.suffix.lower() == ".lif"
        if source_is_lif:
            eng.compute_focus_plane_from_stack(stack=None, downsample=4, n_sampling=10, patch=50, source_is_lif=True, image_index=job.image_index)
        else:
            arr = np.asarray(tifffile.imread(str(eng.image_paths[job.image_index])))
            if arr.ndim == 4:
                ch_candidates = [i for i in range(arr.ndim) if arr.shape[i] < 4]
                ch_axis = ch_candidates[0] if ch_candidates else int(np.argmin(arr.shape))
                ch_idx = min(eng.channel, arr.shape[ch_axis] - 1)
                arr = np.take(arr, ch_idx, axis=ch_axis)
            eng.last_stack = arr.astype(np.float32)
            z_um, y_um, x_um = read_voxel_size_um(eng.image_paths[job.image_index], source_is_lif=False)
            eng.set_last_voxel_steps(z_um, y_um, x_um)
            _, zmap, _, _ = eng.curved_plane_refocus(eng.last_stack, grid=10, patch=50)
            eng.last_focus_zmap_full = np.asarray(zmap, dtype=np.int16)

        stack = eng.last_stack
        if stack is None:
            raise RuntimeError("Could not load 3D stack")

        px_xy = um_to_xy_pixels(job.device_width_um, job.x_step_um, job.y_step_um)
        if px_xy is None:
            raise RuntimeError("Cannot convert device width to pixels — missing voxel size")
        expand_x_px, expand_y_px = px_xy
        outer_corners_xy = eng.expand_rectangle_corners(job.inner_corners, expand_x_px, expand_y_px)
        if outer_corners_xy is None:
            raise RuntimeError("Could not expand rectangle corners")

        scale = max(1, int(job.focus_downsample))
        if eng.last_focus_zmap_full is not None:
            refocused = eng.refocus_stack_around_plane(stack, eng.last_focus_zmap_full)
            if refocused is not None:
                stack = refocused

        cropped_stack = eng.crop_rectified(stack, outer_corners_xy * float(scale))
        if cropped_stack is None:
            raise RuntimeError("Perspective warp failed for stack")

        inner_corners_full = job.inner_corners * float(scale)
        roi_stack = eng.crop_rectified(stack, inner_corners_full)
        vote_counts = None
        if roi_stack is not None:
            vote_counts = eng.compute_focus_patch_votes_for_stack(roi_stack.astype(np.float32), n_sampling=job.focus_n_sampling, patch=job.focus_patch)

        cropped_organoid_mask = None
        if job.organoid_mask_xy is not None and job.organoid_mode != "off":
            org_mask = np.asarray(job.organoid_mask_xy, dtype=bool)
            target_h, target_w = stack.shape[1], stack.shape[2]
            if scale > 1:
                org_mask = np.repeat(np.repeat(org_mask.astype(np.uint8), scale, axis=0), scale, axis=1).astype(bool)
            if org_mask.shape[0] < target_h:
                org_mask = np.pad(org_mask, ((0, target_h - org_mask.shape[0]), (0, 0)), mode="constant", constant_values=False)
            if org_mask.shape[1] < target_w:
                org_mask = np.pad(org_mask, ((0, 0), (0, target_w - org_mask.shape[1])), mode="constant", constant_values=False)
            org_mask = org_mask[:target_h, :target_w]
            cropped_org = eng.crop_rectified_from_corners(org_mask.astype(np.float32), outer_corners_xy * float(scale))
            if cropped_org is not None:
                cropped_organoid_mask = np.asarray(cropped_org > 0.5, dtype=bool)

        z_votes = {int(i): int(c) for i, c in enumerate(vote_counts)} if vote_counts is not None and len(vote_counts) > 0 else None

        job.finalised_outputs = {"cropped_stack": cropped_stack, "device_width_um": job.device_width_um, "pixel_size_um": job.pixel_size_um, "z_votes": z_votes, "image_name": job.image_name, "mask_central_region_enabled": job.organoid_mode != "off", "cropped_organoid_mask_xy": cropped_organoid_mask, "source_path": str(job.source_path), "image_index": job.image_index}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open(self) -> "CurationApp":
        """Auto-detect all images, build the napari viewer, and return immediately.

        The viewer remains open via Jupyter's Qt event-loop integration.
        Navigate with keyboard shortcuts (n=next, b=back, a=accept, s=skip)
        or the panel buttons. When done curating call ``finalise()`` from
        the next notebook cell, or click *Done – Start Batch*.

        Returns self for chaining.
        """
        if not self.jobs:
            print("No jobs to curate.")
            return self
        self.auto_detect_all()
        self.build_viewer()
        self.add_key_bindings()
        self.show_job(0)
        n_ok = sum(1 for j in self.jobs if j.status == "curated")
        n_fail = sum(1 for j in self.jobs if j.status == "failed")
        print(f"\n{len(self.jobs)} images loaded ({n_ok} auto-accepted, {n_fail} failed).")
        print("Shortcuts: n=next, b=back, s=skip, a=re-accept after editing.")
        return self

    def finalise(self) -> list[CuratedJob]:
        """Save current edits, finalise curated jobs, and return the job list.

        Call this from a new notebook cell after finishing curation in the
        napari viewer. All curated jobs will have ``finalised_outputs`` ready
        for the VascuMap pipeline.
        """
        self.save_current_state()
        self.finalise_jobs()
        return self.jobs

    def run(self) -> list[CuratedJob]:
        """Open the viewer, block until closed, finalise, and return jobs.

        Prefer ``open()`` + ``finalise()`` in notebooks.
        """
        self.open()
        napari.run(force=True)
        return self.finalise()

    def add_key_bindings(self):
        """Register n/b/a/s keyboard shortcuts on the napari viewer."""
        @self.viewer.bind_key("n")
        def key_next(viewer):
            """Navigate to the next image."""
            self.go_next()

        @self.viewer.bind_key("b")
        def key_prev(viewer):
            """Navigate to the previous image."""
            self.go_prev()

        @self.viewer.bind_key("a")
        def key_accept(viewer):
            """Accept the current curation."""
            self.accept_current()

        @self.viewer.bind_key("s")
        def key_skip(viewer):
            """Skip the current image."""
            self.mark_skip()

    # ------------------------------------------------------------------
    # Manifest I/O
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
            entry = {"source_path": str(job.source_path), "image_index": job.image_index, "image_name": job.image_name, "organoid_mode": job.organoid_mode, "device_width_um": job.device_width_um, "status": job.status, "error_msg": job.error_msg, "inner_corners": job.inner_corners.tolist() if job.inner_corners is not None else None, "pixel_size_um": job.pixel_size_um, "z_step_um": job.z_step_um, "y_step_um": job.y_step_um, "x_step_um": job.x_step_um}
            if job.organoid_mask_xy is not None:
                mask_path = manifest_dir / f"organoid_mask_{i}_{job.image_name}.npy"
                np.save(str(mask_path), job.organoid_mask_xy.astype(np.uint8))
                entry["organoid_mask_npy"] = mask_path.name
            entries.append(entry)
        manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print(f"Manifest saved to {manifest_path}")

    @classmethod
    def load_manifest(cls, path: Path, device_width_um: float = 35.0) -> list[CuratedJob]:
        """Load curation state from a previously saved manifest JSON."""
        manifest_path = Path(path)
        manifest_dir = manifest_path.parent
        entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = []
        for i, entry in enumerate(entries):
            job = CuratedJob(source_path=Path(entry["source_path"]), image_index=entry["image_index"], image_name=entry["image_name"], organoid_mode=entry.get("organoid_mode", "off"), device_width_um=entry.get("device_width_um", device_width_um), status=entry.get("status", "pending"), error_msg=entry.get("error_msg", ""), pixel_size_um=entry.get("pixel_size_um"), z_step_um=entry.get("z_step_um"), y_step_um=entry.get("y_step_um"), x_step_um=entry.get("x_step_um"))
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
