from pathlib import Path
from typing import Optional, Tuple
import tifffile
from liffile import LifFile
import numpy as np
import napari
from magicgui import magicgui
from magicgui.widgets import Container, TextEdit
from skimage import util
from skimage.filters import threshold_triangle, median, sobel, gaussian, threshold_yen, threshold_otsu, threshold_minimum
from skimage.measure import label, regionprops_table, regionprops, moments_central
from skimage.morphology import disk, remove_small_objects, remove_small_holes, closing
from skimage.segmentation import expand_labels
from skimage.transform import ProjectiveTransform, warp, probabilistic_hough_line
from scipy.ndimage import rotate, binary_dilation, binary_erosion, binary_fill_holes
from skimage.draw import line


def mp_run_threshold_minimum(img_arr, out_path):
    """Subprocess target: run threshold_minimum and save result to .npy."""
    import numpy as _np
    from skimage.filters import threshold_minimum as _tm
    t = _tm(img_arr, max_num_iter=500)
    _np.save(out_path, _np.array([t]))


def read_voxel_size_um(
    source_path: Optional[Path],
    source_is_lif: bool,
    selected_lif: Optional[Path] = None,
    image_index: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return voxel size tuple (z_um, y_um, x_um) from LIF or TIFF metadata."""

    z_um = y_um = x_um = None

    if source_is_lif:
        lif_path = selected_lif if selected_lif is not None else source_path
        if lif_path is None:
            return (None, None, None)
        try:
            with LifFile(lif_path) as lif:
                idx = int(image_index if image_index is not None else 0)
                if idx < 0 or idx >= len(lif.images):
                    return (None, None, None)
                img = lif.images[idx]

                # Primary path (preferred): LifImage.coords from liffile.
                coords = getattr(img, "coords", None)
                if isinstance(coords, dict):
                    for axis in ("X", "Y", "Z"):
                        coord = coords.get(axis)
                        if coord is None:
                            continue
                        try:
                            if len(coord) < 2:
                                continue
                            step_um = abs(float(coord[1] - coord[0])) * 1e6
                            if not np.isfinite(step_um) or step_um <= 0:
                                continue
                            if axis == "X":
                                x_um = step_um
                            elif axis == "Y":
                                y_um = step_um
                            else:
                                z_um = step_um
                        except Exception:
                            continue

                # Minimal fallback: xarray coordinates if available.
                if x_um is None or y_um is None or z_um is None:
                    try:
                        xa = img.asxarray()
                        xa_coords = getattr(xa, "coords", None)
                        if xa_coords is not None:
                            for axis in ("X", "Y", "Z"):
                                coord = xa_coords.get(axis) if axis in xa_coords else None
                                if coord is None:
                                    for key in xa_coords.keys():
                                        if str(key).lower() == axis.lower():
                                            coord = xa_coords[key]
                                            break
                                if coord is None:
                                    continue
                                if len(coord) < 2:
                                    continue
                                step_um = abs(float(coord[1] - coord[0])) * 1e6
                                if not np.isfinite(step_um) or step_um <= 0:
                                    continue
                                if axis == "X" and x_um is None:
                                    x_um = step_um
                                elif axis == "Y" and y_um is None:
                                    y_um = step_um
                                elif axis == "Z" and z_um is None:
                                    z_um = step_um
                    except Exception:
                        pass
        except Exception:
            return (None, None, None)

        if x_um is None and y_um is not None:
            x_um = y_um
        if y_um is None and x_um is not None:
            y_um = x_um
        return (z_um, y_um, x_um)

    if source_path is not None and str(source_path).lower().endswith((".tif", ".tiff")):
        try:
            with tifffile.TiffFile(str(source_path)) as tif:
                tif_tags = {}
                for tag in tif.pages[0].tags.values():
                    tif_tags[tag.name] = tag.value

                if "XResolution" in tif_tags:
                    xres = tif_tags["XResolution"]
                    x_pixel_size_um = 1.0 / (float(xres[0]) / float(xres[1]))
                    x_um = x_pixel_size_um
                if "YResolution" in tif_tags:
                    yres = tif_tags["YResolution"]
                    y_pixel_size_um = 1.0 / (float(yres[0]) / float(yres[1]))
                    y_um = y_pixel_size_um

                try:
                    z_um = float(str(tif_tags["IJMetadata"]).split("nscales=")[1].split(",")[2].split("\\nunit")[0])
                except Exception:
                    try:
                        z_um = float(str(tif_tags["ImageDescription"]).split("spacing=")[1].split("loop")[0])
                    except Exception:
                        z_um = None
        except Exception:
            return (None, None, None)

        if x_um is None and y_um is not None:
            x_um = y_um
        if y_um is None and x_um is not None:
            y_um = x_um
        return (z_um, y_um, x_um)

    return (None, None, None)

def um_to_xy_pixels(
    width_um: float,
    x_um: Optional[float],
    y_um: Optional[float],
) -> Optional[Tuple[float, float]]:
    try:
        width_um = float(width_um)
    except Exception:
        return None
    if width_um < 0:
        return None

    if x_um is None and y_um is None:
        return None
    if x_um is None:
        x_um = y_um
    if y_um is None:
        y_um = x_um

    try:
        x_um = float(x_um)
        y_um = float(y_um)
    except Exception:
        return None
    if x_um <= 0 or y_um <= 0:
        return None

    return width_um / x_um, width_um / y_um

class DeviceSegmentationApp:
    def __init__(
        self,
        enable_gui: bool = True,
        low_frac: float = 0.82,
        smooth_window: int = 5,
        bin_size: float = 2.0,
        min_run_frac: float = 0.25,
        typical_pct: float = 50.0,
        line_length: int = 300,
        line_gap: int = 100,
        hough_threshold: int = 120,
        mask_sigma: float = 5.0,
    ):
        self.enable_gui = bool(enable_gui)
        self.low_frac = low_frac
        self.smooth_window = smooth_window
        self.bin_size = bin_size
        self.min_run_frac = min_run_frac
        self.typical_pct = typical_pct
        self.line_length = line_length
        self.line_gap = line_gap
        self.hough_threshold = hough_threshold
        self.mask_sigma = mask_sigma
        self.channel = 0

        self.viewer = napari.Viewer(show=self.enable_gui)

        self._selected_image_folder: Optional[Path] = None
        self._selected_lif: Optional[Path] = None
        self.image_paths = []
        self.image_choice_map = {}

        self._roi_layer = None
        self._roi_outer_layer = None
        self._active_device_width_um = None
        self._syncing_outer_geometry = False
        self._cropped_layer = None
        self.last_image = None
        self._hough_fallback_used = False
        self.last_segment_debug = None

        self.last_stack = None
        self.last_focus_downsample = 1
        self.last_focus_n_sampling = 10
        self.last_focus_patch = 50
        self.last_focus_zmap_full = None

        self.last_z_step_um = None
        self.last_y_step_um = None
        self.last_x_step_um = None
        self.last_xy_step_um = None
        self._last_geometry_vote_counts = None
        self._loaded_voxel_um = (None, None, None)

        self._cropped_stack_xy_raw = None
        self._cropped_stack_z_raw = None
        self.cropped_xyz = None
        self.image_name = None
        self._mask_central_region_enabled = False
        self.last_organoid_region = None
        self._cropped_organoid_mask_xy_raw = None


        self.images_output = TextEdit(value="")
        try:
            self.images_output.native.setReadOnly(True)
        except Exception:
            pass
        self.images_output.min_height = 120
        self.images_output.max_height = 300

        @magicgui(
            image_source={"label": "Folder/.tif/.lif", "mode": "r"},
            call_button="Load images",
        )
        def list_images(image_source = Path()):
            self.list_images(image_source)

        @magicgui(
            image_choice={"label": "Image", "choices": ["(load images)"], "widget_type": "ComboBox"},
            mask_central_region={"label": "Mask central region"},
            clear_layers={"label": "Clear viewer first"},
            call_button="Segment + View",
        )
        def segment_and_view(image_choice: str = "(load images)",
            mask_central_region: bool = False,
            clear_layers: bool = True,
        ):
            self.segment_and_view(
                image_choice=image_choice,
                focus_downsample=4,
                focus_n_sampling=10,
                focus_patch=50,
                mask_central_region=mask_central_region,
                clear_layers=clear_layers,
            )

        @magicgui(call_button="Create cropped aligned")
        def apply_crop():
            self.apply_crop_from_roi()

        @magicgui(
            auto_call=True,
            call_button=False,
            device_width_um={"label": "Device width (um)", "min": 0.0, "max": 1000.0, "step": 1.0},
        )
        def device_width_ok(device_width_um: float = 0.0):
            self.apply_device_width_layer(device_width_um)

        self.list_images_widget = list_images
        self.segment_and_view_widget = segment_and_view
        self.apply_crop = apply_crop
        self.device_width_ok = device_width_ok


        self.main_panel = Container(
            widgets=[
                self.list_images_widget,
                self.segment_and_view_widget,
                self.device_width_ok,
                self.apply_crop,
                self.images_output,
            ]
        )
        self.viewer.window.add_dock_widget(self.main_panel, area="right")

        self.update_segment_button()

    def fmt_um(self, value):
        try:
            vf = float(value)
            return f"{vf:.4g}" if np.isfinite(vf) else "NA"
        except Exception:
            return "NA"

    def voxel_log_text(self, z_um, y_um, x_um):
        return f"Voxel size (um): x={self.fmt_um(x_um)}, y={self.fmt_um(y_um)}, z={self.fmt_um(z_um)}"

    def reset_image_choices(self):
        self.image_choice_map = {}
        self.segment_and_view_widget.image_choice.choices = ["(load images)"]
        self.segment_and_view_widget.image_choice.value = "(load images)"

    def set_last_voxel_steps(self, z_um, y_um, x_um):
        self.last_z_step_um = z_um
        self.last_y_step_um = y_um
        self.last_x_step_um = x_um
        self.last_xy_step_um = np.nanmean([v for v in [x_um, y_um] if v is not None]) if (x_um is not None or y_um is not None) else None

    def run_automatic(
        self,
        image_source: Path,
        image_index: int = 0,
        device_width_um: float = 35.0,
        mask_central_region = False,
        failure_output_dir: Optional[Path] = None,
    ):
        source = Path(image_source)
        self.list_images(source)

        if not self.image_choice_map:
            raise ValueError("Could not load images for automatic segmentation.")

        selected_label = None
        for label, idx in self.image_choice_map.items():
            if int(idx) == int(image_index):
                selected_label = label
                break

        if selected_label is None:
            raise ValueError(f"image_index {image_index} is out of range for source: {source}")

        self.segment_and_view(
            image_choice=selected_label,
            focus_downsample=4,
            focus_n_sampling=10,
            focus_patch=50,
            mask_central_region=mask_central_region,
            clear_layers=True,
        )

        self.apply_device_width_layer(float(device_width_um))
        self.apply_crop_from_roi()

        if self.cropped_xyz is None:
            msg = str(getattr(self.images_output, "value", "Automatic segmentation failed."))
            if failure_output_dir is not None:
                fail_dir = Path(failure_output_dir)
                fail_dir.mkdir(parents=True, exist_ok=True)
                name = getattr(self, "image_name", None) or f"img{image_index}"
                # Compute stack height in um
                stack_height_um = "unknown"
                if self.last_stack is not None and self.last_z_step_um is not None:
                    stack_height_um = f"{self.last_stack.shape[0] * self.last_z_step_um:.1f}"
                elif self.last_stack is not None:
                    stack_height_um = f"{self.last_stack.shape[0]} slices (z step unknown)"
                txt_path = fail_dir / f"{name}_debug.txt"
                txt_path.write_text(
                    f"Image: {name}\n"
                    f"Source: {source}\n"
                    f"Image index: {image_index}\n"
                    f"Device width (um): {device_width_um}\n"
                    f"Mask central region: {mask_central_region}\n"
                    f"Stack height (um): {stack_height_um}\n\n"
                    f"Error: {msg}\n",
                    encoding="utf-8",
                )
            raise RuntimeError(msg)

        return self.get_cropped_outputs()

    def save_overlay_and_slice_tifs(self, name_prefix: str, run_suffix: int, output_dir: Optional[Path] = None):
        if self.last_image is None:
            return None, None

        out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        base = np.asarray(self.last_image)
        if base.ndim == 3:
            base = np.mean(base, axis=-1)
        base_u8 = self.scale_to_uint8_view(base)

        overlay = np.stack([base_u8, base_u8, base_u8], axis=-1)

        def draw_mask(mask_xy, rgb=(0, 255, 255), alpha=0.55):
            if mask_xy is None:
                return
            mask_xy = np.asarray(mask_xy, dtype=bool)
            if mask_xy.size == 0:
                return
            if mask_xy.shape != overlay.shape[:2]:
                return
            if not np.any(mask_xy):
                return

            base_float = overlay[mask_xy].astype(np.float32)
            tint = np.array(rgb, dtype=np.float32)
            blended = (1.0 - float(alpha)) * base_float + float(alpha) * tint
            overlay[mask_xy] = np.clip(blended, 0, 255).astype(np.uint8)

            # Add a strong contour so the region is visible on bright backgrounds.
            try:
                eroded = binary_erosion(mask_xy, structure=np.ones((3, 3), dtype=bool))
                boundary = mask_xy & ~eroded
                overlay[boundary, 0] = int(rgb[0])
                overlay[boundary, 1] = int(rgb[1])
                overlay[boundary, 2] = int(rgb[2])
            except Exception:
                pass

        def draw_corners(corners_yx, rgb):
            if corners_yx is None:
                return
            corners_yx = np.asarray(corners_yx)
            if corners_yx.shape[0] < 4:
                return
            corners_yx = corners_yx.astype(int)
            for i in range(len(corners_yx)):
                y0, x0 = corners_yx[i]
                y1, x1 = corners_yx[(i + 1) % len(corners_yx)]
                rr, cc = line(int(y0), int(x0), int(y1), int(x1))
                rr = np.clip(rr, 0, overlay.shape[0] - 1)
                cc = np.clip(cc, 0, overlay.shape[1] - 1)
                overlay[rr, cc, 0] = int(rgb[0])
                overlay[rr, cc, 1] = int(rgb[1])
                overlay[rr, cc, 2] = int(rgb[2])

        inner_corners_yx = None
        if self._roi_layer is not None and self._roi_layer in self.viewer.layers and len(getattr(self._roi_layer, "data", [])) > 0:
            inner_corners_yx = np.asarray(self._roi_layer.data[0])

        outer_corners_yx = None
        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers and len(getattr(self._roi_outer_layer, "data", [])) > 0:
            outer_corners_yx = np.asarray(self._roi_outer_layer.data[0])

        draw_corners(inner_corners_yx, (255, 0, 0))
        draw_corners(outer_corners_yx, (255, 255, 0))

        # Prefer cached mask, fallback to debug snapshot if needed.
        organoid_mask = self.last_organoid_region
        if organoid_mask is None and self.last_segment_debug is not None:
            organoid_mask = self.last_segment_debug.get("organoid_region", None)
        draw_mask(organoid_mask, rgb=(0, 255, 255), alpha=0.55)

        hough_tag = "_hough" if self._hough_fallback_used else ""
        overlay_path = out_dir / f"{name_prefix}_overlay_geometry{hough_tag}_{int(run_suffix)}.tif"
        tifffile.imwrite(str(overlay_path), overlay)

        return overlay_path

    # -------- Focus helpers (integrated; no separate class) --------
    def to_gray(self, im):
        if im.ndim == 2:
            return im
        return np.mean(im, axis=-1)

    def focus_score(self, patch):
        p = np.asarray(self.to_gray(patch), dtype=float)
        return float(np.std(sobel(p)))

    def curved_plane_refocus(self, stack_zyx, grid=20, patch=50, mask=None):
        Z, H, W = stack_zyx.shape
        ys = np.linspace(patch // 2, H - patch // 2 - 1, grid).astype(int)
        xs = np.linspace(patch // 2, W - patch // 2 - 1, grid).astype(int)

        pts, zs = [], []
        for y in ys:
            for x in xs:
                if mask is not None and not mask[y, x]:
                    continue
                sl = (slice(y - patch // 2, y + patch // 2), slice(x - patch // 2, x + patch // 2))
                f = np.array([self.focus_score(stack_zyx[z][sl]) for z in range(Z)], dtype=np.float32)
                pts.append((x, y))
                zs.append(int(np.argmax(f)))

        if len(zs) < 6:
            scores = []
            for z in range(Z):
                sl = self.to_gray(stack_zyx[z])
                if mask is not None:
                    sl = sl * mask
                scores.append(np.std(sobel(sl)))
            z0 = int(np.argmax(scores))
            img = self.to_gray(stack_zyx[z0])
            zmap = np.full(stack_zyx.shape[1:], z0, dtype=np.int16)
            return img, zmap, np.array(pts), np.array(zs)

        pts = np.array(pts, dtype=np.float32)
        zs = np.array(zs, dtype=np.float32)

        Xc = pts[:, 0] - pts[:, 0].mean()
        Yc = pts[:, 1] - pts[:, 1].mean()
        scale = max(W, H)
        Xn = Xc / scale
        Yn = Yc / scale

        B = np.column_stack((Xn**2, Yn**2, Xn * Yn, Xn, Yn, np.ones_like(Xn)))
        coeffs, *_ = np.linalg.lstsq(B, zs, rcond=None)

        Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
        mean_x = pts[:, 0].mean()
        mean_y = pts[:, 1].mean()
        Xg_n = (Xg - mean_x) / scale
        Yg_n = (Yg - mean_y) / scale
        zmap = (
            coeffs[0] * Xg_n**2
            + coeffs[1] * Yg_n**2
            + coeffs[2] * Xg_n * Yg_n
            + coeffs[3] * Xg_n
            + coeffs[4] * Yg_n
            + coeffs[5]
        )
        zmap = np.clip(np.rint(zmap).astype(np.int16), 0, Z - 1)
        img = np.take_along_axis(stack_zyx, zmap[None, :, :], axis=0)[0]
        return img, zmap, pts, zs
    def compute_focus_plane_from_stack(self, stack: Optional[np.ndarray], downsample: int, n_sampling: int, patch: int, source_is_lif: bool = False, image_index: Optional[int] = None):
        # Unified focus path: if LIF, load + normalize here; elif use provided stack.
        if source_is_lif:
            if self._selected_lif is None:
                raise ValueError("No LIF file selected")
            with LifFile(self._selected_lif) as lif:
                idx = int(image_index if image_index is not None else 0)
                img = lif.images[idx]
                image = img.asarray()
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = arr[np.newaxis, ...]
            elif arr.ndim == 3:
                arr = arr if arr.shape[0] < 64 else arr[np.newaxis, ...]
            elif arr.ndim == 4:
                # Multi-channel: channel dim has size < 4; z is always > 5
                ch_candidates = [i for i in range(arr.ndim) if arr.shape[i] < 4]
                ch_axis = ch_candidates[0] if ch_candidates else int(np.argmin(arr.shape))
                ch_idx = min(self.channel, arr.shape[ch_axis] - 1)
                arr = np.take(arr, ch_idx, axis=ch_axis)
            else:
                raise ValueError(f"Unsupported LIF array shape: {arr.shape}")

            self.last_stack = arr.astype(np.float32)
            z_um, y_um, x_um = read_voxel_size_um(
                self._selected_lif,
                source_is_lif=True,
                selected_lif=self._selected_lif,
                image_index=idx,
            )
            self.last_z_step_um = z_um
            self.last_y_step_um = y_um
            self.last_x_step_um = x_um
            self.last_xy_step_um = np.nanmean([v for v in [x_um, y_um] if v is not None]) if (x_um is not None or y_um is not None) else None
        else:
            arr = np.asarray(stack)

        ds = max(1, int(downsample))
        self.last_focus_downsample = ds

        stack_score_full = np.mean(arr, axis=-1) if arr.ndim == 4 else arr
        focus_full, zmap_full, _, _ = self.curved_plane_refocus(
            stack_score_full,
            grid=int(n_sampling),
            patch=int(patch),
            mask=None,
        )
        self.last_focus_zmap_full = np.asarray(zmap_full, dtype=np.int16) if zmap_full is not None else None

        if ds > 1:
            focus_out = focus_full[::ds, ::ds]
        else:
            focus_out = focus_full

        return focus_out

    def refocus_stack_around_plane(self, stack: np.ndarray, zmap: np.ndarray):
        if stack is None or zmap is None:
            return stack

        arr = np.asarray(stack)
        if arr.ndim not in (3, 4):
            return arr

        Z = arr.shape[0]
        H, W = arr.shape[1], arr.shape[2]
        zm = np.asarray(zmap, dtype=np.int32)
        if zm.shape != (H, W):
            return arr

        z_ref = int(np.clip(np.rint(np.median(zm)), 0, Z - 1))
        offsets = (np.arange(Z, dtype=np.int32) - z_ref)[:, None, None]
        src = np.clip(zm[None, :, :] + offsets, 0, Z - 1)

        if arr.ndim == 3:
            return np.take_along_axis(arr, src, axis=0)

        out = np.empty_like(arr)
        for c in range(arr.shape[-1]):
            out[..., c] = np.take_along_axis(arr[..., c], src, axis=0)
        return out

    # -------- segmentation + geometry --------
    def is_color_image_2d(self, arr: np.ndarray) -> bool:
        return arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 32 and arr.shape[1] > 32

    def scale_to_uint8_view(self, arr):
        if arr is None:
            return None
        data = np.asarray(arr)
        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            return np.zeros_like(data, dtype=np.uint8)
        scaled = np.clip((data - dmin) / (dmax - dmin), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def order_corners_clockwise(self, corners_xy: np.ndarray):
        c = np.asarray(corners_xy, dtype=float)
        centroid = c.mean(axis=0)
        angles = np.arctan2(c[:, 1] - centroid[1], c[:, 0] - centroid[0])
        c = c[np.argsort(angles)]
        start = np.argmin(c[:, 0] + c[:, 1])
        return np.roll(c, -start, axis=0)

    def crop_rectified_from_corners(self, in_focus_plane: np.ndarray, corners_xy: np.ndarray):
        if corners_xy is None:
            return None
        c = self.order_corners_clockwise(corners_xy)
        w1 = np.hypot(c[1, 0] - c[0, 0], c[1, 1] - c[0, 1])
        w2 = np.hypot(c[2, 0] - c[3, 0], c[2, 1] - c[3, 1])
        h1 = np.hypot(c[3, 0] - c[0, 0], c[3, 1] - c[0, 1])
        h2 = np.hypot(c[2, 0] - c[1, 0], c[2, 1] - c[1, 1])
        width = int(max(w1, w2))
        height = int(max(h1, h2))
        if width <= 1 or height <= 1:
            return None

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
        tform = ProjectiveTransform()
        if not tform.estimate(dst, c):
            return None
        warped = warp(in_focus_plane, tform, output_shape=(height, width), order=1, preserve_range=True)
        return warped.astype(in_focus_plane.dtype)

    def crop_rectified_stack_from_corners(self, stack: np.ndarray, corners_xy: np.ndarray):
        # C3: delegate to the unified crop_rectified method
        return self.crop_rectified(stack, corners_xy)

    def crop_rectified(self, data: np.ndarray, corners_xy: np.ndarray):
        """C3: unified crop for 2-D planes, 3-D stacks, and 4-D stacks (z, h, w, c).

        Replaces the separate crop_rectified_stack_from_corners path; the 2-D
        crop_rectified_from_corners still handles the single-plane case directly
        for clarity.
        """
        if corners_xy is None or data is None:
            return None
        c = self.order_corners_clockwise(corners_xy)
        w1 = np.hypot(c[1, 0] - c[0, 0], c[1, 1] - c[0, 1])
        w2 = np.hypot(c[2, 0] - c[3, 0], c[2, 1] - c[3, 1])
        h1 = np.hypot(c[3, 0] - c[0, 0], c[3, 1] - c[0, 1])
        h2 = np.hypot(c[2, 0] - c[1, 0], c[2, 1] - c[1, 1])
        width = int(max(w1, w2))
        height = int(max(h1, h2))
        if width <= 1 or height <= 1:
            return None

        dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
        tform = ProjectiveTransform()
        if not tform.estimate(dst, c):
            return None

        if data.ndim == 2:
            warped = warp(data, tform, output_shape=(height, width), order=1, preserve_range=True)
        elif data.ndim == 3:
            warped = np.stack(
                [warp(data[z], tform, output_shape=(height, width), order=1, preserve_range=True)
                 for z in range(data.shape[0])],
                axis=0,
            )
        elif data.ndim == 4:
            warped = np.stack(
                [warp(data[z], tform, output_shape=(height, width), order=1,
                      preserve_range=True, channel_axis=-1)
                 for z in range(data.shape[0])],
                axis=0,
            )
        else:
            return None

        return warped.astype(data.dtype)

    def signed_orientation(self, region):
        img = binary_fill_holes(region.image).astype(float)
        mu = moments_central(img)
        angle_rad = 0.5 * np.arctan2(2 * mu[1, 1], mu[2, 0] - mu[0, 2])
        return np.rad2deg(angle_rad)

    def corners_touch_border(self, corners_xy: np.ndarray, shape, margin=0):
        H, W = shape
        x = corners_xy[:, 0]
        y = corners_xy[:, 1]
        return (x <= margin).any() or (x >= (W - 1 - margin)).any() or (y <= margin).any() or (y >= (H - 1 - margin)).any()

    def mask_out_organoid(self, in_focus_plane, mode: str = "dark"):
        """Segment the central organoid region.

        Parameters
        ----------
        mode : "dark" | "light"
            "light" — Yen threshold on the smoothed image, label + expand.
                      If area is 1–15 % → accept as organoid mask.
                      Otherwise fall through to the dark path.
            "dark" (or failed light) —
                      Yen threshold on the *inverted* smoothed image.
                      If we came from a failed light step, the binary is the
                      union of the light region and the dark Yen region.
                      If area >15 %, retry with Otsu on the inverted image.
            In all cases the most-central reasonably-sized label is kept.
        """
        raw = np.asarray(in_focus_plane, dtype=np.float32)
        smoothed = gaussian(raw, sigma=float(self.mask_sigma)).astype(np.float32)

        H, W = in_focus_plane.shape[:2]
        xy_area = H * W
        cyi, cxi = H // 2, W // 2
        r = int(min(H, W) * 0.1)
        yy, xx = np.ogrid[:H, :W]
        central_roi = (yy - cyi) ** 2 + (xx - cxi) ** 2 <= r**2

        smoothed = np.nan_to_num(smoothed, nan=0.0, posinf=0.0, neginf=0.0)

        def score(p):
            overlap = np.sum(labelled[central_roi] == p.label)
            py, px = p.centroid
            dist2 = (py - cyi) ** 2 + (px - cxi) ** 2
            return (-overlap, dist2)

        # ------------------------------------------------------------------
        # LIGHT path: Yen threshold on the smoothed (non-inverted) image
        # ------------------------------------------------------------------
        light_ok = False
        light_region_mask = None
        union_light_with_dark = False
        if mode == "light":
            thresh = threshold_yen(smoothed)
            labelled = label(smoothed > thresh)
            labelled = expand_labels(labelled, distance=20)
            props = regionprops(labelled)

            if len(props) > 0:
                best_prop = min(props, key=score)
                area_frac = best_prop.area / xy_area
                light_region_mask = (labelled == best_prop.label)

                if 0.02 < area_frac <= 0.15:
                    light_ok = True
                elif 0.01 <= area_frac <= 0.02:
                    # Small but real central object — union with dark path
                    union_light_with_dark = True

        # ------------------------------------------------------------------
        # DARK path (also used when light fails): Yen on inverted image
        # ------------------------------------------------------------------
        if not light_ok:
            inverted = gaussian(util.invert(raw), sigma=float(self.mask_sigma)).astype(np.float32)
            inverted = np.nan_to_num(inverted, nan=0.0, posinf=0.0, neginf=0.0)

            thresh = threshold_yen(inverted)
            hole_cutoff = int(0.015 * xy_area)
            dark_binary = remove_small_holes(inverted > thresh, area_threshold=hole_cutoff)
            labelled = label(dark_binary)
            props = regionprops(labelled)
            if len(props) == 0:
                return np.zeros((H, W), dtype=bool)

            best_prop = min(props, key=score)
            area_frac = best_prop.area / xy_area

            # If Yen result is out of range, retry with Otsu
            if area_frac < 0.02 or area_frac > 0.15:
                thresh = threshold_otsu(inverted)
                otsu_binary = remove_small_holes(inverted > thresh, area_threshold=hole_cutoff)
                labelled = label(otsu_binary)
                props = regionprops(labelled)
                if len(props) == 0:
                    return np.zeros((H, W), dtype=bool)
                best_prop = min(props, key=score)
                area_frac_otsu = best_prop.area / xy_area

                # If Otsu region is too large, try eroding and re-scoring
                if area_frac_otsu > 0.15:
                    eroded_binary = binary_erosion(otsu_binary, structure=disk(20))
                    labelled_eroded = expand_labels(label(eroded_binary), distance = 20)
                    props_eroded = regionprops(labelled_eroded)
                    if props_eroded:
                        best_eroded = min(props_eroded, key=score)
                        area_frac_eroded = best_eroded.area / xy_area
                        if 0.02 <= area_frac_eroded <= 0.15:
                            labelled = labelled_eroded
                            best_prop = best_eroded
                            area_frac_otsu = area_frac_eroded

                # If Otsu (or eroded Otsu) still out of range, try threshold_minimum
                if area_frac_otsu < 0.02 or area_frac_otsu > 0.15:
                    # threshold_minimum can hang indefinitely in C code —
                    # run it in a subprocess with a hard kill timeout.
                    import multiprocessing as _mp_min, tempfile as _tf_min
                    _tmp_min = _tf_min.mktemp(suffix=".npy")
                    _proc_min = _mp_min.Process(
                        target=mp_run_threshold_minimum,
                        args=(inverted, _tmp_min),
                    )
                    _proc_min.start()
                    _proc_min.join(timeout=30)
                    if _proc_min.is_alive():
                        _proc_min.kill()
                        _proc_min.join(timeout=5)
                        thresh = threshold_otsu(inverted)
                    elif _proc_min.exitcode != 0:
                        thresh = threshold_otsu(inverted)
                    else:
                        thresh = float(np.load(_tmp_min)[0])
                    try:
                        import os as _os_min
                        _os_min.remove(_tmp_min)
                    except OSError:
                        pass
                    min_binary = remove_small_holes(inverted > thresh, area_threshold=hole_cutoff)
                    labelled = label(min_binary)
                    props = regionprops(labelled)
                    if len(props) == 0:
                        return np.zeros((H, W), dtype=bool)
                    best_prop = min(props, key=score)
                    area_frac_min = best_prop.area / xy_area

                    if area_frac_min < 0.02 or area_frac_min > 0.15:
                        raise RuntimeError(
                            f"Organoid segmentation failed: all thresholds out of range "
                            f"(Yen {area_frac:.1%}, Otsu {area_frac_otsu:.1%}, "
                            f"minimum {area_frac_min:.1%}) — skipping image"
                        )

        organoid_region = labelled == best_prop.label

        # Union the dark result with the small light region if a 1–2 % central
        # light object was detected in the first pass, but only when they
        # are physically adjacent (dilate one by 1 px and check overlap).
        if union_light_with_dark and light_region_mask is not None:
            touching = np.any(binary_dilation(light_region_mask) & organoid_region)
            if touching:
                organoid_region = organoid_region | light_region_mask

        organoid_region = remove_small_holes(organoid_region, area_threshold=10000)
        organoid_region = closing(organoid_region, disk(5))
        return organoid_region

    def oriented_rect_corners_crop_necks_and_flares(self, mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None, None, None

        cx, cy = xs.mean(), ys.mean()
        centroid_xy = np.array([cx, cy], float)

        mu = moments_central(binary_fill_holes(mask).astype(np.uint8))
        angle_rad = 0.5 * np.arctan2(2 * mu[1, 1], (mu[0, 2] - mu[2, 0]))

        c, s = np.cos(angle_rad), np.sin(angle_rad)
        dx = xs - cx
        dy = ys - cy
        u = c * dx + s * dy
        v = -s * dx + c * dy

        u_span = float(u.max() - u.min())
        v_span = float(v.max() - v.min())
        if v_span >= u_span:
            long, short, long_name = v, u, "v"
        else:
            long, short, long_name = u, v, "u"

        short_min_full = float(short.min())
        short_max_full = float(short.max())
        long_bin = np.floor(long / self.bin_size).astype(int)
        bins, inv = np.unique(long_bin, return_inverse=True)

        short_min = np.full(bins.shape, np.inf)
        short_max = np.full(bins.shape, -np.inf)
        np.minimum.at(short_min, inv, short)
        np.maximum.at(short_max, inv, short)
        widths = short_max - short_min

        if self.smooth_window and self.smooth_window > 1:
            w = int(self.smooth_window)
            if w % 2 == 0:
                w += 1
            pad = w // 2
            widths_s = np.convolve(np.pad(widths, (pad, pad), mode="edge"), np.ones(w) / w, mode="valid")
        else:
            widths_s = widths

        typical_width = float(np.percentile(widths_s, self.typical_pct))
        low_thr = self.low_frac * typical_width
        keep = widths_s >= low_thr

        best_start = best_end = None
        best_len = 0
        cur_start = cur_end = None
        for i in range(len(bins)):
            if not keep[i]:
                if cur_start is not None:
                    L = cur_end - cur_start + 1
                    if L > best_len:
                        best_len = L
                        best_start, best_end = cur_start, cur_end
                    cur_start = cur_end = None
                continue
            if cur_start is None:
                cur_start = cur_end = bins[i]
            elif bins[i] == cur_end + 1:
                cur_end = bins[i]
            else:
                L = cur_end - cur_start + 1
                if L > best_len:
                    best_len = L
                    best_start, best_end = cur_start, cur_end
                cur_start = cur_end = bins[i]

        if cur_start is not None:
            L = cur_end - cur_start + 1
            if L > best_len:
                best_len = L
                best_start, best_end = cur_start, cur_end

        total_len_bins = int(bins.max() - bins.min() + 1)
        min_run_bins = int(np.ceil(self.min_run_frac * total_len_bins))

        if best_start is None or best_len < min_run_bins:
            long_min = float(long.min())
            long_max = float(long.max())
        else:
            long_min = float(best_start * self.bin_size)
            long_max = float((best_end + 1) * self.bin_size)

        in_long_band = (long >= long_min) & (long <= long_max)
        short_min_use = float(short[in_long_band].min()) if in_long_band.any() else short_min_full
        short_max_use = float(short[in_long_band].max()) if in_long_band.any() else short_max_full

        if long_name == "v":
            umin, umax = short_min_use, short_max_use
            vmin, vmax = long_min, long_max
        else:
            umin, umax = long_min, long_max
            vmin, vmax = short_min_use, short_max_use

        corners_uv = np.array([[umin, vmin], [umax, vmin], [umax, vmax], [umin, vmax]], dtype=float)
        x = cx + c * corners_uv[:, 0] - s * corners_uv[:, 1]
        y = cy + s * corners_uv[:, 0] + c * corners_uv[:, 1]
        return np.stack([x, y], axis=1), angle_rad, centroid_xy

    def segment_from_plane(self, in_focus_plane: np.ndarray, mask_central_region, return_debug: bool):
        flag = False
        # Normalise legacy bool True → "dark" (original invert-based behaviour)
        if mask_central_region is True:
            mask_central_region = "dark"
        in_focus_plane = self.to_gray(in_focus_plane)

        median_thresholded = median(np.asarray(in_focus_plane, dtype=np.float32), footprint=disk(7)).astype(np.float32)
        sobel_operated = sobel(median_thresholded).astype(np.float32)
        thresh = threshold_triangle(sobel_operated)
        binary = sobel_operated > thresh

        h, w = in_focus_plane.shape[:2]
        binary[h // 3 : 2 * (h // 3), w // 3 : 2 * (w // 3)] = 0

        organoid_region = None
        if mask_central_region:
            try:
                organoid_region = self.mask_out_organoid(in_focus_plane, mode=mask_central_region)
            except RuntimeError:
                organoid_region = None
            if organoid_region is not None:
                # Use an expanded organoid mask for device segmentation only —
                # removes organoid-proximal artefacts that disrupt edge detection.
                organoid_mask_for_device = expand_labels(
                    label(organoid_region), distance=50
                ).astype(bool)
                binary[organoid_mask_for_device] = 0

        labels = label(binary)
        data = regionprops_table(labels, binary, properties=("label", "area", "eccentricity"))
        condition = (data["area"] > 100) & (data["eccentricity"] > 0.5)
        labels_to_dilate = util.map_array(labels, data["label"], data["label"] * condition)

        dilated_output = np.zeros_like(labels, dtype=np.uint8)
        base_selem = np.zeros((71, 71), dtype=bool)
        base_selem[35, :] = 1
        pad = base_selem.shape[0] // 2

        for region in regionprops(labels_to_dilate):
            angle_to_rotate = self.signed_orientation(region)
            rotated_selem = rotate(base_selem.astype(float), angle=90 + angle_to_rotate, reshape=False, order=0) > 0.5

            minr, minc, maxr, maxc = region.bbox
            r0 = max(minr - pad, 0)
            r1 = min(maxr + pad, labels_to_dilate.shape[0])
            c0 = max(minc - pad, 0)
            c1 = min(maxc + pad, labels_to_dilate.shape[1])

            mask_roi = labels_to_dilate[r0:r1, c0:c1] == region.label
            dilated = binary_dilation(mask_roi.astype(bool), structure=rotated_selem.astype(bool))
            dilated_output[r0:r1, c0:c1][dilated] = 255

        post_dilation_mask = np.logical_or(dilated_output, binary)
        clean_labels = label(~post_dilation_mask)
        props = regionprops(clean_labels)
        largest_prop = max(props, key=lambda p: p.area)
        device_mask = clean_labels == largest_prop.label

        edges = reconstructed = reconstructed_mask = None
        new_corners = new_angle_rad = new_centroid_xy = None
        rescue_closed_mask = None
        rescue_radius = None
        hough_attempts_log = []

        corners, angle_rad, centroid_xy = self.oriented_rect_corners_crop_necks_and_flares(device_mask)
        # Check if device mask itself touches the image border (more reliable
        # than only checking the oriented-rectangle corners, which can be
        # pulled inward by neck/flare cropping).
        h, w = device_mask.shape
        mask_touches_border = (
            device_mask[0, :].any() or device_mask[-1, :].any() or
            device_mask[:, 0].any() or device_mask[:, -1].any()
        )
        if corners is None or self.corners_touch_border(corners, device_mask.shape, margin=5) or mask_touches_border:
            flag = True
            if corners is None:
                reason = "corners is None"
            elif mask_touches_border:
                reason = "device mask touches image border"
            else:
                reason = "corners touch border"

            # Step 1: Dilation rescue — seal small border gaps with progressively
            # larger disk structuring elements before falling back to Hough.
            # If dilating post_dilation_mask produces a region that is fully
            # interior (doesn't touch any image border), use it as the device.
            rescued = False
            for dil_radius in (3, 6, 10):
                closed = binary_dilation(post_dilation_mask.astype(bool),
                                         structure=disk(dil_radius))
                clab = label(~closed)
                cprops = regionprops(clab)
                if not cprops:
                    continue
                ch, cw = closed.shape
                non_border = [
                    p for p in cprops
                    if p.bbox[0] > 0 and p.bbox[1] > 0
                    and p.bbox[2] < ch and p.bbox[3] < cw
                ]
                if not non_border:
                    continue
                best_c = max(non_border, key=lambda p: p.area)
                rescue_mask = clab == best_c.label
                rc, ra, rce = self.oriented_rect_corners_crop_necks_and_flares(rescue_mask)
                area_fraction = best_c.area / (ch * cw)
                if rc is not None and not self.corners_touch_border(rc, rescue_mask.shape, margin=5):
                    if area_fraction < 0.40:
                        break
                    new_corners, new_angle_rad, new_centroid_xy = rc, ra, rce
                    rescue_closed_mask = closed
                    rescue_radius = dil_radius
                    rescued = True
                    break

            # Step 2: Hough fallback — only if dilation rescue failed.
            #         Retry with +100 line_length/line_gap if corners still touch borders.
            if not rescued:
                self._hough_fallback_used = True
                edges = post_dilation_mask

                ll = self.line_length
                lg = self.line_gap
                lt = self.hough_threshold
        
                for hough_attempt in range(4):
                    if hough_attempt > 0:
                        ll += 0
                        lg += 100
                        lt += 5

                    segs = probabilistic_hough_line(
                        edges,
                        line_length=ll,
                        line_gap=lg,
                        threshold=lt,
                    )
                    reconstructed = np.zeros_like(edges, dtype=bool)
                    for (x0, y0), (x1, y1) in segs:
                        rr, cc = line(y0, x0, y1, x1)
                        reconstructed[rr, cc] = True
                    # Dilate lines so they are thick enough to sever regions
                    # under 8-connectivity labelling.
                    reconstructed = binary_dilation(reconstructed, structure=disk(1))

                    reconstructed_mask = np.logical_or(reconstructed, post_dilation_mask)
                    updated_clean_labels = label(~reconstructed_mask)
                    props = regionprops(updated_clean_labels)
                    largest_prop = max(props, key=lambda p: p.area)
                    new_device_mask = updated_clean_labels == largest_prop.label
                    new_corners, new_angle_rad, new_centroid_xy = self.oriented_rect_corners_crop_necks_and_flares(new_device_mask)

                    # Diagnostic checks for this attempt
                    h_dm, w_dm = new_device_mask.shape
                    mask_touches = (
                        new_device_mask[0, :].any() or new_device_mask[-1, :].any() or
                        new_device_mask[:, 0].any() or new_device_mask[:, -1].any()
                    )
                    corners_touch = (
                        new_corners is not None and
                        self.corners_touch_border(new_corners, new_device_mask.shape, margin=5)
                    )
                    device_area_frac = float(new_device_mask.sum()) / (h_dm * w_dm)
                    n_regions = len(props)

                    hough_attempts_log.append({
                        "attempt": hough_attempt,
                        "line_length": ll,
                        "line_gap": lg,
                        "threshold": lt,
                        "n_hough_segments": len(segs),
                        "reconstructed": reconstructed.copy(),
                        "reconstructed_mask": reconstructed_mask.copy(),
                        "device_mask": new_device_mask.copy(),
                        "corners": new_corners,
                        "corners_touch_border": corners_touch,
                        "mask_touches_border": mask_touches,
                        "device_area_frac": device_area_frac,
                        "n_regions": n_regions,
                    })

                    if new_corners is not None and not corners_touch:
                        break

        if flag:
            final_corners = new_corners
            final_angle_rad = new_angle_rad
            final_centroid_xy = new_centroid_xy
        else:
            final_corners = corners
            final_angle_rad = angle_rad
            final_centroid_xy = centroid_xy

        # Fallback: if no device was found at all, use the entire image as the device
        if final_corners is None:
            h, w = in_focus_plane.shape[:2]
            final_corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=float)
            final_angle_rad = 0.0
            final_centroid_xy = np.array([w / 2.0, h / 2.0])

        cropped_rotated = self.crop_rectified_from_corners(in_focus_plane, final_corners)

        if return_debug:
            debug = {
                "median_thresholded": median_thresholded,
                "sobel_operated": sobel_operated,
                "binary": binary,
                "labels_to_dilate": labels_to_dilate,
                "post_dilation_mask": post_dilation_mask,
                "device_mask": device_mask,
                "organoid_region": organoid_region,
                "edges": edges,
                "reconstructed": reconstructed,
                "reconstructed_mask": reconstructed_mask,
                "rescue_closed_mask": rescue_closed_mask,
                "rescue_radius": rescue_radius,
                "hough_attempts": hough_attempts_log if hough_attempts_log else None,
                "flag": flag,
                "final_corners": final_corners,
                "final_centroid_xy": final_centroid_xy,
                "final_angle_rad": final_angle_rad,
                "new_corners": new_corners,
                "new_centroid_xy": new_centroid_xy,
                "new_angle_rad": new_angle_rad,
                "cropped_rotated": cropped_rotated,
                "gpu_used_preprocess": False,
                "gpu_used_dilation": False,
            }
            return in_focus_plane, organoid_region, final_corners, cropped_rotated, debug

        return in_focus_plane, organoid_region, final_corners, cropped_rotated

    # -------- IO / GUI flow --------
    def list_images(self, image_source: Path):
        p = Path(image_source)
        if not p.exists():
            self.images_output.value = "[WARN] Please select a folder, a .tif/.tiff file, or a .lif file."
            return

        self._selected_image_folder = None
        self._selected_lif = None
        self.image_paths = []
        self.last_stack = None
        self.last_focus_downsample = 1
        choices = []
        choice_map = {}

        if p.is_dir():
            self._selected_image_folder = p
            image_files = sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in (".tif", ".tiff")])
            if not image_files:
                self.images_output.value = "[WARN] No .tif/.tiff images found in the selected folder."
                self.reset_image_choices()
                self.update_segment_button()
                return
            self.image_paths = image_files
            for i, pth in enumerate(image_files):
                label_txt = f"{i}: {pth.name}"
                choices.append(label_txt)
                choice_map[label_txt] = i
            self.image_choice_map = choice_map
            self.segment_and_view_widget.image_choice.choices = choices
            self.segment_and_view_widget.image_choice.value = choices[0]
            z_um, y_um, x_um = read_voxel_size_um(image_files[0], source_is_lif=False)
            self._loaded_voxel_um = (z_um, y_um, x_um)
            self.images_output.value = f"[OK] Found {len(choices)} images in folder. Select one and click Segment + View. {self.voxel_log_text(z_um, y_um, x_um)}"
            self.update_segment_button()
            return

        if p.suffix.lower() == ".lif":
            self._selected_lif = p
            try:
                with LifFile(self._selected_lif) as lif:
                    for i, img in enumerate(lif.images):
                        name = img.name if hasattr(img, "name") and img.name else f"image_{i}"
                        label_txt = f"{i}: {name}"
                        choices.append(label_txt)
                        choice_map[label_txt] = i
            except Exception:
                self.images_output.value = "[ERROR] Unable to read .lif contents."
                self.reset_image_choices()
                self.update_segment_button()
                return

            if not choices:
                self.images_output.value = "[WARN] No readable images found inside the selected .lif file."
                self.reset_image_choices()
                self.update_segment_button()
                return

            self.image_choice_map = choice_map
            self.segment_and_view_widget.image_choice.choices = choices
            self.segment_and_view_widget.image_choice.value = choices[0]
            first_idx = choice_map[choices[0]]
            z_um, y_um, x_um = read_voxel_size_um(
                self._selected_lif,
                source_is_lif=True,
                selected_lif=self._selected_lif,
                image_index=first_idx,
            )
            self._loaded_voxel_um = (z_um, y_um, x_um)
            self.images_output.value = f"[OK] Found {len(choices)} images in .lif. Select one and click Segment + View. {self.voxel_log_text(z_um, y_um, x_um)}"
            self.update_segment_button()
            return

        if p.suffix.lower() in (".tif", ".tiff"):
            self.image_paths = [p]
            label_txt = f"0: {p.name}"
            self.image_choice_map = {label_txt: 0}
            self.segment_and_view_widget.image_choice.choices = [label_txt]
            self.segment_and_view_widget.image_choice.value = label_txt
            z_um, y_um, x_um = read_voxel_size_um(p, source_is_lif=False)
            self._loaded_voxel_um = (z_um, y_um, x_um)
            self.images_output.value = f"[OK] Loaded single .tif file. Select it and click Segment + View. {self.voxel_log_text(z_um, y_um, x_um)}"
            self.update_segment_button()
            return

        self.images_output.value = "[WARN] Unsupported selection. Choose a folder, a .tif/.tiff file, or a .lif file."
        self.reset_image_choices()
        self.update_segment_button()

    def segment_and_view(
        self,
        image_choice: str,
        focus_downsample: int,
        focus_n_sampling: int,
        focus_patch: int,
        mask_central_region,
        clear_layers: bool,
    ):
        source_is_lif = bool(getattr(self, "_selected_lif", None) and self._selected_lif.exists())

        if not source_is_lif and not self.image_paths:
            self.images_output.value = "[WARN] Select a folder/.tif/.lif and click 'Load images' first."
            return
        if not self.image_choice_map:
            self.images_output.value = "[WARN] Click 'Load images' to populate the dropdown."
            return

        image_index = self.image_choice_map.get(image_choice)
        if image_index is None:
            self.images_output.value = "[WARN] Please select an image from the dropdown."
            return

        label = str(image_choice)
        if ":" in label:
            label = label.split(":", 1)[1].strip()
        self.image_name = Path(label).stem

        self.last_focus_n_sampling = int(focus_n_sampling)
        self.last_focus_patch = int(focus_patch)

        try:
            if source_is_lif:
                in_focus_plane = self.compute_focus_plane_from_stack(
                    stack=None,
                    downsample=focus_downsample,
                    n_sampling=focus_n_sampling,
                    patch=focus_patch,
                    source_is_lif=True,
                    image_index=image_index,
                )
                in_focus_plane, organoid_region, final_corners, _, debug = self.segment_from_plane(
                    in_focus_plane,
                    mask_central_region,
                    return_debug=True,
                )
            else:
                source_path = self.image_paths[image_index]
                arr = np.asarray(tifffile.imread(str(source_path)))

                # Multi-channel TIFF: extract the requested channel to get a 3-D stack
                if arr.ndim == 4:
                    # Channel dim has size < 4; z is always > 5
                    ch_candidates = [i for i in range(arr.ndim) if arr.shape[i] < 4]
                    ch_axis = ch_candidates[0] if ch_candidates else int(np.argmin(arr.shape))
                    ch_idx = min(self.channel, arr.shape[ch_axis] - 1)
                    arr = np.take(arr, ch_idx, axis=ch_axis)

                z_um, y_um, x_um = read_voxel_size_um(Path(source_path), source_is_lif=False)
                self.set_last_voxel_steps(z_um, y_um, x_um)

                if self.is_color_image_2d(arr) or arr.ndim < 3:
                    in_focus_plane = arr.astype(np.float32)
                    self.last_stack = None
                    self.last_focus_downsample = 1
                else:
                    stack = arr.astype(np.float32)
                    in_focus_plane = self.compute_focus_plane_from_stack(stack, focus_downsample, focus_n_sampling, focus_patch)
                    self.last_stack = stack
                    self.last_focus_downsample = max(1, int(focus_downsample))

                in_focus_plane, organoid_region, final_corners, _, debug = self.segment_from_plane(
                    in_focus_plane,
                    mask_central_region,
                    return_debug=True,
                )
        except Exception as e:
            self.images_output.value = f"[ERROR] Segmentation failed: {type(e).__name__}: {e}"
            # Preserve partial data so _save_failure_diagnostic can still
            # generate a PNG showing whatever was computed before the crash.
            try:
                self.last_image = in_focus_plane
            except NameError:
                pass
            return

        self.last_segment_debug = debug

        if clear_layers:
            self.viewer.layers.clear()
            self._roi_layer = None
            self._roi_outer_layer = None
            self._cropped_layer = None

        self.last_image = in_focus_plane
        self._active_device_width_um = 30.0
        self._last_geometry_vote_counts = None
        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers:
            self.viewer.layers.remove(self._roi_outer_layer)
        self._roi_outer_layer = None
        self._cropped_stack_xy_raw = None
        self._cropped_stack_z_raw = None
        self.cropped_xyz = None
        self._cropped_organoid_mask_xy_raw = None
        self._mask_central_region_enabled = bool(mask_central_region)
        self.last_organoid_region = organoid_region.astype(bool) if organoid_region is not None else None
        self.device_width_ok.device_width_um.enabled = True
        self.device_width_ok.device_width_um.value = 30.0


        self.add_layer_if_nonzero(in_focus_plane, name="original", layer_type="image")

        if mask_central_region and organoid_region is not None:
            organoid_layer = self.add_layer_if_nonzero(organoid_region.astype(np.uint8), name="organoid_region", layer_type="labels")
            if organoid_layer is not None:
                organoid_layer.opacity = 0.4

        force_roi = clear_layers or self._roi_layer is None or len(getattr(self._roi_layer, "data", [])) == 0
        self.set_roi_layer(final_corners, force=force_roi)
        self.update_outer_geometry_from_current_roi(self.device_width_ok.device_width_um.value, update_message=False)
        self.images_output.value = (
            "[OK] Segmentation complete. Outer geometry is shown at default Device width=30 um; adjust width as needed."
        )


    def set_roi_layer(self, corners_xy, force=False):
        if corners_xy is None:
            self.images_output.value = "[WARN] No corners found for ROI."
            return
        corners_yx = np.asarray(corners_xy)[:, ::-1]
        if self._roi_layer is None or self._roi_layer not in self.viewer.layers:
            self._roi_layer = self.viewer.add_shapes(name="geometry")
            try:
                self._roi_layer.events.data.connect(self.on_roi_layer_data_changed)
            except Exception:
                pass
        if force or len(self._roi_layer.data) == 0:
            self._roi_layer.data = [corners_yx]
        self._roi_layer.shape_type = ["rectangle"]
        self._roi_layer.edge_color = "red"
        self._roi_layer.face_color = "transparent"
        self._roi_layer.editable = True
        self._roi_layer.mode = "select"

    def on_roi_layer_data_changed(self, event=None):
        if self._syncing_outer_geometry:
            return
        if self._roi_outer_layer is None or self._roi_outer_layer not in self.viewer.layers:
            return

        width_um = self._active_device_width_um
        if width_um is None:
            try:
                width_um = float(self.device_width_ok.device_width_um.value)
            except Exception:
                return

        if not np.isfinite(width_um) or width_um < 0:
            return

        try:
            self._syncing_outer_geometry = True
            self.update_outer_geometry_from_current_roi(width_um, update_message=False)
        finally:
            self._syncing_outer_geometry = False

    def get_current_roi_corners_xy(self):
        if self._roi_layer is None or len(self._roi_layer.data) == 0:
            return None
        corners_xy = np.asarray(self._roi_layer.data[0])[:, ::-1]
        return self.order_corners_clockwise(corners_xy)

    def expand_rectangle_corners(self, corners_xy: np.ndarray, expand_x_px: float, expand_y_px: float):
        c = self.order_corners_clockwise(corners_xy)
        center = c.mean(axis=0)

        e0 = c[1] - c[0]
        e1 = c[3] - c[0]
        l0 = float(np.linalg.norm(e0))
        l1 = float(np.linalg.norm(e1))
        if l0 <= 0 or l1 <= 0:
            return None

        u0 = e0 / l0
        u1 = e1 / l1
        h0 = l0 * 0.5
        h1 = l1 * 0.5

        expanded = np.array(
            [
                center - (h0 + expand_x_px) * u0 - (h1 + expand_y_px) * u1,
                center + (h0 + expand_x_px) * u0 - (h1 + expand_y_px) * u1,
                center + (h0 + expand_x_px) * u0 + (h1 + expand_y_px) * u1,
                center - (h0 + expand_x_px) * u0 + (h1 + expand_y_px) * u1,
            ],
            dtype=float,
        )
        return expanded

    def update_outer_geometry_from_current_roi(self, device_width_um: float, update_message: bool = True):
        corners_xy = self.get_current_roi_corners_xy()
        if corners_xy is None:
            if update_message:
                self.images_output.value = "[WARN] Geometry layer is missing. Run 'Segment + View' again."
            return False

        px_xy = um_to_xy_pixels(device_width_um, self.last_x_step_um, self.last_y_step_um)
        if px_xy is None:
            if update_message:
                self.images_output.value = (
                    "[WARN] Could not convert Device width (um) to pixels. Check image voxel metadata (x/y step)."
                )
            return False

        expand_x_px, expand_y_px = px_xy
        expanded_xy = self.expand_rectangle_corners(corners_xy, expand_x_px, expand_y_px)
        if expanded_xy is None:
            if update_message:
                self.images_output.value = "[WARN] Could not compute expanded geometry."
            return False

        expanded_yx = expanded_xy[:, ::-1]
        if self._roi_outer_layer is None or self._roi_outer_layer not in self.viewer.layers:
            self._roi_outer_layer = self.viewer.add_shapes(name="geometry_device_width")

        self._roi_outer_layer.data = [expanded_yx]
        self._roi_outer_layer.shape_type = ["rectangle"]
        self._roi_outer_layer.edge_color = "yellow"
        self._roi_outer_layer.face_color = "transparent"
        self._roi_outer_layer.editable = False

        if update_message:
            self.images_output.value = (
                f"[OK] Added outer geometry layer using Device width={float(device_width_um):.4g} um "
                f"(~dx={expand_x_px:.2f}px, dy={expand_y_px:.2f}px on each side)."
            )
        return True

    def apply_device_width_layer(self, device_width_um: float):
        corners_xy = self.get_current_roi_corners_xy()
        if corners_xy is None:
            self.images_output.value = "[WARN] Draw or adjust rectangle first."
            return

        try:
            self._active_device_width_um = float(device_width_um)
        except Exception:
            self.images_output.value = "[WARN] Device width must be numeric."
            return

        self.update_outer_geometry_from_current_roi(self._active_device_width_um, update_message=True)

    def compute_focus_patch_votes_for_stack(self, stack: np.ndarray, n_sampling: int, patch: int):
        if stack is None:
            return None

        stack_arr = np.asarray(stack)
        if stack_arr.ndim == 3:
            stack_gray = stack_arr
        else:
            return None

        Z, H, W = stack_gray.shape
        if Z <= 0 or H <= 0 or W <= 0:
            return None

        patch = max(5, int(patch))
        grid = max(4, int(n_sampling))
        half = patch // 2

        if H <= patch or W <= patch:
            scores = [np.std(sobel(self.to_gray(stack_gray[z]))) for z in range(Z)]
            z_best = int(np.argmax(scores))
            counts = np.zeros(Z, dtype=np.int32)
            counts[z_best] = 1
            return counts

        ys = np.linspace(half, H - half - 1, grid).astype(int)
        xs = np.linspace(half, W - half - 1, grid).astype(int)

        counts = np.zeros(Z, dtype=np.int32)
        for y in ys:
            for x in xs:
                y0 = max(0, y - half)
                y1 = min(H, y + half)
                x0 = max(0, x - half)
                x1 = min(W, x + half)
                f = np.array([self.focus_score(stack_gray[z, y0:y1, x0:x1]) for z in range(Z)], dtype=np.float32)
                best_z = int(np.argmax(f))
                counts[best_z] += 1
        return counts

    def apply_crop_from_roi(self):
        corners_xy = None
        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers:
            if len(getattr(self._roi_outer_layer, "data", [])) > 0:
                outer_xy = np.asarray(self._roi_outer_layer.data[0])[:, ::-1]
                corners_xy = self.order_corners_clockwise(outer_xy)
        if corners_xy is None:
            corners_xy = self.get_current_roi_corners_xy()
        if corners_xy is None:
            self.images_output.value = "[WARN] Draw or adjust geometry first."
            return
        if self.last_image is None:
            self.images_output.value = "[WARN] Run 'Segment + View' first."
            return

        if self.last_stack is not None:
            scale = max(1, int(self.last_focus_downsample))
            stack_for_crop = self.last_stack
            if self.last_focus_zmap_full is not None:
                refocused = self.refocus_stack_around_plane(self.last_stack, self.last_focus_zmap_full)
                if refocused is not None:
                    stack_for_crop = refocused

            cropped_stack = self.crop_rectified_stack_from_corners(stack_for_crop, corners_xy * float(scale))
            if cropped_stack is None:
                self.images_output.value = "[WARN] Crop failed for selected geometry (stack)."
                return

            roi_inner_xy = self.get_current_roi_corners_xy()
            roi_stack_only = None
            if roi_inner_xy is not None:
                roi_stack_only = self.crop_rectified_stack_from_corners(stack_for_crop, roi_inner_xy * float(scale))
            self._last_geometry_vote_counts = None

            self._cropped_stack_xy_raw = cropped_stack
            self._cropped_stack_z_raw = None
            self.cropped_xyz = None
            self._cropped_organoid_mask_xy_raw = None

            cropped_view = self.scale_to_uint8_view(cropped_stack)
            if self._cropped_layer is not None and self._cropped_layer in self.viewer.layers:
                self._cropped_layer.data = cropped_view
            else:
                self._cropped_layer = self.add_layer_if_nonzero(
                    cropped_view,
                    name="cropped_rotated",
                    layer_type="image",
                )
            counts = None
            if roi_stack_only is not None:
                counts = self.compute_focus_patch_votes_for_stack(
                    roi_stack_only.astype(np.float32),
                    n_sampling=int(self.last_focus_n_sampling or 10),
                    patch=int(self.last_focus_patch or 50),
                )

            self._cropped_stack_z_raw = self._cropped_stack_xy_raw
            self.cropped_xyz = self._cropped_stack_z_raw

            if self._mask_central_region_enabled and self.last_organoid_region is not None:
                organoid_mask = np.asarray(self.last_organoid_region, dtype=bool)
                target_h, target_w = int(stack_for_crop.shape[1]), int(stack_for_crop.shape[2])
                if scale > 1:
                    organoid_mask = np.repeat(np.repeat(organoid_mask.astype(np.uint8), scale, axis=0), scale, axis=1).astype(bool)

                if organoid_mask.shape[0] < target_h:
                    pad_h = target_h - organoid_mask.shape[0]
                    organoid_mask = np.pad(organoid_mask, ((0, pad_h), (0, 0)), mode="constant", constant_values=False)
                if organoid_mask.shape[1] < target_w:
                    pad_w = target_w - organoid_mask.shape[1]
                    organoid_mask = np.pad(organoid_mask, ((0, 0), (0, pad_w)), mode="constant", constant_values=False)
                organoid_mask = organoid_mask[:target_h, :target_w]

                cropped_organoid = self.crop_rectified_from_corners(
                    organoid_mask.astype(np.float32),
                    corners_xy * float(scale),
                )
                if cropped_organoid is not None:
                    self._cropped_organoid_mask_xy_raw = np.asarray(cropped_organoid > 0.5, dtype=bool)

            if counts is not None and len(counts) > 0:
                self._last_geometry_vote_counts = np.asarray(counts, dtype=int)
            self.images_output.value = "[OK] Cropped aligned stack created from current geometry."
            return

        cropped_img = self.crop_rectified_from_corners(self.last_image, corners_xy)
        if cropped_img is None:
            self.images_output.value = "[WARN] Crop failed for selected geometry."
            return
        self.images_output.value = "[OK] Cropped aligned image created from current geometry."

    def get_cropped_outputs(self):
        """Return cropped data and geometry-vote metadata for downstream use."""
        z_um, y_um, x_um = self._loaded_voxel_um
        if self.last_z_step_um is not None:
            z_um = self.last_z_step_um
        if self.last_y_step_um is not None:
            y_um = self.last_y_step_um
        if self.last_x_step_um is not None:
            x_um = self.last_x_step_um

        xy_um = self.last_xy_step_um
        if xy_um is None and (x_um is not None or y_um is not None):
            xy_um = float(np.nanmean([v for v in [x_um, y_um] if v is not None]))

        z_votes = None
        if self._last_geometry_vote_counts is not None:
            counts = np.asarray(self._last_geometry_vote_counts, dtype=int)
            z_votes = {int(i): int(c) for i, c in enumerate(counts)}

        pixel_size_um = {
            "x_um": x_um,
            "y_um": y_um,
            "z_um": z_um,
            "xy_um": xy_um,
        }

        return (
            self.cropped_xyz,
            self._active_device_width_um,
            pixel_size_um,
            z_votes,
            self.image_name,
            bool(self._mask_central_region_enabled),
            self._cropped_organoid_mask_xy_raw,
        )

    # -------- viewer helpers --------
    def add_layer_if_nonzero(self, data, name, layer_type="image", **kwargs):
        if data is None or not np.any(data):
            return None
        if layer_type == "labels":
            return self.viewer.add_labels(data, name=name, **kwargs)
        return self.viewer.add_image(data, name=name, **kwargs)

    def update_segment_button(self, *_):
        self.segment_and_view_widget.call_button.enabled = True
