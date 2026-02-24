from pathlib import Path
from typing import Optional, Tuple
import tifffile
from liffile import LifFile
import numpy as np
import napari
from magicgui import magicgui
from magicgui.widgets import Container, TextEdit
from skimage import util
from skimage.filters import threshold_triangle, median, sobel, gaussian, threshold_yen
from skimage.measure import label, regionprops_table, regionprops, moments_central
from skimage.morphology import disk, remove_small_objects, remove_small_holes, closing
from skimage.transform import ProjectiveTransform, warp, probabilistic_hough_line
from scipy.ndimage import rotate, binary_dilation
from skimage.draw import line

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
        high_frac: float = 1.10,
        smooth_window: int = 5,
        bin_size: float = 2.0,
        min_run_frac: float = 0.25,
        typical_pct: float = 50.0,
        line_length: int = 400,
        line_gap: int = 900,
        hough_threshold: int = 70,
        mask_sigma: float = 2.0,
        mask_frac_thresh: float = 0.70,
    ):
        self.enable_gui = bool(enable_gui)
        self.low_frac = low_frac
        self.high_frac = high_frac
        self.smooth_window = smooth_window
        self.bin_size = bin_size
        self.min_run_frac = min_run_frac
        self.typical_pct = typical_pct
        self.line_length = line_length
        self.line_gap = line_gap
        self.hough_threshold = hough_threshold
        self.mask_sigma = mask_sigma
        self.mask_frac_thresh = mask_frac_thresh

        self.viewer = napari.Viewer(show=self.enable_gui)

        self._selected_image_folder: Optional[Path] = None
        self._selected_lif: Optional[Path] = None
        self._image_paths = []
        self._image_choice_map = {}

        self._roi_layer = None
        self._roi_outer_layer = None
        self._active_device_width_um = None
        self._syncing_outer_geometry = False
        self._cropped_layer = None
        self._last_image = None

        self._last_stack = None
        self._last_focus_downsample = 1
        self._last_focus_n_sampling = 10
        self._last_focus_patch = 50
        self._last_focus_zmap_full = None

        self._last_z_step_um = None
        self._last_y_step_um = None
        self._last_x_step_um = None
        self._last_xy_step_um = None
        self._last_geometry_vote_counts = None
        self._loaded_voxel_um = (None, None, None)

        self._cropped_stack_xy_raw = None
        self._cropped_stack_z_raw = None
        self.cropped_xyz = None
        self.image_name = None


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
            self._list_images(image_source)

        @magicgui(
            image_choice={"label": "Image", "choices": ["(load images)"], "widget_type": "ComboBox"},
            mask_central_region={"label": "Mask central region"},
            see_interim_layers={"label": "See interim layers (debug)"},
            clear_layers={"label": "Clear viewer first"},
            call_button="Segment + View",
        )
        def segment_and_view(image_choice: str = "(load images)",
            mask_central_region: bool = False,
            see_interim_layers: bool = False,
            clear_layers: bool = True,
        ):
            self._segment_and_view(
                image_choice=image_choice,
                focus_downsample=4,
                focus_n_sampling=10,
                focus_patch=50,
                mask_central_region=mask_central_region,
                see_interim_layers=see_interim_layers,
                clear_layers=clear_layers,
            )

        @magicgui(call_button="Create cropped aligned")
        def apply_crop():
            self._apply_crop_from_roi()

        @magicgui(
            auto_call=True,
            call_button=False,
            device_width_um={"label": "Device width (um)", "min": 0.0, "max": 1000.0, "step": 1.0},
        )
        def device_width_ok(device_width_um: float = 0.0):
            self._apply_device_width_layer(device_width_um)

        self.list_images = list_images
        self.segment_and_view = segment_and_view
        self.apply_crop = apply_crop
        self.device_width_ok = device_width_ok


        self.main_panel = Container(
            widgets=[
                self.list_images,
                self.segment_and_view,
                self.device_width_ok,
                self.apply_crop,
                self.images_output,
            ]
        )
        self.viewer.window.add_dock_widget(self.main_panel, area="right")

        self._update_segment_button()

    def _fmt_um(self, value):
        if value is None:
            return "NA"
        try:
            vf = float(value)
        except Exception:
            return "NA"
        if not np.isfinite(vf):
            return "NA"
        return f"{vf:.4g}"

    def _voxel_log_text(self, z_um, y_um, x_um):
        return f"Voxel size (um): x={self._fmt_um(x_um)}, y={self._fmt_um(y_um)}, z={self._fmt_um(z_um)}"

    def _reset_image_choices(self):
        self._image_choice_map = {}
        self.segment_and_view.image_choice.choices = ["(load images)"]
        self.segment_and_view.image_choice.value = "(load images)"

    def _set_last_voxel_steps(self, z_um, y_um, x_um):
        self._last_z_step_um = z_um
        self._last_y_step_um = y_um
        self._last_x_step_um = x_um
        self._last_xy_step_um = np.nanmean([v for v in [x_um, y_um] if v is not None]) if (x_um is not None or y_um is not None) else None

    def _image_name_from_choice(self, image_choice: str) -> str:
        label = str(image_choice)
        if ":" in label:
            label = label.split(":", 1)[1].strip()
        return Path(label).stem

    def run_automatic(
        self,
        image_source: Path,
        image_index: int = 0,
        device_width_um: float = 35.0,
        mask_central_region: bool = False,
    ):
        source = Path(image_source)
        self._list_images(source)

        if not self._image_choice_map:
            raise ValueError("Could not load images for automatic segmentation.")

        selected_label = None
        for label, idx in self._image_choice_map.items():
            if int(idx) == int(image_index):
                selected_label = label
                break

        if selected_label is None:
            raise ValueError(f"image_index {image_index} is out of range for source: {source}")

        self._segment_and_view(
            image_choice=selected_label,
            focus_downsample=4,
            focus_n_sampling=10,
            focus_patch=50,
            mask_central_region=mask_central_region,
            see_interim_layers=False,
            clear_layers=True,
        )

        self._apply_device_width_layer(float(device_width_um))
        self._apply_crop_from_roi()

        if self.cropped_xyz is None:
            msg = str(getattr(self.images_output, "value", "Automatic segmentation failed."))
            raise RuntimeError(msg)

        return self.get_cropped_outputs()

    def save_overlay_and_slice_tifs(self, name_prefix: str, run_suffix: int, output_dir: Optional[Path] = None):
        if self._last_image is None:
            return None, None

        out_dir = Path(output_dir) if output_dir is not None else Path.cwd()
        out_dir.mkdir(parents=True, exist_ok=True)

        base = np.asarray(self._last_image)
        if base.ndim == 3:
            base = np.mean(base, axis=-1)
        base_u8 = self._scale_to_uint8_view(base)

        overlay = np.stack([base_u8, base_u8, base_u8], axis=-1)

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

        overlay_path = out_dir / f"{name_prefix}_overlay_geometry_{int(run_suffix)}.tif"
        tifffile.imwrite(str(overlay_path), overlay)
        return overlay_path

    # -------- Focus helpers (integrated; no separate class) --------
    def _to_gray(self, im):
        if im.ndim == 2:
            return im
        return np.mean(im, axis=-1)

    def _focus_score(self, patch):
        p = np.asarray(self._to_gray(patch), dtype=float)
        return float(np.std(sobel(p)))

    def _curved_plane_refocus(self, stack_zyx, grid=20, patch=50, mask=None):
        Z, H, W = stack_zyx.shape
        ys = np.linspace(patch // 2, H - patch // 2 - 1, grid).astype(int)
        xs = np.linspace(patch // 2, W - patch // 2 - 1, grid).astype(int)

        pts, zs = [], []
        for y in ys:
            for x in xs:
                if mask is not None and not mask[y, x]:
                    continue
                sl = (slice(y - patch // 2, y + patch // 2), slice(x - patch // 2, x + patch // 2))
                f = np.array([self._focus_score(stack_zyx[z][sl]) for z in range(Z)], dtype=np.float32)
                pts.append((x, y))
                zs.append(int(np.argmax(f)))

        if len(zs) < 6:
            scores = []
            for z in range(Z):
                sl = self._to_gray(stack_zyx[z])
                if mask is not None:
                    sl = sl * mask
                scores.append(np.std(sobel(sl)))
            z0 = int(np.argmax(scores))
            img = self._to_gray(stack_zyx[z0])
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
    def _compute_focus_plane_from_stack(self, stack: Optional[np.ndarray], downsample: int, n_sampling: int, patch: int, source_is_lif: bool = False, image_index: Optional[int] = None):
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
            elif arr.ndim != 4:
                raise ValueError(f"Unsupported LIF array shape: {arr.shape}")

            self._last_stack = arr.astype(np.float32)
            z_um, y_um, x_um = read_voxel_size_um(
                self._selected_lif,
                source_is_lif=True,
                selected_lif=self._selected_lif,
                image_index=idx,
            )
            self._last_z_step_um = z_um
            self._last_y_step_um = y_um
            self._last_x_step_um = x_um
            self._last_xy_step_um = np.nanmean([v for v in [x_um, y_um] if v is not None]) if (x_um is not None or y_um is not None) else None
        else:
            arr = np.asarray(stack)

        ds = max(1, int(downsample))
        self._last_focus_downsample = ds

        stack_score_full = np.mean(arr, axis=-1) if arr.ndim == 4 else arr
        focus_full, zmap_full, _, _ = self._curved_plane_refocus(
            stack_score_full,
            grid=int(n_sampling),
            patch=int(patch),
            mask=None,
        )
        self._last_focus_zmap_full = np.asarray(zmap_full, dtype=np.int16) if zmap_full is not None else None

        if ds > 1:
            focus_out = focus_full[::ds, ::ds]
        else:
            focus_out = focus_full

        return focus_out

    def _refocus_stack_around_plane(self, stack: np.ndarray, zmap: np.ndarray):
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
    def _is_color_image_2d(self, arr: np.ndarray) -> bool:
        return arr.ndim == 3 and arr.shape[-1] in (3, 4) and arr.shape[0] > 32 and arr.shape[1] > 32

    def _scale_to_uint8_view(self, arr):
        if arr is None:
            return None
        data = np.asarray(arr)
        dmin = float(np.nanmin(data))
        dmax = float(np.nanmax(data))
        if not np.isfinite(dmin) or not np.isfinite(dmax) or dmax <= dmin:
            return np.zeros_like(data, dtype=np.uint8)
        scaled = np.clip((data - dmin) / (dmax - dmin), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    def _order_corners_clockwise(self, corners_xy: np.ndarray):
        c = np.asarray(corners_xy, dtype=float)
        centroid = c.mean(axis=0)
        angles = np.arctan2(c[:, 1] - centroid[1], c[:, 0] - centroid[0])
        c = c[np.argsort(angles)]
        start = np.argmin(c[:, 0] + c[:, 1])
        return np.roll(c, -start, axis=0)

    def _crop_rectified_from_corners(self, in_focus_plane: np.ndarray, corners_xy: np.ndarray):
        if corners_xy is None:
            return None
        c = self._order_corners_clockwise(corners_xy)
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

    def _crop_rectified_stack_from_corners(self, stack: np.ndarray, corners_xy: np.ndarray):
        if corners_xy is None or stack is None:
            return None
        c = self._order_corners_clockwise(corners_xy)
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

        if stack.ndim == 3:
            warped = np.stack(
                [warp(stack[z], tform, output_shape=(height, width), order=1, preserve_range=True) for z in range(stack.shape[0])],
                axis=0,
            )
        elif stack.ndim == 4:
            warped = np.stack(
                [
                    warp(
                        stack[z],
                        tform,
                        output_shape=(height, width),
                        order=1,
                        preserve_range=True,
                        channel_axis=-1,
                    )
                    for z in range(stack.shape[0])
                ],
                axis=0,
            )
        else:
            return None

        return warped.astype(stack.dtype)

    def _signed_orientation(self, region):
        img = region.image.astype(float)
        mu = moments_central(img)
        angle_rad = 0.5 * np.arctan2(2 * mu[1, 1], mu[2, 0] - mu[0, 2])
        return np.rad2deg(angle_rad)

    def _corners_touch_border(self, corners_xy: np.ndarray, shape, margin=0):
        H, W = shape
        x = corners_xy[:, 0]
        y = corners_xy[:, 1]
        return (x <= margin).any() or (x >= (W - 1 - margin)).any() or (y <= margin).any() or (y >= (H - 1 - margin)).any()

    def _mask_out_organoid(self, in_focus_plane):
        inverted = gaussian(
            util.invert(np.asarray(in_focus_plane, dtype=np.float32)),
            sigma=float(self.mask_sigma),
            mode="nearest",
            preserve_range=True,
        ).astype(np.float32)
        H, W = in_focus_plane.shape[:2]
        xy_area = H * W
        cyi, cxi = H // 2, W // 2
        r = int(min(H, W) * 0.1)
        yy, xx = np.ogrid[:H, :W]
        central_roi = (yy - cyi) ** 2 + (xx - cxi) ** 2 <= r**2

        thresh = threshold_yen(inverted)
        labelled = label(inverted > thresh)
        props = regionprops(labelled)
        if len(props) == 0:
            return np.zeros((H, W), dtype=bool)

        def score(p):
            overlap = np.sum(labelled[central_roi] == p.label)
            py, px = p.centroid
            dist2 = (py - cyi) ** 2 + (px - cxi) ** 2
            return (-overlap, dist2)

        best_prop = min(props, key=score)
        if (best_prop.area > xy_area * self.mask_frac_thresh) or (best_prop.solidity < 0.5):
            thresh = threshold_triangle(inverted)
            labelled = label(inverted > thresh)
            props = regionprops(labelled)
            if len(props) == 0:
                return np.zeros((H, W), dtype=bool)
            best_prop = min(props, key=score)

        organoid_region = labelled == best_prop.label
        organoid_region = remove_small_holes(organoid_region, area_threshold=50000)
        organoid_region = closing(organoid_region, disk(5))
        return organoid_region

    def _oriented_rect_corners_crop_necks_and_flares(self, mask: np.ndarray):
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            return None, None, None

        cx, cy = xs.mean(), ys.mean()
        centroid_xy = np.array([cx, cy], float)

        mu = moments_central(mask.astype(np.uint8))
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
        high_thr = self.high_frac * typical_width
        keep = (~(widths_s < low_thr)) & (~(widths_s > high_thr))
        too_wide = widths_s > high_thr

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
            crop_width = False
        else:
            long_min = float(best_start * self.bin_size)
            long_max = float((best_end + 1) * self.bin_size)
            band_mask_bins = (bins >= best_start) & (bins <= best_end)
            crop_width = bool(np.any(too_wide & (~band_mask_bins)))

        in_long_band = (long >= long_min) & (long <= long_max)
        if crop_width:
            short_min_use = float(short[in_long_band].min())
            short_max_use = float(short[in_long_band].max())
        else:
            short_min_use = short_min_full
            short_max_use = short_max_full

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

    def _segment_from_plane(self, in_focus_plane: np.ndarray, mask_central_region: bool, return_debug: bool):
        flag = False
        in_focus_plane = self._to_gray(in_focus_plane)

        median_thresholded = median(np.asarray(in_focus_plane, dtype=np.float32), footprint=disk(7)).astype(np.float32)
        sobel_operated = sobel(median_thresholded).astype(np.float32)
        thresh = threshold_triangle(sobel_operated)
        binary = sobel_operated > thresh

        h, w = in_focus_plane.shape[:2]
        binary[h // 3 : 2 * (h // 3), w // 3 : 2 * (w // 3)] = 0

        organoid_region = None
        if mask_central_region:
            organoid_region = self._mask_out_organoid(in_focus_plane)
            binary[organoid_region] = 0

        labels = label(binary)
        data = regionprops_table(labels, binary, properties=("label", "area", "eccentricity"))
        condition = (data["area"] > 100) & (data["eccentricity"] > 0.5)
        labels_to_dilate = util.map_array(labels, data["label"], data["label"] * condition)

        dilated_output = np.zeros_like(labels, dtype=np.uint8)
        base_selem = np.zeros((31, 31), dtype=bool)
        base_selem[15, :] = 1
        pad = base_selem.shape[0] // 2

        for region in regionprops(labels_to_dilate):
            angle_to_rotate = self._signed_orientation(region)
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

        corners, angle_rad, centroid_xy = self._oriented_rect_corners_crop_necks_and_flares(device_mask)
        if corners is None or self._corners_touch_border(corners, device_mask.shape, margin=5):
            flag = True
            edges = remove_small_objects(labels_to_dilate > 0)
            segs = probabilistic_hough_line(
                edges,
                line_length=self.line_length,
                line_gap=self.line_gap,
                threshold=self.hough_threshold,
            )
            reconstructed = np.zeros_like(edges, dtype=bool)
            for (x0, y0), (x1, y1) in segs:
                rr, cc = line(y0, x0, y1, x1)
                reconstructed[rr, cc] = True

            reconstructed_mask = np.logical_or(reconstructed, post_dilation_mask)
            updated_clean_labels = label(~reconstructed_mask)
            props = regionprops(updated_clean_labels)
            largest_prop = max(props, key=lambda p: p.area)
            new_device_mask = updated_clean_labels == largest_prop.label
            new_corners, new_angle_rad, new_centroid_xy = self._oriented_rect_corners_crop_necks_and_flares(new_device_mask)

        if flag:
            final_corners = new_corners
            final_angle_rad = new_angle_rad
            final_centroid_xy = new_centroid_xy
        else:
            final_corners = corners
            final_angle_rad = angle_rad
            final_centroid_xy = centroid_xy

        cropped_rotated = self._crop_rectified_from_corners(in_focus_plane, final_corners)

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
    def _list_images(self, image_source: Path):
        p = Path(image_source)
        if not p.exists():
            self.images_output.value = "[WARN] Please select a folder, a .tif/.tiff file, or a .lif file."
            return

        self._selected_image_folder = None
        self._selected_lif = None
        self._image_paths = []
        self._last_stack = None
        self._last_focus_downsample = 1
        choices = []
        choice_map = {}

        if p.is_dir():
            self._selected_image_folder = p
            image_files = sorted([q for q in p.iterdir() if q.is_file() and q.suffix.lower() in (".tif", ".tiff")])
            if not image_files:
                self.images_output.value = "[WARN] No .tif/.tiff images found in the selected folder."
                self._reset_image_choices()
                self._update_segment_button()
                return
            self._image_paths = image_files
            for i, pth in enumerate(image_files):
                label_txt = f"{i}: {pth.name}"
                choices.append(label_txt)
                choice_map[label_txt] = i
            self._image_choice_map = choice_map
            self.segment_and_view.image_choice.choices = choices
            self.segment_and_view.image_choice.value = choices[0]
            z_um, y_um, x_um = read_voxel_size_um(image_files[0], source_is_lif=False)
            self._loaded_voxel_um = (z_um, y_um, x_um)
            self.images_output.value = f"[OK] Found {len(choices)} images in folder. Select one and click Segment + View. {self._voxel_log_text(z_um, y_um, x_um)}"
            self._update_segment_button()
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
                self._reset_image_choices()
                self._update_segment_button()
                return

            if not choices:
                self.images_output.value = "[WARN] No readable images found inside the selected .lif file."
                self._reset_image_choices()
                self._update_segment_button()
                return

            self._image_choice_map = choice_map
            self.segment_and_view.image_choice.choices = choices
            self.segment_and_view.image_choice.value = choices[0]
            first_idx = choice_map[choices[0]]
            z_um, y_um, x_um = read_voxel_size_um(
                self._selected_lif,
                source_is_lif=True,
                selected_lif=self._selected_lif,
                image_index=first_idx,
            )
            self._loaded_voxel_um = (z_um, y_um, x_um)
            self.images_output.value = f"[OK] Found {len(choices)} images in .lif. Select one and click Segment + View. {self._voxel_log_text(z_um, y_um, x_um)}"
            self._update_segment_button()
            return

        if p.suffix.lower() in (".tif", ".tiff"):
            self._image_paths = [p]
            label_txt = f"0: {p.name}"
            self._image_choice_map = {label_txt: 0}
            self.segment_and_view.image_choice.choices = [label_txt]
            self.segment_and_view.image_choice.value = label_txt
            z_um, y_um, x_um = read_voxel_size_um(p, source_is_lif=False)
            self._loaded_voxel_um = (z_um, y_um, x_um)
            self.images_output.value = f"[OK] Loaded single .tif file. Select it and click Segment + View. {self._voxel_log_text(z_um, y_um, x_um)}"
            self._update_segment_button()
            return

        self.images_output.value = "[WARN] Unsupported selection. Choose a folder, a .tif/.tiff file, or a .lif file."
        self._reset_image_choices()
        self._update_segment_button()

    def _segment_and_view(
        self,
        image_choice: str,
        focus_downsample: int,
        focus_n_sampling: int,
        focus_patch: int,
        mask_central_region: bool,
        see_interim_layers: bool,
        clear_layers: bool,
    ):
        source_is_lif = bool(getattr(self, "_selected_lif", None) and self._selected_lif.exists())

        if not source_is_lif and not self._image_paths:
            self.images_output.value = "[WARN] Select a folder/.tif/.lif and click 'Load images' first."
            return
        if not self._image_choice_map:
            self.images_output.value = "[WARN] Click 'Load images' to populate the dropdown."
            return

        image_index = self._image_choice_map.get(image_choice)
        if image_index is None:
            self.images_output.value = "[WARN] Please select an image from the dropdown."
            return

        self.image_name = self._image_name_from_choice(image_choice)

        self._last_focus_n_sampling = int(focus_n_sampling)
        self._last_focus_patch = int(focus_patch)

        try:
            if source_is_lif:
                in_focus_plane = self._compute_focus_plane_from_stack(
                    stack=None,
                    downsample=focus_downsample,
                    n_sampling=focus_n_sampling,
                    patch=focus_patch,
                    source_is_lif=True,
                    image_index=image_index,
                )
                in_focus_plane, organoid_region, final_corners, _, debug = self._segment_from_plane(
                    in_focus_plane,
                    mask_central_region,
                    return_debug=True,
                )
            else:
                source_path = self._image_paths[image_index]
                arr = np.asarray(tifffile.imread(str(source_path)))

                z_um, y_um, x_um = read_voxel_size_um(Path(source_path), source_is_lif=False)
                self._set_last_voxel_steps(z_um, y_um, x_um)

                if self._is_color_image_2d(arr) or arr.ndim < 3:
                    in_focus_plane = arr.astype(np.float32)
                    self._last_stack = None
                    self._last_focus_downsample = 1
                else:
                    stack = arr.astype(np.float32)
                    in_focus_plane = self._compute_focus_plane_from_stack(stack, focus_downsample, focus_n_sampling, focus_patch)
                    self._last_stack = stack
                    self._last_focus_downsample = max(1, int(focus_downsample))

                in_focus_plane, organoid_region, final_corners, _, debug = self._segment_from_plane(
                    in_focus_plane,
                    mask_central_region,
                    return_debug=True,
                )
        except Exception as e:
            self.images_output.value = f"[ERROR] Segmentation failed: {type(e).__name__}: {e}"
            return

        if clear_layers:
            self.viewer.layers.clear()
            self._roi_layer = None
            self._roi_outer_layer = None
            self._cropped_layer = None

        self._last_image = in_focus_plane
        self._active_device_width_um = 30.0
        self._last_geometry_vote_counts = None
        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers:
            self.viewer.layers.remove(self._roi_outer_layer)
        self._roi_outer_layer = None
        self._cropped_stack_xy_raw = None
        self._cropped_stack_z_raw = None
        self.cropped_xyz = None
        self.device_width_ok.device_width_um.enabled = True
        self.device_width_ok.device_width_um.value = 30.0


        self._add_layer_if_nonzero(in_focus_plane, name="original", layer_type="image")

        if see_interim_layers:
            self._add_layer_if_nonzero(debug.get("sobel_operated"), name="sobel", layer_type="image")
            self._add_layer_if_nonzero(debug.get("binary").astype(np.uint8), name="binary", layer_type="image")
            self._add_layer_if_nonzero(debug.get("labels_to_dilate").astype(np.int32), name="labels_to_dilate", layer_type="labels")
            final_layer = self._add_layer_if_nonzero(debug.get("post_dilation_mask").astype(np.uint8), name="post_dilation_mask", layer_type="labels")
            if final_layer is not None:
                final_layer.opacity = 0.4
            device_layer = self._add_layer_if_nonzero(debug.get("device_mask").astype(np.uint8), name="device_mask", layer_type="labels")
            if device_layer is not None:
                device_layer.opacity = 0.4
            if organoid_region is not None:
                organoid_layer = self._add_layer_if_nonzero(organoid_region.astype(np.uint8), name="organoid_region", layer_type="labels")
                if organoid_layer is not None:
                    organoid_layer.opacity = 0.4

        force_roi = clear_layers or self._roi_layer is None or len(getattr(self._roi_layer, "data", [])) == 0
        self._set_roi_layer(final_corners, force=force_roi)
        self._update_outer_geometry_from_current_roi(30.0, update_message=False)
        self.images_output.value = (
            "[OK] Segmentation complete. Outer geometry is shown at default Device width=30 um; adjust width as needed."
        )


    def _set_roi_layer(self, corners_xy, force=False):
        if corners_xy is None:
            self.images_output.value = "[WARN] No corners found for ROI."
            return
        corners_yx = np.asarray(corners_xy)[:, ::-1]
        if self._roi_layer is None or self._roi_layer not in self.viewer.layers:
            self._roi_layer = self.viewer.add_shapes(name="geometry")
            try:
                self._roi_layer.events.data.connect(self._on_roi_layer_data_changed)
            except Exception:
                pass
        if force or len(self._roi_layer.data) == 0:
            self._roi_layer.data = [corners_yx]
        self._roi_layer.shape_type = ["rectangle"]
        self._roi_layer.edge_color = "red"
        self._roi_layer.face_color = "transparent"
        self._roi_layer.editable = True
        self._roi_layer.mode = "select"

    def _on_roi_layer_data_changed(self, event=None):
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
            self._update_outer_geometry_from_current_roi(width_um, update_message=False)
        finally:
            self._syncing_outer_geometry = False

    def _get_current_roi_corners_xy(self):
        if self._roi_layer is None or len(self._roi_layer.data) == 0:
            return None
        corners_xy = np.asarray(self._roi_layer.data[0])[:, ::-1]
        return self._order_corners_clockwise(corners_xy)

    def _expand_rectangle_corners(self, corners_xy: np.ndarray, expand_x_px: float, expand_y_px: float):
        c = self._order_corners_clockwise(corners_xy)
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

    def _update_outer_geometry_from_current_roi(self, device_width_um: float, update_message: bool = True):
        corners_xy = self._get_current_roi_corners_xy()
        if corners_xy is None:
            if update_message:
                self.images_output.value = "[WARN] Geometry layer is missing. Run 'Segment + View' again."
            return False

        px_xy = um_to_xy_pixels(device_width_um, self._last_x_step_um, self._last_y_step_um)
        if px_xy is None:
            if update_message:
                self.images_output.value = (
                    "[WARN] Could not convert Device width (um) to pixels. Check image voxel metadata (x/y step)."
                )
            return False

        expand_x_px, expand_y_px = px_xy
        expanded_xy = self._expand_rectangle_corners(corners_xy, expand_x_px, expand_y_px)
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

    def _apply_device_width_layer(self, device_width_um: float):
        corners_xy = self._get_current_roi_corners_xy()
        if corners_xy is None:
            self.images_output.value = "[WARN] Draw or adjust rectangle first."
            return

        try:
            self._active_device_width_um = float(device_width_um)
        except Exception:
            self.images_output.value = "[WARN] Device width must be numeric."
            return

        self._update_outer_geometry_from_current_roi(self._active_device_width_um, update_message=True)

    def _compute_focus_patch_votes_for_stack(self, stack: np.ndarray, n_sampling: int, patch: int):
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
            scores = [np.std(sobel(self._to_gray(stack_gray[z]))) for z in range(Z)]
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
                f = np.array([self._focus_score(stack_gray[z, y0:y1, x0:x1]) for z in range(Z)], dtype=np.float32)
                best_z = int(np.argmax(f))
                counts[best_z] += 1
        return counts

    def _apply_crop_from_roi(self):
        corners_xy = None
        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers:
            if len(getattr(self._roi_outer_layer, "data", [])) > 0:
                outer_xy = np.asarray(self._roi_outer_layer.data[0])[:, ::-1]
                corners_xy = self._order_corners_clockwise(outer_xy)
        if corners_xy is None:
            corners_xy = self._get_current_roi_corners_xy()
        if corners_xy is None:
            self.images_output.value = "[WARN] Draw or adjust geometry first."
            return
        if self._last_image is None:
            self.images_output.value = "[WARN] Run 'Segment + View' first."
            return

        if self._last_stack is not None:
            scale = max(1, int(self._last_focus_downsample))
            stack_for_crop = self._last_stack
            if self._last_focus_zmap_full is not None:
                refocused = self._refocus_stack_around_plane(self._last_stack, self._last_focus_zmap_full)
                if refocused is not None:
                    stack_for_crop = refocused

            cropped_stack = self._crop_rectified_stack_from_corners(stack_for_crop, corners_xy * float(scale))
            if cropped_stack is None:
                self.images_output.value = "[WARN] Crop failed for selected geometry (stack)."
                return

            roi_inner_xy = self._get_current_roi_corners_xy()
            roi_stack_only = None
            if roi_inner_xy is not None:
                roi_stack_only = self._crop_rectified_stack_from_corners(stack_for_crop, roi_inner_xy * float(scale))
            self._last_geometry_vote_counts = None

            self._cropped_stack_xy_raw = cropped_stack
            self._cropped_stack_z_raw = None
            self.cropped_xyz = None


            cropped_view = self._scale_to_uint8_view(cropped_stack)
            if self._cropped_layer is not None and self._cropped_layer in self.viewer.layers:
                self._cropped_layer.data = cropped_view
            else:
                self._cropped_layer = self._add_layer_if_nonzero(
                    cropped_view,
                    name="cropped_rotated",
                    layer_type="image",
                )
            counts = None
            if roi_stack_only is not None:
                counts = self._compute_focus_patch_votes_for_stack(
                    roi_stack_only.astype(np.float32),
                    n_sampling=int(self._last_focus_n_sampling or 10),
                    patch=int(self._last_focus_patch or 50),
                )

            self._cropped_stack_z_raw = self._cropped_stack_xy_raw
            self.cropped_xyz = self._cropped_stack_z_raw

            if counts is not None and len(counts) > 0:
                self._last_geometry_vote_counts = np.asarray(counts, dtype=int)
            self.images_output.value = "[OK] Cropped aligned stack created from current geometry."
            return

        cropped_img = self._crop_rectified_from_corners(self._last_image, corners_xy)
        if cropped_img is None:
            self.images_output.value = "[WARN] Crop failed for selected geometry."
            return
        self.images_output.value = "[OK] Cropped aligned image created from current geometry."

    def get_cropped_outputs(self):
        """Return cropped data and geometry-vote metadata for downstream use."""
        z_um, y_um, x_um = self._loaded_voxel_um
        if self._last_z_step_um is not None:
            z_um = self._last_z_step_um
        if self._last_y_step_um is not None:
            y_um = self._last_y_step_um
        if self._last_x_step_um is not None:
            x_um = self._last_x_step_um

        xy_um = self._last_xy_step_um
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

        return self.cropped_xyz, self._active_device_width_um, pixel_size_um, z_votes, self.image_name

    # -------- viewer helpers --------
    def _add_layer_if_nonzero(self, data, name, layer_type="image", **kwargs):
        if data is None or not np.any(data):
            return None
        if layer_type == "labels":
            return self.viewer.add_labels(data, name=name, **kwargs)
        return self.viewer.add_image(data, name=name, **kwargs)

    def _update_segment_button(self, *_):
        self.segment_and_view.call_button.enabled = True
