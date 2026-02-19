import numpy as np
import napari
from pathlib import Path
from typing import Optional

from magicgui import magicgui
from magicgui.widgets import Container, TextEdit
from liffile import LifFile
from skimage import util
from skimage.filters import threshold_triangle, sobel, gaussian, threshold_yen
from skimage.measure import label, regionprops, moments_central
from skimage.morphology import disk, remove_small_holes, closing
from skimage.transform import ProjectiveTransform, warp
import tifffile

try:
    from device_segmentation_initial import segment_from_plane_initial
except ModuleNotFoundError:
    from bels_project_attempt.device_segmentation_initial import segment_from_plane_initial


class DeviceSegmentationApp:

    def __init__(

        self,

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



        self.viewer = napari.Viewer()



        self._selected_image_folder: Optional[Path] = None

        self._selected_lif: Optional[Path] = None

        self._image_paths = []

        self._image_choice_map = {}



        self._roi_layer = None

        self._roi_outer_layer = None

        self._active_device_width_um = None

        self._syncing_outer_geometry = False

        self._focus_votes_layer = None

        self._cropped_layer = None

        self._refocus_zmap_layer = None

        self._refocus_before_layer = None

        self._refocus_after_layer = None

        self._refocus_delta_layer = None

        self._last_prerefocus_zmap = None

        self._last_image = None

        self._last_image_path = None

        self._last_see_interim_layers = True



        self._last_stack = None

        self._last_focus_downsample = 1

        self._last_focus_n_sampling = 10

        self._last_focus_patch = 50



        self._last_z_step_um = None

        self._last_y_step_um = None

        self._last_x_step_um = None

        self._last_xy_step_um = None

        self._last_center_z0 = None

        self._last_sampled_best_z_min = None

        self._last_sampled_best_z_max = None

        self._last_geometry_best_z_min = None

        self._last_geometry_best_z_max = None

        self._last_geometry_vote_counts = None

        self._last_nz = None

        self._loaded_voxel_um = (None, None, None)



        self._cropped_stack_xy_raw = None

        self._cropped_stack_xy_refocused = None

        self._cropped_stack_z_raw = None

        self.cropped_xyz = None

        self._updating_z_widgets = False



        self._z_tracker = {}

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

        def list_images(image_source: Path = Path()):

            self._list_images(image_source)



        @magicgui(

            image_choice={"label": "Image", "choices": ["(load images)"], "widget_type": "ComboBox"},

            focus_downsample={"label": "LIF downsample", "min": 1, "max": 16, "step": 1},

            focus_n_sampling={"label": "LIF n_sampling", "min": 4, "max": 100, "step": 1},

            focus_patch={"label": "LIF patch", "min": 5, "max": 512, "step": 1},

            mask_central_region={"label": "Mask central region"},

            see_interim_layers={"label": "See interim layers (debug)"},

            clear_layers={"label": "Clear viewer first"},

            call_button="Segment + View",

        )

        def segment_and_view(

            image_choice: str = "(load images)",

            focus_downsample: int = 4,

            focus_n_sampling: int = 10,

            focus_patch: int = 50,

            mask_central_region: bool = False,

            see_interim_layers: bool = False,

            clear_layers: bool = True,

        ):

            self._segment_and_view(

                image_choice=image_choice,

                focus_downsample=focus_downsample,

                focus_n_sampling=focus_n_sampling,

                focus_patch=focus_patch,

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

            device_width_um={"label": "Device width (um)", "min": 0.0, "max": 1000000.0, "step": 1.0},

        )

        def device_width_ok(device_width_um: float = 0.0):

            self._apply_device_width_layer(device_width_um)



        @magicgui(

            z_range_um={"label": "Z range (um)", "min": 1.0, "max": 100000.0, "step": 1.0},

            z_min={"label": "z_min (slice)", "widget_type": "SpinBox", "min": 0, "step": 1},

            z_max={"label": "z_max (slice)", "widget_type": "SpinBox", "min": 0, "step": 1},

            call_button="Crop Z range",

        )

        def crop_z(z_range_um: float = 200.0, z_min: int = 0, z_max: int = 0):

            self._apply_z_crop(z_range_um=z_range_um)



        self.list_images = list_images

        self.segment_and_view = segment_and_view

        self.apply_crop = apply_crop

        self.device_width_ok = device_width_ok

        self.crop_z = crop_z



        self.main_panel = Container(

            widgets=[

                self.list_images,

                self.segment_and_view,

                self.device_width_ok,

                self.apply_crop,

                self.crop_z,

                self.images_output,

            ]

        )

        self.viewer.window.add_dock_widget(self.main_panel, area="right")



        self.segment_and_view.image_choice.changed.connect(self._update_segment_button)

        self.segment_and_view.call_button.enabled = False

        self._update_segment_button()

        self.device_width_ok.device_width_um.enabled = False

        self.crop_z.call_button.enabled = False

        self._set_z_controls_enabled(False, reset_values=True)



        try:

            self.crop_z.z_range_um.changed.connect(lambda *_: self._update_z_crop_bounds(show_warning=True))

            self.crop_z.z_min.changed.connect(lambda *_: self._update_z_range_from_slices(show_warning=True))

            self.crop_z.z_max.changed.connect(lambda *_: self._update_z_range_from_slices(show_warning=True))

        except Exception:

            pass



    # -------- Unified metadata reader (LIF/TIFF) --------

    def _read_voxel_size_um(self, source_path: Optional[Path], source_is_lif: bool, image_index: Optional[int] = None):

        """Return (z_um, y_um, x_um) using one function:

        if lif -> xarray coord steps, elif tif -> TIFF tags.

        """



        def step(axis, xa):

            if axis not in xa.coords or xa.coords[axis].size < 2:

                return None

            return float(xa.coords[axis][1] - xa.coords[axis][0])



        z_um = y_um = x_um = None



        if source_is_lif:

            if self._selected_lif is None:

                return (None, None, None)

            try:

                with LifFile(self._selected_lif) as lif:

                    idx = int(image_index if image_index is not None else 0)

                    if idx < 0 or idx >= len(lif.images):

                        return (None, None, None)

                    img = lif.images[idx]

                    xa = img.asxarray()



                    x_step = step("X", xa)

                    y_step = step("Y", xa)

                    z_step = step("Z", xa)



                    x_um = x_step * 1e6 if x_step is not None else None

                    y_um = y_step * 1e6 if y_step is not None else None

                    z_um = z_step * 1e6 if z_step is not None else None

            except Exception:

                return (None, None, None)

            return (z_um, y_um, x_um)



        elif source_path is not None and str(source_path).lower().endswith((".tif", ".tiff")):

            try:

                with tifffile.TiffFile(str(source_path)) as tif:

                    tif_tags = {}

                    for tag in tif.pages[0].tags.values():

                        tif_tags[tag.name] = tag.value



                    # XY from resolution tags (as requested)

                    if "XResolution" in tif_tags:

                        xres = tif_tags["XResolution"]

                        x_pixel_size_um = 1.0 / (float(xres[0]) / float(xres[1]))

                        x_um = x_pixel_size_um

                    if "YResolution" in tif_tags:

                        yres = tif_tags["YResolution"]

                        y_pixel_size_um = 1.0 / (float(yres[0]) / float(yres[1]))

                        y_um = y_pixel_size_um



                    # Z from IJMetadata first, fallback to ImageDescription spacing

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



    def _fmt_um(self, v):

        if v is None:

            return "NA"

        try:

            vf = float(v)

        except Exception:

            return "NA"

        if not np.isfinite(vf):

            return "NA"

        return f"{vf:.4g}"



    def _voxel_log_text(self, z_um, y_um, x_um):

        return f"Voxel size (um): x={self._fmt_um(x_um)}, y={self._fmt_um(y_um)}, z={self._fmt_um(z_um)}"



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



    def _legacy_style_prerefocus_stack_zyx(self, stack_zyx, n_sampling: int, patch: int):

        stack = np.asarray(stack_zyx)

        if stack.ndim != 3:

            raise ValueError(f"Expected ZYX stack for pre-refocus, got shape={stack.shape}")



        _, zmap, _, _ = self._curved_plane_refocus(stack, grid=int(n_sampling), patch=int(patch), mask=None)

        zmap = np.asarray(zmap, dtype=np.int32)



        z_max = int(stack.shape[0])

        half_win = int(z_max // 2)

        deltas = np.arange(z_max, dtype=np.int32) - half_win



        z_idx = zmap[None, :, :] + deltas[:, None, None]

        z_idx = np.clip(z_idx, 0, z_max - 1)



        y_idx, x_idx = np.indices(zmap.shape, dtype=np.int32)

        y_idx = np.broadcast_to(y_idx, z_idx.shape)

        x_idx = np.broadcast_to(x_idx, z_idx.shape)



        z_refocused = stack[z_idx, y_idx, x_idx]

        return np.asarray(z_refocused, dtype=stack.dtype), zmap.astype(np.int16)

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

            z_um, y_um, x_um = self._read_voxel_size_um(self._selected_lif, source_is_lif=True, image_index=idx)

            self._last_z_step_um = z_um

            self._last_y_step_um = y_um

            self._last_x_step_um = x_um

            self._last_xy_step_um = np.nanmean([v for v in [x_um, y_um] if v is not None]) if (x_um is not None or y_um is not None) else None

        else:

            arr = np.asarray(stack)



        nz = int(arr.shape[0])

        ds = max(1, int(downsample))

        self._last_focus_downsample = ds



        stack_score_full = np.mean(arr, axis=-1) if arr.ndim == 4 else arr

        focus_full, _, _, zs = self._curved_plane_refocus(stack_score_full, grid=int(n_sampling), patch=int(patch), mask=None)



        if ds > 1:

            focus_out = focus_full[::ds, ::ds]

        else:

            focus_out = focus_full



        self._last_center_z0 = None

        self._last_nz = nz

        if zs is None or len(zs) == 0:

            self._last_sampled_best_z_min = None

            self._last_sampled_best_z_max = None

        else:

            zs_arr = np.asarray(zs, dtype=int)

            zs_arr = zs_arr[(zs_arr >= 0) & (zs_arr < nz)]

            if zs_arr.size > 0:

                counts = np.bincount(zs_arr, minlength=nz)

                self._last_center_z0 = int(np.argmax(counts))

                self._last_sampled_best_z_min = int(zs_arr.min())

                self._last_sampled_best_z_max = int(zs_arr.max())

            else:

                self._last_sampled_best_z_min = None

                self._last_sampled_best_z_max = None



        return focus_out



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

        return segment_from_plane_initial(

            in_focus_plane,

            mask_central_region,

            return_debug,

            to_gray=self._to_gray,

            mask_out_organoid=self._mask_out_organoid,

            signed_orientation=self._signed_orientation,

            oriented_rect_corners_crop_necks_and_flares=self._oriented_rect_corners_crop_necks_and_flares,

            corners_touch_border=self._corners_touch_border,

            crop_rectified_from_corners=self._crop_rectified_from_corners,

            line_length=self.line_length,

            line_gap=self.line_gap,

            hough_threshold=self.hough_threshold,

        )



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

                self._image_choice_map = {}

                self.segment_and_view.image_choice.choices = ["(load images)"]

                self.segment_and_view.image_choice.value = "(load images)"

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

            z_um, y_um, x_um = self._read_voxel_size_um(image_files[0], source_is_lif=False)

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

                self._image_choice_map = {}

                self.segment_and_view.image_choice.choices = ["(load images)"]

                self.segment_and_view.image_choice.value = "(load images)"

                self._update_segment_button()

                return



            if not choices:

                self.images_output.value = "[WARN] No readable images found inside the selected .lif file."

                self._image_choice_map = {}

                self.segment_and_view.image_choice.choices = ["(load images)"]

                self.segment_and_view.image_choice.value = "(load images)"

                self._update_segment_button()

                return



            self._image_choice_map = choice_map

            self.segment_and_view.image_choice.choices = choices

            self.segment_and_view.image_choice.value = choices[0]

            first_idx = choice_map[choices[0]]

            z_um, y_um, x_um = self._read_voxel_size_um(self._selected_lif, source_is_lif=True, image_index=first_idx)

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

            z_um, y_um, x_um = self._read_voxel_size_um(p, source_is_lif=False)

            self._loaded_voxel_um = (z_um, y_um, x_um)

            self.images_output.value = f"[OK] Loaded single .tif file. Select it and click Segment + View. {self._voxel_log_text(z_um, y_um, x_um)}"

            self._update_segment_button()

            return



        self.images_output.value = "[WARN] Unsupported selection. Choose a folder, a .tif/.tiff file, or a .lif file."

        self._image_choice_map = {}

        self.segment_and_view.image_choice.choices = ["(load images)"]

        self.segment_and_view.image_choice.value = "(load images)"

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

                in_focus_plane, organoid_region, final_corners, cropped_rotated, debug = self._segment_from_plane(

                    in_focus_plane,

                    mask_central_region,

                    return_debug=True,

                )

                self._last_image_path = self._selected_lif

            else:

                source_path = self._image_paths[image_index]

                arr = np.asarray(tifffile.imread(str(source_path)))



                z_um, y_um, x_um = self._read_voxel_size_um(Path(source_path), source_is_lif=False)

                self._last_z_step_um = z_um

                self._last_y_step_um = y_um

                self._last_x_step_um = x_um

                self._last_xy_step_um = np.nanmean([v for v in [x_um, y_um] if v is not None]) if (x_um is not None or y_um is not None) else None



                if self._is_color_image_2d(arr) or arr.ndim < 3:

                    in_focus_plane = arr.astype(np.float32)

                    self._last_stack = None

                    self._last_focus_downsample = 1

                    self._last_center_z0 = None

                    self._last_sampled_best_z_min = None

                    self._last_sampled_best_z_max = None

                    self._last_nz = None

                else:

                    stack = arr.astype(np.float32)

                    in_focus_plane = self._compute_focus_plane_from_stack(stack, focus_downsample, focus_n_sampling, focus_patch)

                    self._last_stack = stack

                    self._last_focus_downsample = max(1, int(focus_downsample))



                in_focus_plane, organoid_region, final_corners, cropped_rotated, debug = self._segment_from_plane(

                    in_focus_plane,

                    mask_central_region,

                    return_debug=True,

                )

                self._last_image_path = source_path

        except Exception as e:

            self.images_output.value = f"[ERROR] Segmentation failed: {type(e).__name__}: {e}"

            return



        if clear_layers:

            self.viewer.layers.clear()

            self._roi_layer = None

            self._roi_outer_layer = None

            self._cropped_layer = None



        self._last_image = in_focus_plane

        self._last_see_interim_layers = see_interim_layers

        self._active_device_width_um = 30.0

        self._last_geometry_best_z_min = None

        self._last_geometry_best_z_max = None

        self._last_geometry_vote_counts = None

        self._z_tracker = {}

        if self._last_stack is not None:

            try:

                src_nz = int(np.asarray(self._last_stack).shape[0])

                self._register_z_stage(

                    stage_name="original_stack",

                    nz=src_nz,

                    orig_min=0.0,

                    orig_max=float(max(0, src_nz - 1)),

                    note="Original loaded stack z indices.",

                )

            except Exception:

                self._z_tracker = {}

        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers:

            self.viewer.layers.remove(self._roi_outer_layer)

        self._roi_outer_layer = None

        if self._focus_votes_layer is not None and self._focus_votes_layer in self.viewer.layers:

            self.viewer.layers.remove(self._focus_votes_layer)

        self._focus_votes_layer = None

        if self._refocus_zmap_layer is not None and self._refocus_zmap_layer in self.viewer.layers:

            self.viewer.layers.remove(self._refocus_zmap_layer)

        self._refocus_zmap_layer = None

        if self._refocus_before_layer is not None and self._refocus_before_layer in self.viewer.layers:

            self.viewer.layers.remove(self._refocus_before_layer)

        self._refocus_before_layer = None

        if self._refocus_after_layer is not None and self._refocus_after_layer in self.viewer.layers:

            self.viewer.layers.remove(self._refocus_after_layer)

        self._refocus_after_layer = None

        if self._refocus_delta_layer is not None and self._refocus_delta_layer in self.viewer.layers:

            self.viewer.layers.remove(self._refocus_delta_layer)

        self._refocus_delta_layer = None

        self._last_prerefocus_zmap = None

        self._cropped_stack_xy_raw = None

        self._cropped_stack_xy_refocused = None

        self._cropped_stack_z_raw = None

        self.cropped_xyz = None

        self.device_width_ok.device_width_um.enabled = True

        self.device_width_ok.device_width_um.value = 30.0

        self.crop_z.call_button.enabled = False

        self._set_z_controls_enabled(False, reset_values=True)



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

        self._update_z_crop_bounds(show_warning=False)



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



    def _get_current_crop_corners_xy(self):

        if self._roi_outer_layer is not None and self._roi_outer_layer in self.viewer.layers:

            if len(getattr(self._roi_outer_layer, "data", [])) > 0:

                outer_xy = np.asarray(self._roi_outer_layer.data[0])[:, ::-1]

                return self._order_corners_clockwise(outer_xy)

        return self._get_current_roi_corners_xy()



    def _um_to_xy_pixels(self, width_um: float):

        try:

            width_um = float(width_um)

        except Exception:

            return None

        if not np.isfinite(width_um) or width_um < 0:

            return None



        x_um = self._last_x_step_um

        y_um = self._last_y_step_um

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



        px_xy = self._um_to_xy_pixels(device_width_um)

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

            return None, None



        stack_arr = np.asarray(stack)

        if stack_arr.ndim == 4:

            stack_gray = np.mean(stack_arr, axis=-1)

        elif stack_arr.ndim == 3:

            stack_gray = stack_arr

        else:

            return None, None



        Z, H, W = stack_gray.shape

        if Z <= 0 or H <= 0 or W <= 0:

            return None, None



        patch = max(5, int(patch))

        grid = max(4, int(n_sampling))

        half = patch // 2



        vote_volume = np.zeros((Z, H, W), dtype=np.float32)



        if H <= patch or W <= patch:

            scores = [np.std(sobel(self._to_gray(stack_gray[z]))) for z in range(Z)]

            z_best = int(np.argmax(scores))

            vote_volume[z_best, H // 2, W // 2] = 1.0

            counts = np.zeros(Z, dtype=np.int32)

            counts[z_best] = 1

            return counts, vote_volume



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

                yy0 = max(0, y - 1)

                yy1 = min(H, y + 2)

                xx0 = max(0, x - 1)

                xx1 = min(W, x + 2)

                vote_volume[best_z, yy0:yy1, xx0:xx1] += 1.0



        return counts, vote_volume



    def _format_geometry_vote_planes(self):

        counts = self._last_geometry_vote_counts

        if counts is None:

            return "none"

        counts = np.asarray(counts).astype(int)

        if counts.size == 0:

            return "none"

        hits = np.flatnonzero(counts > 0)

        if hits.size == 0:

            return "none"

        return ", ".join([f"z{int(i)}:{int(counts[i])}" for i in hits])



    def _geometry_vote_summary_suffix(self):

        txt = self._format_geometry_vote_planes()

        return f" Geometry vote planes (all nonzero): {txt}."



    def _z_linear_interp(self, z_idx: float, in_nz: int, out_min: float, out_max: float):

        if in_nz <= 1:

            return float(out_min)

        return float(out_min + (float(z_idx) / float(in_nz - 1)) * (float(out_max) - float(out_min)))



    def _register_z_stage(self, stage_name: str, nz: int, orig_min: float, orig_max: float, note: str = ""):

        self._z_tracker[str(stage_name)] = {

            "nz": int(max(0, int(nz))),

            "orig_min": float(orig_min),

            "orig_max": float(orig_max),

            "note": str(note),

        }



    def _register_z_stage_from_parent(

        self,

        parent_stage: str,

        stage_name: str,

        child_nz: int,

        parent_z_min: int = 0,

        parent_z_max: Optional[int] = None,

        note: str = "",

    ):

        p = self._z_tracker.get(str(parent_stage))

        if p is None:

            return False



        p_nz = int(p.get("nz", 0))

        if p_nz <= 0:

            return False



        pmin = int(np.clip(int(parent_z_min), 0, max(0, p_nz - 1)))

        pmax = int(p_nz - 1 if parent_z_max is None else np.clip(int(parent_z_max), 0, max(0, p_nz - 1)))

        if pmin > pmax:

            pmin, pmax = pmax, pmin



        o0 = self._z_linear_interp(pmin, p_nz, float(p["orig_min"]), float(p["orig_max"]))

        o1 = self._z_linear_interp(pmax, p_nz, float(p["orig_min"]), float(p["orig_max"]))

        self._register_z_stage(stage_name, int(child_nz), float(o0), float(o1), note=note)

        return True



    def z_to_original(self, stage_name: str, z_index: int):

        stage = self._z_tracker.get(str(stage_name))

        if stage is None:

            return None

        nz = int(stage.get("nz", 0))

        if nz <= 0:

            return None

        zi = int(np.clip(int(z_index), 0, max(0, nz - 1)))

        return self._z_linear_interp(zi, nz, float(stage["orig_min"]), float(stage["orig_max"]))



    def get_z_tracker(self):

        return dict(self._z_tracker)



    def z_tracker_summary(self):

        if not self._z_tracker:

            return "No z-tracker entries yet."

        lines = ["Z tracker (stage -> original z-space)"]

        for key, item in self._z_tracker.items():

            lines.append(

                f"  {key}: nz={int(item['nz'])}, original~[{float(item['orig_min']):.2f}, {float(item['orig_max']):.2f}]"

            )

        return "\n".join(lines)



    def _apply_crop_from_roi(self):

        corners_xy = self._get_current_crop_corners_xy()

        if corners_xy is None:

            self.images_output.value = "[WARN] Draw or adjust geometry first."

            return

        if self._last_image is None:

            self.images_output.value = "[WARN] Run 'Segment + View' first."

            return



        if self._last_stack is not None:

            scale = max(1, int(self._last_focus_downsample))

            cropped_stack = self._crop_rectified_stack_from_corners(self._last_stack, corners_xy * float(scale))

            if cropped_stack is None:

                self.images_output.value = "[WARN] Crop failed for selected geometry (stack)."

                return



            roi_inner_xy = self._get_current_roi_corners_xy()

            roi_stack_only_refocused = None

            roi_stack_only_raw = None

            if roi_inner_xy is not None:

                roi_stack_only_raw = self._crop_rectified_stack_from_corners(self._last_stack, roi_inner_xy * float(scale))

            self._last_geometry_best_z_min = None

            self._last_geometry_best_z_max = None

            self._last_geometry_vote_counts = None



            self._cropped_stack_xy_raw = cropped_stack

            self._cropped_stack_xy_refocused = None

            self._cropped_stack_z_raw = None

            self.cropped_xyz = None

            self._register_z_stage_from_parent(

                parent_stage="original_stack",

                stage_name="cropped_xy_raw",

                child_nz=int(self._cropped_stack_xy_raw.shape[0]),

                parent_z_min=0,

                parent_z_max=int(max(0, self._cropped_stack_xy_raw.shape[0] - 1)),

                note="XY-rectified stack from geometry; z preserved from original stack.",

            )



            try:

                refocused_stack, zmap_refocus = self._legacy_style_prerefocus_stack_zyx(

                    self._cropped_stack_xy_raw.astype(np.float32),

                    n_sampling=int(self._last_focus_n_sampling or 10),

                    patch=int(self._last_focus_patch or 50),

                )

                self._cropped_stack_xy_refocused = refocused_stack.astype(np.float32)

                self._last_prerefocus_zmap = np.asarray(zmap_refocus, dtype=np.float32)

                self._register_z_stage_from_parent(

                    parent_stage="cropped_xy_raw",

                    stage_name="cropped_xy_refocused",

                    child_nz=int(self._cropped_stack_xy_refocused.shape[0]),

                    parent_z_min=0,

                    parent_z_max=int(max(0, self._cropped_stack_xy_refocused.shape[0] - 1)),

                    note="Legacy-style prerefocus stack before z-cropping.",

                )

                self._last_center_z0 = int(max(0, self._cropped_stack_xy_refocused.shape[0] // 2))

            except Exception as e:

                self._cropped_stack_xy_refocused = None

                self._last_prerefocus_zmap = None

                self._last_center_z0 = None

                self.crop_z.call_button.enabled = False

                self._set_z_controls_enabled(False, reset_values=True)

                self.images_output.value = f"[ERROR] Pre-refocus failed, stopping because downstream steps require the refocused stack: {type(e).__name__}: {e}"

                return



            if roi_stack_only_raw is not None:

                try:

                    roi_stack_only_refocused, _ = self._legacy_style_prerefocus_stack_zyx(

                        roi_stack_only_raw.astype(np.float32),

                        n_sampling=int(self._last_focus_n_sampling or 10),

                        patch=int(self._last_focus_patch or 50),

                    )

                    roi_stack_only_refocused = roi_stack_only_refocused.astype(np.float32)

                except Exception:

                    roi_stack_only_refocused = None

            self.crop_z.call_button.enabled = True

            self._set_z_controls_enabled(True)

            self._sync_z_slice_widget_limits()



            self._cropped_layer = self._add_or_update_image_layer(

                self._cropped_layer,

                self._scale_to_uint8_view(self._get_active_stack_for_z()),

                "cropped_rotated",

            )



            if self._last_see_interim_layers and self._cropped_stack_xy_refocused is not None and self._last_prerefocus_zmap is not None:

                raw_mean = np.asarray(self._cropped_stack_xy_raw, dtype=np.float32).mean(axis=0)

                ref_mean = np.asarray(self._cropped_stack_xy_refocused, dtype=np.float32).mean(axis=0)

                delta_mean = ref_mean - raw_mean



                self._refocus_zmap_layer = self._add_or_update_image_layer(

                    self._refocus_zmap_layer,

                    np.asarray(self._last_prerefocus_zmap, dtype=np.float32),

                    "prerefocus_zmap",

                )

                self._refocus_before_layer = self._add_or_update_image_layer(

                    self._refocus_before_layer,

                    self._scale_to_uint8_view(raw_mean),

                    "prerefocus_before_mean",

                )

                self._refocus_after_layer = self._add_or_update_image_layer(

                    self._refocus_after_layer,

                    self._scale_to_uint8_view(ref_mean),

                    "prerefocus_after_mean",

                )

                self._refocus_delta_layer = self._add_or_update_image_layer(

                    self._refocus_delta_layer,

                    self._scale_to_uint8_view(delta_mean),

                    "prerefocus_delta_mean",

                )



            if self._last_center_z0 is None:

                self._last_center_z0 = int(max(0, self._get_active_stack_for_z().shape[0] // 2))

            counts, vote_volume = (None, None)

            if roi_stack_only_refocused is not None:

                counts, vote_volume = self._compute_focus_patch_votes_for_stack(

                    roi_stack_only_refocused.astype(np.float32),

                    n_sampling=int(self._last_focus_n_sampling or 10),

                    patch=int(self._last_focus_patch or 50),

                )



            if counts is not None and len(counts) > 0:

                self._last_center_z0 = int(np.argmax(counts))

                self._last_nz = int(len(counts))

                self._last_geometry_vote_counts = np.asarray(counts, dtype=int)

                nz_hits = np.flatnonzero(np.asarray(counts) > 0)

                if nz_hits.size > 0:

                    self._last_sampled_best_z_min = int(nz_hits.min())

                    self._last_sampled_best_z_max = int(nz_hits.max())

                    self._last_geometry_best_z_min = int(nz_hits.min())

                    self._last_geometry_best_z_max = int(nz_hits.max())

                else:

                    self._last_sampled_best_z_min = None

                    self._last_sampled_best_z_max = None

                    self._last_geometry_best_z_min = None

                    self._last_geometry_best_z_max = None



                if vote_volume is not None:

                    self._focus_votes_layer = self._add_or_update_image_layer(

                        self._focus_votes_layer,

                        self._scale_to_uint8_view(vote_volume),

                        "focus_votes_geometry_only",

                    )



                top_n = min(5, len(counts))

                top_idx = np.argsort(counts)[::-1][:top_n]

                top_txt = ", ".join([f"z{int(i)}:{int(counts[i])}" for i in top_idx])

                all_txt = self._format_geometry_vote_planes()

                self.images_output.value = (

                    f"[OK] Cropped aligned stack created from current geometry. center_z={self._last_center_z0}. "

                    f"Geometry-only focus patch votes on refocused stack (inner geometry only; top planes): {top_txt}. "

                    f"All nonzero voted planes: {all_txt}. Now choose Z range and click Crop Z range."

                )

            else:

                self.images_output.value = (

                    f"[OK] Cropped aligned stack created from current geometry. center_z={self._last_center_z0}. "

                    "Now choose Z range and click Crop Z range."

                )



            self._update_z_crop_bounds(show_warning=False)

            return



        cropped_img = self._crop_rectified_from_corners(self._last_image, corners_xy)

        if cropped_img is None:

            self.images_output.value = "[WARN] Crop failed for selected geometry."

            return

        self.images_output.value = "[OK] Cropped aligned image created from current geometry."



    # -------- z-crop --------

    def _get_active_stack_for_z(self):

        if self._cropped_stack_xy_refocused is not None:

            return self._cropped_stack_xy_refocused

        return None



    def _set_z_controls_enabled(self, enabled: bool, reset_values: bool = False):

        try:

            self.crop_z.z_range_um.enabled = bool(enabled)

            self.crop_z.z_min.enabled = bool(enabled)

            self.crop_z.z_max.enabled = bool(enabled)

        except Exception:

            pass



        if reset_values:

            try:

                self._updating_z_widgets = True

                self.crop_z.z_min.value = 0

                self.crop_z.z_max.value = 0

            except Exception:

                pass

            finally:

                self._updating_z_widgets = False



    def _sync_z_slice_widget_limits(self):

        try:

            active_stack = self._get_active_stack_for_z()

            nz = int(active_stack.shape[0]) if active_stack is not None else 0

            max_z = max(0, nz - 1)

            self.crop_z.z_min.min = 0

            self.crop_z.z_max.min = 0

            self.crop_z.z_min.max = max_z

            self.crop_z.z_max.max = max_z

        except Exception:

            pass



    def _largest_vote_run_above_threshold(self, counts: np.ndarray, min_votes_exclusive: float = 1.0):

        arr = np.asarray(counts).astype(float)

        nz_idx = np.flatnonzero(arr > float(min_votes_exclusive))

        if nz_idx.size == 0:

            return None, None



        best_start = best_end = None

        best_len = -1

        best_sum = -1.0



        cur_start = int(nz_idx[0])

        cur_end = int(nz_idx[0])

        for i in nz_idx[1:]:

            ii = int(i)

            if ii == (cur_end + 1):

                cur_end = ii

            else:

                cur_len = int(cur_end - cur_start + 1)

                cur_sum = float(np.sum(arr[cur_start : cur_end + 1]))

                if (cur_len > best_len) or (cur_len == best_len and cur_sum > best_sum):

                    best_start, best_end = cur_start, cur_end

                    best_len = cur_len

                    best_sum = cur_sum

                cur_start = ii

                cur_end = ii



        cur_len = int(cur_end - cur_start + 1)

        cur_sum = float(np.sum(arr[cur_start : cur_end + 1]))

        if (cur_len > best_len) or (cur_len == best_len and cur_sum > best_sum):

            best_start, best_end = cur_start, cur_end



        return int(best_start), int(best_end)



    def _compute_vote_driven_z_bounds(self, min_total_um: float = 160.0, min_down_um: float = 30.0):

        stack = self._get_active_stack_for_z()

        if stack is None or self._last_z_step_um is None or float(self._last_z_step_um) <= 0:

            return None, None, "[DEBUG] vote-run z-crop unavailable: missing active refocused stack or z-step metadata."



        nz = int(stack.shape[0])

        self._last_nz = nz

        z_step_um = float(self._last_z_step_um)



        counts = self._last_geometry_vote_counts

        if counts is None:

            return None, None, "[DEBUG] vote-run z-crop unavailable: no geometry vote counts found."



        counts = np.asarray(counts).astype(float)

        if counts.size != nz:

            aligned = np.zeros(nz, dtype=np.float32)

            ncopy = int(min(nz, counts.size))

            aligned[:ncopy] = counts[:ncopy]

            counts = aligned



        run_min, run_max = self._largest_vote_run_above_threshold(counts, min_votes_exclusive=1.0)

        if run_min is None or run_max is None:

            return None, None, "[DEBUG] vote-run z-crop unavailable: no contiguous run with vote count >1."



        z_min = int(run_min)

        z_max = int(run_max)

        run_len = int(z_max - z_min + 1)



        min_required_slices = int(np.floor(float(min_total_um) / z_step_um)) + 1

        min_down_slices = max(0, int(np.ceil(float(min_down_um) / z_step_um)))



        current_slices = int(z_max - z_min + 1)

        needed_for_total = max(0, int(min_required_slices - current_slices))



        desired_down = int(max(min_down_slices, needed_for_total))

        down_room = int(z_min)

        down_add = int(min(down_room, desired_down))

        z_min -= down_add



        current_slices = int(z_max - z_min + 1)

        needed_after_down = max(0, int(min_required_slices - current_slices))

        hit_z0 = bool(z_min == 0)



        up_add = 0

        if hit_z0 and needed_after_down > 0:

            up_room = int((nz - 1) - z_max)

            up_add = int(min(up_room, needed_after_down))

            z_max += up_add



        z_min = int(np.clip(z_min, 0, max(0, nz - 1)))

        z_max = int(np.clip(z_max, 0, max(0, nz - 1)))

        if z_min > z_max:

            z_min, z_max = z_max, z_min



        final_slices = int(z_max - z_min + 1)

        final_um = float(final_slices * z_step_um)

        target_met = bool(final_slices >= min_required_slices)

        dbg_txt = (

            f"[DEBUG] vote-run z-crop (vote>1 contiguous rule): run={int(run_min)}..{int(run_max)} ({run_len} slices). "

            f"target>={float(min_total_um):.1f}um (~{min_required_slices} slices at {z_step_um:.4g}um). "

            f"down_min={min_down_slices} slices (~{float(min_down_um):.1f}um); down_add={down_add}; hit_z0={hit_z0}. "

            f"up_add={up_add} (only after hit_z0). "

            f"final={z_min}..{z_max} ({final_slices} slices, {final_um:.2f}um)."

        )

        if not target_met:

            dbg_txt += " [WARN] Could not reach minimum target due to stack bounds."



        return z_min, z_max, dbg_txt



    def _compute_z_min_max_slices(self, z_range_um: float):

        z_min, z_max, _ = self._compute_vote_driven_z_bounds(min_total_um=160.0, min_down_um=30.0)

        return z_min, z_max



    def _update_z_crop_bounds(self, *_, show_warning: bool = False):

        active_stack = self._get_active_stack_for_z()

        if self._updating_z_widgets or active_stack is None:

            return

        z_min, z_max, dbg_txt = self._compute_vote_driven_z_bounds(min_total_um=160.0, min_down_um=30.0)

        if z_min is None or z_max is None:

            if show_warning:

                self.images_output.value = dbg_txt

            return



        try:

            self._updating_z_widgets = True

            nz = int(active_stack.shape[0])

            self.crop_z.z_min.max = max(0, nz - 1)

            self.crop_z.z_max.max = max(0, nz - 1)

            self.crop_z.z_min.value = int(z_min)

            self.crop_z.z_max.value = int(z_max)

            if self._last_z_step_um is not None and float(self._last_z_step_um) > 0:

                self.crop_z.z_range_um.value = float((int(z_max) - int(z_min) + 1) * float(self._last_z_step_um))

        finally:

            self._updating_z_widgets = False



        if show_warning and self._last_geometry_best_z_min is not None:

            if (self._last_geometry_best_z_min < z_min) or (self._last_geometry_best_z_max > z_max):

                self.images_output.value = (

                    f"[WARN] Auto z-bounds slices {z_min}..{z_max} "

                    f"does NOT cover geometry sampled best-zs ({self._last_geometry_best_z_min}..{self._last_geometry_best_z_max})."

                    f"{self._geometry_vote_summary_suffix()} {dbg_txt}"

                )

            else:

                self.images_output.value = dbg_txt



    def _update_z_range_from_slices(self, *_, show_warning: bool = False):

        active_stack = self._get_active_stack_for_z()

        if self._updating_z_widgets or active_stack is None:

            return

        nz = int(active_stack.shape[0])

        try:

            z_min = int(self.crop_z.z_min.value)

            z_max = int(self.crop_z.z_max.value)

        except Exception:

            return



        z_min = int(np.clip(z_min, 0, nz - 1))

        z_max = int(np.clip(z_max, 0, nz - 1))

        if z_min > z_max:

            z_min, z_max = z_max, z_min



        try:

            self._updating_z_widgets = True

            self.crop_z.z_min.value = z_min

            self.crop_z.z_max.value = z_max

        finally:

            self._updating_z_widgets = False



        if self._last_z_step_um is not None and float(self._last_z_step_um) > 0:

            try:

                self._updating_z_widgets = True

                self.crop_z.z_range_um.value = float((z_max - z_min + 1) * float(self._last_z_step_um))

            finally:

                self._updating_z_widgets = False



        if show_warning and self._last_geometry_best_z_min is not None:

            if (self._last_geometry_best_z_min < z_min) or (self._last_geometry_best_z_max > z_max):

                self.images_output.value = (

                    f"[WARN] slices {z_min}..{z_max} do NOT cover geometry sampled best-zs "

                    f"({self._last_geometry_best_z_min}..{self._last_geometry_best_z_max})."

                    f"{self._geometry_vote_summary_suffix()}"

                )



    def _apply_z_crop(self, z_range_um: float = 200.0):

        active_stack = self._get_active_stack_for_z()

        if active_stack is None:

            self.images_output.value = "[WARN] Create cropped aligned stack first."

            return



        nz = int(active_stack.shape[0])

        z_min, z_max, dbg_txt = self._compute_vote_driven_z_bounds(min_total_um=160.0, min_down_um=30.0)

        if z_min is None or z_max is None:

            self.images_output.value = f"[WARN] Cannot determine z_min/z_max. {dbg_txt}"

            return



        z_min = int(np.clip(z_min, 0, nz - 1))

        z_max = int(np.clip(z_max, 0, nz - 1))

        if z_min > z_max:

            z_min, z_max = z_max, z_min



        self._cropped_stack_z_raw = active_stack[z_min : z_max + 1]

        self.cropped_xyz = self._cropped_stack_z_raw

        parent_stage_name = "cropped_xy_refocused"

        self._register_z_stage_from_parent(

            parent_stage=parent_stage_name,

            stage_name="cropped_xyz",

            child_nz=int(self._cropped_stack_z_raw.shape[0]),

            parent_z_min=int(z_min),

            parent_z_max=int(z_max),

            note="User-selected z crop from geometry-rectified stack.",

        )

        self._cropped_layer = self._add_or_update_image_layer(

            self._cropped_layer,

            self._scale_to_uint8_view(self._cropped_stack_z_raw),

            "cropped_rotated",

        )

        try:

            self._updating_z_widgets = True

            self.crop_z.z_min.value = int(z_min)

            self.crop_z.z_max.value = int(z_max)

            if self._last_z_step_um is not None and float(self._last_z_step_um) > 0:

                self.crop_z.z_range_um.value = float((int(z_max) - int(z_min) + 1) * float(self._last_z_step_um))

        finally:

            self._updating_z_widgets = False



        mapped_min = self.z_to_original("cropped_xyz", 0)

        mapped_max = self.z_to_original("cropped_xyz", int(max(0, self._cropped_stack_z_raw.shape[0] - 1)))

        if mapped_min is None or mapped_max is None:

            self.images_output.value = f"[OK] Z-cropped stack created (slices {z_min}..{z_max}). {dbg_txt}"

        else:

            self.images_output.value = (

                f"[OK] Z-cropped stack created using vote-run rule (slices {z_min}..{z_max}). "

                f"Original-z approx range: [{mapped_min:.2f}, {mapped_max:.2f}] from source stack. {dbg_txt}"

            )



    def get_cropped_outputs(self):

        """Return (cropped_xyz, None) for backward compatibility."""

        return self.cropped_xyz, None



    # -------- viewer helpers --------

    def _add_or_update_image_layer(self, layer, data, name):

        if layer is not None and layer in self.viewer.layers:

            layer.data = data

            return layer

        return self._add_layer_if_nonzero(data, name=name, layer_type="image")



    def _has_signal(self, arr) -> bool:

        return arr is not None and np.any(arr)



    def _add_layer_if_nonzero(self, data, name, layer_type="image", **kwargs):

        if not self._has_signal(data):

            return None

        if layer_type == "labels":

            return self.viewer.add_labels(data, name=name, **kwargs)

        return self.viewer.add_image(data, name=name, **kwargs)



    def _update_segment_button(self, *_):

        image_value = self.segment_and_view.image_choice.value

        image_ok = bool(self._image_choice_map) and image_value in self._image_choice_map

        self.segment_and_view.call_button.enabled = image_ok




def create_device_segmentation_app():
    app = DeviceSegmentationApp()
    return app
