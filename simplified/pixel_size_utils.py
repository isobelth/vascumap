from pathlib import Path
from typing import Optional, Tuple

import tifffile
from liffile import LifFile


def read_voxel_size_um(
    source_path: Optional[Path],
    source_is_lif: bool,
    selected_lif: Optional[Path] = None,
    image_index: Optional[int] = None,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return voxel size tuple (z_um, y_um, x_um) from LIF or TIFF metadata."""

    def step(axis, xa):
        if axis not in xa.coords or xa.coords[axis].size < 2:
            return None
        return float(xa.coords[axis][1] - xa.coords[axis][0])

    z_um = y_um = x_um = None

    if source_is_lif:
        if selected_lif is None:
            return (None, None, None)
        try:
            with LifFile(selected_lif) as lif:
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
