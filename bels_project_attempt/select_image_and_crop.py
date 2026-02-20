from __future__ import annotations

from typing import Tuple
import time

import numpy as np
import napari

try:
    from qtpy.QtWidgets import QApplication
except Exception:
    QApplication = None

try:
    from bels_project_attempt.device_segmentation_gui_app import create_device_segmentation_app
except ModuleNotFoundError:
    from device_segmentation_gui_app import create_device_segmentation_app


def _get_src_voxel_um(app) -> Tuple[float, float, float]:
    lz, ly, lx = getattr(app, "_loaded_voxel_um", (None, None, None))
    z_um = lz if lz is not None else getattr(app, "_last_z_step_um", None)
    y_um = ly if ly is not None else (getattr(app, "_last_y_step_um", None) or getattr(app, "_last_xy_step_um", None))
    x_um = lx if lx is not None else (getattr(app, "_last_x_step_um", None) or getattr(app, "_last_xy_step_um", None))

    if z_um is None or y_um is None or x_um is None:
        raise RuntimeError("Missing voxel metadata on GUI app.")
    return float(z_um), float(y_um), float(x_um)


def _window_is_visible(app) -> bool:
    try:
        qt_window = app.viewer.window._qt_window
        return bool(qt_window is not None and qt_window.isVisible())
    except Exception:
        return False


def _wait_for_crop_or_close(app, poll_s: float = 0.05):
    while True:
        if getattr(app, "cropped_xyz", None) is not None:
            return
        if not _window_is_visible(app):
            return
        if QApplication is not None:
            try:
                QApplication.processEvents()
            except Exception:
                pass
        time.sleep(max(0.01, float(poll_s)))


def select_image_and_return_cropped_stack(return_app: bool = False):
    app = create_device_segmentation_app()

    print("Napari launched. Use: Load images -> Segment + View -> Create cropped aligned.")
    print("Z is auto-selected and locked (manual override disabled).")
    print("Close the Napari window when done to return the cropped stack.")

    already_running = False
    try:
        napari.run()
    except RuntimeError:
        already_running = True

    if already_running:
        print("Napari event loop already running; waiting for crop or window close...")
        _wait_for_crop_or_close(app)

    cropped = getattr(app, "cropped_xyz", None)
    if cropped is None:
        print("No cropped stack found. Run 'Create cropped aligned' before closing Napari, then rerun Cell 4.")
        if return_app:
            return None, None, app
        return None

    cropped = np.asarray(cropped, dtype=np.float32).copy()
    src_voxel_um = _get_src_voxel_um(app)

    if return_app:
        return cropped, src_voxel_um, app
    return cropped
