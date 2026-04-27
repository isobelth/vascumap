"""Entry point – run ``python -m vascumap`` from the repository root.

Opens the launcher GUI to collect settings, then dispatches to either
napari-based curation (CurationApp) or fully automatic batch processing.
"""

import sys
import time
from pathlib import Path

from qtpy.QtWidgets import QApplication

# Prepend the package directory so sibling modules can use flat imports.
_PKG_DIR = Path(__file__).resolve().parent
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from launcher_gui import VascuMapLauncherGUI
from models import Pix2Pix, load_segmentation_model
from pipeline import discover_jobs, filter_jobs, run_batch, run_batch_from_curation


def main() -> int:
    """Run the VascuMap launcher GUI and execute the chosen pipeline mode."""
    qapp = QApplication.instance() or QApplication(sys.argv)

    print("VascuMap launcher starting...")
    gui = VascuMapLauncherGUI()
    while not gui.closed:
        qapp.processEvents()
        time.sleep(0.05)

    cfg = gui.config
    if cfg is None:
        print("[INFO] Launcher closed without selecting a run mode.")
        return 0

    print("=" * 70)
    print(f"Source : {cfg['source_dir']}")
    print(f"Output : {cfg['output_dir']}")
    if cfg["skip_dir"]:
        print(f"Skip   : {cfg['skip_dir']}")
    print(f"Mode   : {'GUI curation' if cfg['mode'] == 'gui' else 'Automatic'}")
    print("=" * 70)

    print("[INFO] Loading models...")
    model_p2p = Pix2Pix(model_path=cfg["pix2pix_ckpt"])
    model_unet = load_segmentation_model(cfg["unet_ckpt"])

    source, image_files, all_jobs = discover_jobs(cfg["source_dir"], force_mask=cfg["force_mask"], require_merged=cfg["require_merged"])
    print(f"Source : {source}")
    print(f"Files  : {len(image_files)}  |  Total jobs: {len(all_jobs)}")

    skip_names: set[str] = set()
    if cfg["skip_dir"]:
        skip_root = Path(cfg["skip_dir"])
        if skip_root.is_dir():
            skip_names = {folder.name for folder in skip_root.iterdir() if folder.is_dir()}
            print(f"Skip   : {len(skip_names)} folders found in {skip_root}")

    jobs = filter_jobs(all_jobs, skip_names)
    if not jobs:
        print("[INFO] No jobs to process after filtering. Exiting.")
        return 0

    if cfg["mode"] == "gui":
        from gui_region_detection import CurationApp

        def on_curation_done(curated_jobs):
            """Callback invoked by CurationApp after the user clicks Done."""
            run_batch_from_curation(curated_jobs, cfg["output_dir"], save_all_interim=cfg["save_all_interim"], model_p2p=model_p2p, model_unet=model_unet)

        app = CurationApp(jobs, device_width_um=cfg["device_width_um"], brightfield_channel=cfg["brightfield_channel"], on_done=on_curation_done)
        app.open()
        qapp.exec_()
    else:
        run_batch(jobs, cfg["output_dir"], device_width_um=cfg["device_width_um"], brightfield_channel=cfg["brightfield_channel"], save_all_interim=cfg["save_all_interim"], model_p2p=model_p2p, model_unet=model_unet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
