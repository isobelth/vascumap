"""Stage 1 — Discover jobs and write a manifest CSV.
Opens a basic GUI front end to allow easy inputs/choices.
Scans a source directory for .lif/.tif/.tiff files. Lifs are expanded 
so that one job = one image.
Each job corresponds to one row in the manifest csv.

Usage (GUI, end-user):
    python discover_jobs.py
    -> opens the launcher GUI to pick folders/options, then writes
       <output_dir>/manifest.csv

Usage (Python):
    from discover_jobs import build_manifest, write_manifest_csv
    df = build_manifest(...)
    write_manifest_csv(df, "manifest.csv")
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import List, Optional, Set

import pandas as pd
from liffile import LifFile


# ---------------------------------------------------------------------------
# Manifest schema - 1 row = 1 image = 1 job
# ---------------------------------------------------------------------------

MANIFEST_COLUMNS: List[str] = [
    "job_id",
    "source_file",
    "image_index",
    "image_name",
    "mask_flag",
    "expected_output_name",
    "output_dir",
    "device_width_um",
    "brightfield_channel",
    "pix2pix_ckpt",
    "unet_ckpt",
    "save_all_interim",
    "mode",
    "should_process",
    "skip_reason",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def determine_mask_mode(file_path: Path, image_name: str = "", force_mask: Optional[str] = None):
    """Infer organoid mask mode from filename keywords, or use a forced override.
    Returns 'dark', 'light', or False (no masking).
    """
    if force_mask is not None:
        return force_mask
    combined = (file_path.name + " " + image_name).lower()
    if "marina" in combined and "bead" in combined:
        return "light"
    return "dark" if "marina" in combined else False



# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------

def discover_jobs(source_dir: str | Path, force_mask: Optional[str] = None, require_merged: bool = True,
) -> List[dict]:
    """Scan the source directory for .lif/.tif/.tiff files
    Lif files are expanded so one job = one image (filter for "merged" keyword when "require_merge == True")
    Each job dict contains: source_file, image_index, image_name, mask_flag.
    Returns the job details sepcific to the image (source_file, image_index, image_name, mask_flag (can be inferred from image name))
    """
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source}")

    image_files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in (".lif", ".tif", ".tiff"))
    
    jobs: List[dict] = []
    for file_path in image_files:
        if file_path.suffix.lower() == ".lif":
            try:
                with LifFile(file_path) as lif:
                    for idx, image in enumerate(lif.images):
                        image_name = getattr(image, "name", "")
                        if require_merged and "merged" not in image_name.lower():
                            continue
                        jobs.append({
                            "source_file": file_path,
                            "image_index": idx,
                            "image_name": image_name,
                            "mask_flag": determine_mask_mode(file_path, image_name, force_mask),
                        })
            except Exception as exc:
                print(f"[SKIP] {file_path.name}: {exc}", file=sys.stderr)
        else:
            if require_merged and "merged" not in file_path.name.lower():
                continue
            jobs.append({
                "source_file": file_path,
                "image_index": 0,
                "image_name": file_path.stem,
                "mask_flag": determine_mask_mode(file_path, force_mask=force_mask),
            })
    return jobs

# ---------------------------------------------------------------------------
# Manifest building
# ---------------------------------------------------------------------------

def build_manifest(
    source_dir: str | Path,
    output_dir: str | Path,
    device_width_um: float,
    brightfield_channel: int = 0,
    pix2pix_ckpt: Optional[str | Path] = None,
    unet_ckpt: Optional[str | Path] = None,
    force_mask: Optional[str] = None,
    require_merged: bool = True,
    skip_dir: Optional[str | Path] = None,
    save_all_interim: bool = False,
    mode: str = "auto",
) -> pd.DataFrame:
    """Discover jobs and build a manifest DataFrame with one row per image.
    Skipped jobs are kept as rows with ``should_process=False`` and a ``skip_reason`` for clarity.
    """
    output_dir = Path(output_dir)
    ###### IDENTIFY WHICH FILES TO SKIP
    if (skip_dir is None) or not Path(skip_dir).is_dir():
        skip_names: Set[str] = set()
    else:
        skip_names = {p.name for p in Path(skip_dir).iterdir() if p.is_dir()}

    jobs = discover_jobs(source_dir, force_mask=force_mask, require_merged=require_merged)

    rows: List[dict] = []
    for job in jobs:
        src = Path(job["source_file"])
        if src.suffix.lower() == ".lif":
            safe_name = str(job["image_name"]).replace("/", "_").replace("\\", "_")
            out_name = f"{src.stem}_{safe_name}_img{job['image_index']}"
        else:
            out_name = src.stem
        skip = out_name in skip_names
        job_key = f"{src.resolve()}|{int(job['image_index'])}|{job['image_name']}"
        rows.append({
            "job_id": hashlib.sha1(job_key.encode("utf-8")).hexdigest()[:12],
            "source_file": str(src.resolve()),
            "image_index": int(job["image_index"]),
            "image_name": str(job["image_name"]),
            "mask_flag": job["mask_flag"],
            "expected_output_name": out_name,
            "output_dir": str((output_dir / out_name).resolve()),
            "device_width_um": float(device_width_um),
            "brightfield_channel": int(brightfield_channel),
            "pix2pix_ckpt": str(Path(pix2pix_ckpt).resolve()) if pix2pix_ckpt else "",
            "unet_ckpt": str(Path(unet_ckpt).resolve()) if unet_ckpt else "",
            "save_all_interim": bool(save_all_interim),
            "mode": str(mode),
            "should_process": not skip,
            "skip_reason": "output_already_exists" if skip else "",
        })

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def write_manifest_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write the manifest DataFrame to a CSV file (parents created)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


# ---------------------------------------------------------------------------
# Entry point — launches the GUI launcher to collect settings, then writes
# the manifest CSV. Designed for non-technical users running locally.
# ---------------------------------------------------------------------------

def run_from_config(cfg: dict, manifest_path: Optional[Path] = None) -> Path:
    """Build the manifest from a launcher config dict and write it to disk.

    The manifest is written to ``<output_dir>/manifest.csv`` unless
    ``manifest_path`` is given.
    """
    output_dir = Path(cfg["output_dir"])
    if manifest_path is None:
        manifest_path = output_dir / "manifest.csv"

    df = build_manifest(
        source_dir=cfg["source_dir"],
        output_dir=output_dir,
        device_width_um=cfg["device_width_um"],
        brightfield_channel=cfg.get("brightfield_channel", 0),
        pix2pix_ckpt=cfg.get("pix2pix_ckpt"),
        unet_ckpt=cfg.get("unet_ckpt"),
        force_mask=cfg.get("force_mask"),
        require_merged=cfg.get("require_merged", True),
        skip_dir=cfg.get("skip_dir"),
        save_all_interim=cfg.get("save_all_interim", False),
        mode=cfg.get("mode", "auto"),
    )

    write_manifest_csv(df, manifest_path)
    n_total = len(df)
    n_proc = int(df["should_process"].sum())
    print(f"Wrote manifest: {manifest_path}  ({n_proc}/{n_total} to process)")
    return manifest_path


def main() -> int:
    """Open the launcher GUI, collect settings, and write the manifest CSV."""
    import time

    repo_root = Path(__file__).resolve().parent.parent
    for candidate in (Path(__file__).resolve().parent, repo_root / "vascumap"):
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    from qtpy.QtWidgets import QApplication  # noqa: WPS433 (local import is intentional)
    from launcher_gui import VascuMapLauncherGUI  # noqa: WPS433

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
    if cfg.get("skip_dir"):
        print(f"Skip   : {cfg['skip_dir']}")
    print("=" * 70)

    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
