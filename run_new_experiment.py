"""Batch run for New experiment folder with per-image timing output."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bel_vascumap"))

import numpy as np
import pandas as pd
from liffile import LifFile
from models import Pix2Pix, load_segmentation_model
from vascumap import VascuMap

SOURCE_DIR   = r"Z:\Bel\Farid_bel\New experiment"
OUTPUT_BASE  = r"Z:\Bel\Farid_bel\New_Experiment_Outputs"
DEVICE_WIDTH = 35.0
HOUGH        = 400
CHANNEL      = 0
MAX_RETRIES  = 2
REQUIRE_MERGED = False
FORCE_MASK   = None   # None = auto-detect from filename

MODEL_P2P_PATH  = r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt"
MODEL_UNET_PATH = r"C:\Users\taylorhearn\git_repos\vascumap\luca_models\best_full.pth"


def _auto_mask_mode(p: Path, img_name: str = ""):
    name = (p.name + " " + img_name).lower()
    if "marina" in name and "bead" in name:
        return "light"
    if "marina" in name:
        return "dark"
    return False


def discover_jobs(source_dir):
    src = Path(source_dir)
    jobs = []
    for p in sorted(p for p in src.iterdir() if p.suffix.lower() in (".lif", ".tif", ".tiff")):
        if p.suffix.lower() == ".lif":
            try:
                with LifFile(p) as lif:
                    for idx in range(len(lif.images)):
                        img_name = getattr(lif.images[idx], "name", "")
                        if REQUIRE_MERGED and "merged" not in img_name.lower():
                            continue
                        mask = FORCE_MASK if FORCE_MASK is not None else _auto_mask_mode(p, img_name)
                        jobs.append((p, idx, mask))
            except Exception as exc:
                print(f"[SKIP] {p.name}: {exc}")
        else:
            if REQUIRE_MERGED and "merged" not in p.name.lower():
                continue
            mask = FORCE_MASK if FORCE_MASK is not None else _auto_mask_mode(p)
            jobs.append((p, 0, mask))
    return jobs


def main():
    print("Loading models...")
    model_p2p  = Pix2Pix(model_path=MODEL_P2P_PATH)
    model_unet = load_segmentation_model(MODEL_UNET_PATH)
    print("Models loaded.")

    jobs = discover_jobs(SOURCE_DIR)
    print(f"\nTotal jobs: {len(jobs)}")
    for i, (p, idx, mask) in enumerate(jobs, 1):
        print(f"  {i}. {p.name} [image {idx}]  mask={mask}")

    timing_rows = []
    results = []

    for i, (image_path, image_index, mask_flag) in enumerate(jobs, 1):
        tag = f" (LIF #{image_index})"
        print(f"\n{'='*70}")
        print(f"[{i}/{len(jobs)}] {image_path.name}{tag}  mask={mask_flag}")
        print(f"{'='*70}")

        last_exc = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                _t_job = time.time()
                vm = VascuMap(
                    use_device_segmentation_app=False,
                    image_source_path=str(image_path),
                    image_index=image_index,
                    device_width_um=DEVICE_WIDTH,
                    hough_line_length=HOUGH,
                    mask_central_region=mask_flag,
                    channel=CHANNEL,
                    model_p2p=model_p2p,
                    model_unet=model_unet,
                )
                vm.image_name = f"{image_path.stem}_img{image_index}_{vm.image_name or 'image'}"
                out_dir = Path(OUTPUT_BASE) / vm.image_name
                print(f"  Output → {out_dir}")
                vm.pipeline(output_dir=out_dir, save_all_interim=False)
                _t_job_wall = time.time() - _t_job

                results.append((vm.image_name, "OK", ""))
                timing_rows.append({
                    "image_name":     vm.image_name,
                    "source_file":    image_path.name,
                    "image_index":    image_index,
                    "status":         "OK",
                    "t_device_seg_s": round(getattr(vm, "_t_device_seg", 0), 1),
                    "t_preprocess_s": round(getattr(vm, "_t_preprocess", 0), 1),
                    "t_inference_s":  round(getattr(vm, "_t_inference", 0), 1),
                    "t_analysis_s":   round(getattr(vm, "_t_analysis", 0), 1),
                    "t_pipeline_s":   round(getattr(vm, "_t_total", 0), 1),
                    "t_job_wall_s":   round(_t_job_wall, 1),
                })
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES:
                    print(f"  ⚠ Attempt {attempt} failed: {exc} — retrying...")
                else:
                    print(f"  ✗ FAILED after {MAX_RETRIES} attempts: {exc}")

        if last_exc is not None:
            results.append((image_path.name + tag, "FAILED", str(last_exc)))
            timing_rows.append({
                "image_name":     image_path.name + tag,
                "source_file":    image_path.name,
                "image_index":    image_index,
                "status":         "FAILED",
                "t_device_seg_s": None, "t_preprocess_s": None,
                "t_inference_s":  None, "t_analysis_s":   None,
                "t_pipeline_s":   None, "t_job_wall_s":   None,
            })

        # Save timing CSV after every image (so partial results survive crashes)
        if timing_rows:
            timing_df = pd.DataFrame(timing_rows)
            csv_path = Path(OUTPUT_BASE) / "batch_timings.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            timing_df.to_csv(csv_path, index=False)

    # Final summary
    print(f"\n{'='*70}")
    n_ok = sum(1 for _, s, _ in results if s == "OK")
    print(f"Batch complete: {n_ok}/{len(results)} succeeded")

    timing_df = pd.DataFrame(timing_rows)
    csv_path = Path(OUTPUT_BASE) / "batch_timings.csv"
    timing_df.to_csv(csv_path, index=False)
    print(f"\nTiming breakdown saved → {csv_path}")
    print(timing_df[[
        "image_name", "t_device_seg_s", "t_preprocess_s",
        "t_inference_s", "t_analysis_s", "t_pipeline_s"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
