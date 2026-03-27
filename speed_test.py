"""speed_test.py — Run pipeline on one image and optionally compare against a baseline.

Usage:
    # Run a labelled pipeline pass
    python speed_test.py --label baseline
    python speed_test.py --label phase_A --compare baseline

    # All outputs land in Z:\Bel\Vascumap_Speed_Test\<label>\
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "bel_vascumap"))

TEST_LIF      = r"Z:\Bel\Farid_bel\Old experiment\03.10.25 static 6million day 7.lif"
TEST_IMG_IDX  = 0
OUTPUT_BASE   = r"Z:\Bel\Vascumap_Speed_Test"
DEVICE_WIDTH  = 35.0
HOUGH         = 400
CHANNEL       = 0


def run_pipeline(label: str):
    from models import Pix2Pix, load_segmentation_model
    from vascumap import VascuMap

    model_root = Path(__file__).parent / "luca_models"
    model_p2p  = Pix2Pix(model_path=str(model_root / "epoch=117-val_g_psnr=20.47-val_g_ssim=0.62.ckpt"))
    model_unet = load_segmentation_model(str(model_root / "best_full.pth"))
    print("Models loaded.")

    vm = VascuMap(
        use_device_segmentation_app=False,
        image_source_path=TEST_LIF,
        image_index=TEST_IMG_IDX,
        device_width_um=DEVICE_WIDTH,
        hough_line_length=HOUGH,
        mask_central_region=False,
        channel=CHANNEL,
        model_p2p=model_p2p,
        model_unet=model_unet,
    )
    vm.image_name = f"speed_test_img{TEST_IMG_IDX}"

    out_dir = Path(OUTPUT_BASE) / label / vm.image_name
    print(f"Output → {out_dir}")
    vm.pipeline(output_dir=out_dir, save_all_interim=True)
    print(f"\n✓ {label} run complete → {out_dir}")
    return out_dir


def compare(label_a: str, label_b: str, rtol: float = 1e-4, atol: float = 1e-6):
    """Compare key outputs between two labelled runs."""
    base_a = Path(OUTPUT_BASE) / label_a
    base_b = Path(OUTPUT_BASE) / label_b

    def find_output(base):
        candidates = [p for p in base.glob("speed_test_img*") if p.is_dir()]
        if not candidates:
            raise FileNotFoundError(f"No outputs found under {base}")
        return sorted(candidates)[-1]

    dir_a = find_output(base_a)
    dir_b = find_output(base_b)
    print(f"\nComparing:\n  A: {dir_a}\n  B: {dir_b}\n")

    all_ok = True

    # ── Metrics CSV ──────────────────────────────────────────────────────────
    csv_a = list(dir_a.glob("*_analysis_metrics.csv"))
    csv_b = list(dir_b.glob("*_analysis_metrics.csv"))
    if csv_a and csv_b:
        df_a = pd.read_csv(csv_a[0]).select_dtypes(include="number")
        df_b = pd.read_csv(csv_b[0]).select_dtypes(include="number")
        for col in df_a.columns:
            if col not in df_b.columns:
                print(f"  [WARN] column '{col}' missing in B")
                continue
            va, vb = float(df_a[col].iloc[0]), float(df_b[col].iloc[0])
            if np.isnan(va) and np.isnan(vb):
                pass  # both NaN → OK
            elif abs(va - vb) > atol + rtol * max(abs(va), abs(vb), 1e-12):
                print(f"  [FAIL] {col}: A={va:.6g}  B={vb:.6g}  diff={abs(va-vb):.2e}")
                all_ok = False
            else:
                print(f"  [OK]   {col}: {va:.6g}")
    else:
        print("  [WARN] Could not find analysis_metrics.csv in one or both runs")
        all_ok = False

    # ── .npy arrays ──────────────────────────────────────────────────────────
    npy_checks = [
        "_clean_segmentation.npy",
        "_skeleton.npy",
        "_vessel_mask.npy",
    ]
    for suffix in npy_checks:
        files_a = list(dir_a.glob(f"*{suffix}"))
        files_b = list(dir_b.glob(f"*{suffix}"))
        if not files_a or not files_b:
            print(f"  [SKIP] {suffix} not found")
            continue
        arr_a = np.load(files_a[0])
        arr_b = np.load(files_b[0])
        if arr_a.shape != arr_b.shape:
            print(f"  [FAIL] {suffix}: shape mismatch {arr_a.shape} vs {arr_b.shape}")
            all_ok = False
            continue
        if not np.allclose(arr_a.astype(float), arr_b.astype(float), rtol=rtol, atol=atol):
            diff = np.abs(arr_a.astype(float) - arr_b.astype(float))
            print(f"  [FAIL] {suffix}: max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}")
            all_ok = False
        else:
            print(f"  [OK]   {suffix}: arrays match (shape {arr_a.shape})")

    print()
    if all_ok:
        print("✓ ALL CHECKS PASSED — outputs are equivalent.")
    else:
        print("✗ SOME CHECKS FAILED — review diffs above before committing.")
    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, help="Label for this run, e.g. 'baseline' or 'phase_A'")
    parser.add_argument("--compare", default=None, help="Label of run to compare against, e.g. 'baseline'")
    parser.add_argument("--compare-only", action="store_true", help="Skip pipeline run, only compare")
    args = parser.parse_args()

    if not args.compare_only:
        run_pipeline(args.label)

    if args.compare:
        compare(args.compare, args.label)
