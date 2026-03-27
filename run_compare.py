"""Standalone comparison runner."""
import sys
sys.path.insert(0, r'C:\Users\taylorhearn\git_repos\vascumap')

from pathlib import Path
import numpy as np
import pandas as pd

OUTPUT_BASE = r'Z:\Bel\Vascumap_Speed_Test'
rtol = 1e-4
atol = 1e-6

base_a = Path(OUTPUT_BASE) / 'baseline'
base_b = Path(OUTPUT_BASE) / 'optimised'

def find_output(base):
    candidates = [p for p in base.glob('speed_test_img*') if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f'No outputs found under {base}')
    return sorted(candidates)[-1]

dir_a = find_output(base_a)
dir_b = find_output(base_b)
print(f'A: {dir_a}')
print(f'B: {dir_b}')
print(f'Files A: {len(list(dir_a.iterdir()))}  Files B: {len(list(dir_b.iterdir()))}')
print()

all_ok = True

# CSV metrics
csv_a = list(dir_a.glob('*_analysis_metrics.csv'))
csv_b = list(dir_b.glob('*_analysis_metrics.csv'))
if csv_a and csv_b:
    df_a = pd.read_csv(csv_a[0]).select_dtypes(include='number')
    df_b = pd.read_csv(csv_b[0]).select_dtypes(include='number')
    for col in df_a.columns:
        if col not in df_b.columns:
            print(f'  [WARN] column {col!r} missing in B')
            continue
        va, vb = float(df_a[col].iloc[0]), float(df_b[col].iloc[0])
        if np.isnan(va) and np.isnan(vb):
            print(f'  [OK]   {col}: both NaN')
        elif abs(va - vb) > atol + rtol * max(abs(va), abs(vb), 1e-12):
            print(f'  [FAIL] {col}: A={va:.6g}  B={vb:.6g}  diff={abs(va-vb):.2e}')
            all_ok = False
        else:
            print(f'  [OK]   {col}: {va:.6g}')
else:
    print(f'  [WARN] CSV missing — csv_a={csv_a}  csv_b={csv_b}')
    all_ok = False

# npy arrays
for suffix in ['_clean_segmentation.npy', '_skeleton.npy', '_vessel_mask.npy']:
    files_a = list(dir_a.glob(f'*{suffix}'))
    files_b = list(dir_b.glob(f'*{suffix}'))
    if not files_a or not files_b:
        print(f'  [SKIP] {suffix} not found (a={files_a} b={files_b})')
        continue
    arr_a = np.load(files_a[0])
    arr_b = np.load(files_b[0])
    if arr_a.shape != arr_b.shape:
        print(f'  [FAIL] {suffix}: shape {arr_a.shape} vs {arr_b.shape}')
        all_ok = False
        continue
    if not np.allclose(arr_a.astype(float), arr_b.astype(float), rtol=rtol, atol=atol):
        diff = np.abs(arr_a.astype(float) - arr_b.astype(float))
        print(f'  [FAIL] {suffix}: max_diff={diff.max():.4e}  mean_diff={diff.mean():.4e}')
        all_ok = False
    else:
        print(f'  [OK]   {suffix}: match  shape={arr_a.shape}')

print()
print('ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED')
