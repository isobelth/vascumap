"""Job discovery and batch execution helpers for the VascuMap pipeline."""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from liffile import LifFile

from core import VascuMap


def determine_mask_mode(file_path: Path, image_name: str = "", force_mask: Optional[str] = None):
    """Infer organoid mask mode from filename keywords, or use a forced override.

    When force_mask is given it is returned directly. Otherwise the filename
    and image name are checked for the keywords 'marina' and 'bead' to
    determine whether a dark or light organoid mask is appropriate.
    Returns 'dark', 'light', or False (no masking).
    """
    if force_mask is not None:
        return force_mask
    combined = (file_path.name + " " + image_name).lower()
    if "marina" in combined and "bead" in combined:
        return "light"
    return "dark" if "marina" in combined else False


def discover_jobs(source_dir: str | Path, force_mask: Optional[str] = None, require_merged: bool = True) -> Tuple[Path, List[Path], List[tuple]]:
    """Scan source_dir for .lif/.tif/.tiff files and build a list of pipeline jobs.

    Each job is a tuple of (file_path, image_index, mask_flag, image_name).
    LIF files are expanded to one job per contained image (filtered by
    'merged' keyword when require_merged is True).
    """
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"source_dir does not exist: {source}")
    image_files = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in (".lif", ".tif", ".tiff"))

    jobs: list[tuple] = []
    for file_path in image_files:
        if file_path.suffix.lower() == ".lif":
            try:
                with LifFile(file_path) as lif:
                    for idx, image in enumerate(lif.images):
                        image_name = getattr(image, "name", "")
                        if require_merged and "merged" not in image_name.lower():
                            continue
                        jobs.append((file_path, idx, determine_mask_mode(file_path, image_name, force_mask), image_name))
            except Exception as exc:
                print(f"[SKIP] {file_path.name}: {exc}")
        else:
            if require_merged and "merged" not in file_path.name.lower():
                continue
            jobs.append((file_path, 0, determine_mask_mode(file_path, force_mask=force_mask), file_path.stem))
    return source, image_files, jobs


def expected_output_name(file_path: str | Path, image_index: int, image_name: str) -> str:
    """Return the output folder name the pipeline will use for a given job."""
    file_path = Path(file_path)
    if file_path.suffix.lower() == ".lif":
        safe_name = image_name.replace("/", "_").replace("\\", "_")
        return f"{file_path.stem}_{safe_name}_img{image_index}"
    return file_path.stem


def filter_jobs(jobs: Sequence[tuple], skip_names: Iterable[str]) -> List[tuple]:
    """Remove jobs whose expected output name already exists in skip_names."""
    skip_set: Set[str] = set(skip_names)
    kept: list[tuple] = []
    skipped = 0
    for job in jobs:
        file_path, image_index, mask_flag, image_name = job
        if expected_output_name(file_path, image_index, image_name) in skip_set:
            skipped += 1
        else:
            kept.append(job)
    if skipped:
        print(f"Filtered out {skipped} already-processed images, {len(kept)} remaining.")
    return kept


def run_batch_from_curation(curated_jobs, output_base: str | Path, save_all_interim: bool = False, model_p2p=None, model_unet=None):
    """Run the full pipeline on jobs that have been curated via the napari GUI."""
    results = []
    Path(output_base).mkdir(parents=True, exist_ok=True)
    processable = [j for j in curated_jobs if j.status == "curated" and hasattr(j, "finalised_outputs") and j.finalised_outputs is not None]
    for i, job in enumerate(processable, 1):
        name = expected_output_name(job.source_path, job.image_index, job.image_name)
        print(f"\n{'=' * 70}\n[{i}/{len(processable)}] {name}\n{'=' * 70}")
        try:
            vm = VascuMap(curated_outputs=job.finalised_outputs, model_p2p=model_p2p, model_unet=model_unet)
            vm.image_name = name
            output_dir = Path(output_base) / name
            print(f"  Output → {output_dir}")
            vm.pipeline(output_dir=output_dir, save_all_interim=save_all_interim)
            results.append((name, "OK", ""))
        except Exception as exc:
            print(f"  [SKIP] {exc}")
            results.append((name, "FAILED", str(exc)))
    succeeded = sum(1 for _, status, _ in results if status == "OK")
    skipped_curation = sum(1 for j in curated_jobs if j.status == "skip")
    print(f"\n{'=' * 70}\nBatch complete: {succeeded}/{len(results)} succeeded, {skipped_curation} skipped (curation)")
    for name, status, msg in results:
        if status == "FAILED":
            print(f"  FAILED: {name}: {msg}")
    return results


def run_batch(jobs: Sequence[tuple], output_base: str | Path, device_width_um: float, brightfield_channel: int = 0, save_all_interim: bool = False, model_p2p=None, model_unet=None, start_index: int = 1):
    """Run all jobs in headless automatic mode without any GUI curation."""
    results = []
    Path(output_base).mkdir(parents=True, exist_ok=True)
    failure_diag_dir = Path(output_base) / "FAILED_diagnostics"
    for i, (file_path, image_index, mask_flag, image_name) in enumerate(jobs, start_index):
        name = expected_output_name(file_path, image_index, image_name)
        lif_tag = f" (LIF #{image_index}: {image_name})" if file_path.suffix.lower() == ".lif" else ""
        print(f"\n{'=' * 70}\n[{i}/{len(jobs)}] {file_path.name}{lif_tag}  mask={mask_flag}\n{'=' * 70}")
        try:
            vm = VascuMap(image_source_path=str(file_path), image_index=image_index, device_width_um=device_width_um, mask_central_region=mask_flag, brightfield_channel=brightfield_channel, model_p2p=model_p2p, model_unet=model_unet, failure_output_dir=str(failure_diag_dir))
            vm.image_name = name
            output_dir = Path(output_base) / name
            print(f"  Output → {output_dir}")
            vm.pipeline(output_dir=output_dir, save_all_interim=save_all_interim)
            results.append((name, "OK", ""))
        except Exception as exc:
            print(f"  [SKIP] {exc}")
            results.append((file_path.name + lif_tag, "FAILED", str(exc)))
    succeeded = sum(1 for _, status, _ in results if status == "OK")
    print(f"\n{'=' * 70}\nBatch complete: {succeeded}/{len(results)} succeeded")
    for name, status, msg in results:
        if status == "FAILED":
            print(f"  FAILED: {name}: {msg}")
    return results
