# Vascular Metrics Reference

This document reflects the current compact output set in `bel_skeletonisation.ipynb`.

Primary outputs:
- `global_metrics_df`: one-row sample-level comparison table
- `vessel_metrics_df`: one row per cleaned vessel edge
- `junction_metrics_df`: one row per cleaned graph node

The goal is one metric per biological concept, with shape-invariant normalization emphasized.

## Normalization Basis

- Voxel size: `2 x 2 x 2 um` (`8 um^3` per voxel).
- Chip volume: `chip_volume_um3 = np.prod(shape) * voxel_volume_um3`.
- Vessel volume: `vessel_volume_um3 = occupied_voxels * voxel_volume_um3`.
- Characteristic length: `chip_characteristic_length_um = chip_volume_um3^(1/3)`.
- Characteristic area: `chip_characteristic_area_um2 = chip_characteristic_length_um^2`.

## Compact Headline Metrics (`global_metrics_df`)

| Metric | Interpretation |
|---|---|
| `vessel_volume_fraction` | Vessel occupancy, normalized to chip volume. |
| `vessel_length_per_chip_volume_um_inverse2` | Vessel length density per chip volume. |
| `sprouts_per_chip_volume_um_inverse3` | Sprout endpoint density per chip volume. |
| `junctions_per_chip_volume_um_inverse3` | Junction density per chip volume. |
| `median_sprout_and_branch_tortuosity` | Typical segment winding (shape descriptor). |
| `median_cs_area_over_characteristic_area` | Typical vessel caliber, normalized for chip size. |
| `median_junction_dist_nearest_junction_per_characteristic_length` | Typical branchpoint spacing, size-normalized. |
| `median_sprout_dist_nearest_endpoint_per_characteristic_length` | Typical endpoint spacing, size-normalized. |
| `fractal_dimension` | Space-filling network complexity. |
| `lacunarity` | Pattern heterogeneity / gappiness. |
| `internal_pore_area_fraction_in_filled_vascular_area` | Porosity fraction within filled vascular area. |
| `total_internal_pore_density_per_vessel_volume_um_inverse3` | Internal pore count density per vessel volume. |
| `median_internal_pore_area_um2` | Typical enclosed pore area. |
| `p90_internal_pore_area_um2` | Upper-tail enclosed pore area. |
| `median_internal_pore_max_inscribed_radius_um` | Typical enclosed pore radius scale. |

## Do you need mean, std, and median for all metrics?

No. For most vasculature metrics, that is redundant and inflates output width.

Recommended default for headline reporting:
- Keep **median only** for continuous biological descriptors (robust to outliers).
- Keep **one upper-tail metric** (for pores, `p90`) when tail behavior is biologically relevant.
- Use mean/std only for a specific downstream statistical reason (not by default in the summary table).

In this notebook, grouped summary output has been slimmed to median-only (`add_grouped_stats`).

## Detailed Tables (for drill-down)

`vessel_metrics_df` and `junction_metrics_df` still contain richer per-edge and per-node measurements for diagnostics and custom analysis. The compact `global_metrics_df` is intended for cross-chip comparison and model-ready downstream use.
