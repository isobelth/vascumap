# Vascular Metrics Reference (`global_metrics_df`)

This document is a one-to-one reference for **all columns currently emitted** by `bel_skeletonisation.ipynb` in the single-row output dataframe `global_metrics_df`.

## Scope and Definitions

All values are sample-level metrics derived from the final cleaned segmentation/skeleton/graph.

Symbols used below:
- $V_{chip}$: chip volume ($\mu m^3$)
- $V_{vessel}$: segmented vessel volume ($\mu m^3$)
- $L_{total}$: total centerline length of all graph edges ($\mu m$)
- $N_{junction}$: number of non-sprout graph nodes
- $N_{sprout}$: number of graph edges touching at least one sprout node
- $L_c = V_{chip}^{1/3}$: characteristic chip length ($\mu m$)
- $A_c = L_c^2$: characteristic chip area ($\mu m^2$)

Default voxel size in the notebook is `(2, 2, 2) um`.

## Distance Convention (important)

The notebook currently sets `junction_distance_mode = 'skeleton'`.

- `skeleton`: nearest-neighbor distances are graph shortest-path lengths along vessel centerlines (biologically traversable route).
- `euclidean`: nearest-neighbor distances are straight-line distances in physical space.

Use one mode consistently across a study.

## Complete Output Parameter Dictionary

| Column | Units | Mathematical meaning | Biological interpretation |
|---|---:|---|---|
| `chip_volume_um3` | $\mu m^3$ | $V_{chip}$ | Physical assay/imaged volume used for normalization. |
| `vessel_volume_um3` | $\mu m^3$ | $V_{vessel}$ from vessel-positive voxels | Total vascular biomass/occupancy in the sample. |
| `vessel_volume_fraction` | unitless | $V_{vessel}/V_{chip}$ | Fraction of chip occupied by vessels. |
| `total_vessel_length_um` | $\mu m$ | $L_{total}$ from summed edge polyline lengths | Total vascular extent/coverage by centerline length. |
| `vessel_length_per_chip_volume_um_inverse2` | $\mu m^{-2}$ | $L_{total}/V_{chip}$ | 3D vessel packing density (length density). |
| `sprouts_per_vessel_length_um_inverse` | $\mu m^{-1}$ | $N_{sprout}/L_{total}$ | Sprouting intensity relative to network size. |
| `junctions_per_vessel_length_um_inverse` | $\mu m^{-1}$ | $N_{junction}/L_{total}$ | Branching intensity per unit vessel length. |
| `fractal_dimension` | unitless | Box-counting slope of segmented vascular mask | Space-filling geometric complexity of vasculature. |
| `lacunarity` | unitless | Gap/heterogeneity statistic from box-mass distribution | Patchiness/inhomogeneity of vessel occupancy. |
| `median_sprout_and_branch_tortuosity` | unitless | Median of per-edge $\tau = L_{path}/L_{endpoints}$ (clipped to [0,5]) | Typical vessel winding vs straightness. |
| `p90_minus_p10_sprout_and_branch_tortuosity` | unitless | $P90(\tau)-P10(\tau)$ | Heterogeneity of vessel tortuosity. |
| `median_sprout_and_branch_median_cs_area_um2` | $\mu m^2$ | Median of per-edge median sampled cross-sectional area values | Typical vessel caliber (thickness proxy). |
| `p90_minus_p10_sprout_and_branch_median_cs_area_um2` | $\mu m^2$ | Spread $P90-P10$ of per-edge median cross-sectional area | Heterogeneity of vessel caliber. |
| `median_junction_dist_nearest_junction_um` | $\mu m$ | Median nearest-neighbor junction distance | Typical branchpoint spacing. |
| `p90_minus_p10_junction_dist_nearest_junction_um` | $\mu m$ | Spread $P90-P10$ of nearest junction distances | Heterogeneity of branchpoint spacing. |
| `median_sprout_dist_nearest_endpoint_um` | $\mu m$ | Median nearest-neighbor sprout-endpoint distance | Typical tip-to-tip spacing (terminal clustering). |
| `p90_minus_p10_sprout_dist_nearest_endpoint_um` | $\mu m$ | Spread $P90-P10$ of nearest sprout distances | Heterogeneity of endpoint spacing. |
| `total_internal_pore_count` | count | Number of valid internal pores across slices | Total enclosed void events in vascular area. |
| `internal_pore_area_fraction_in_filled_vascular_area` | unitless | Total valid pore area / total hole-filled vascular area | Fraction of vascular area that is porous/voided. |
| `median_internal_pore_area_um2` | $\mu m^2$ | Median valid pore area | Typical pore size by area. |
| `p90_minus_p10_internal_pore_area_um2` | $\mu m^2$ | Spread $P90-P10$ of pore area | Heterogeneity of pore area. |
| `median_internal_pore_max_inscribed_radius_um` | $\mu m$ | Median pore max-inscribed radius | Typical intrinsic pore radius scale. |
| `p90_minus_p10_internal_pore_max_inscribed_radius_um` | $\mu m$ | Spread $P90-P10$ of max-inscribed radius | Heterogeneity of pore radius scale. |
| `chip_characteristic_length_um` | $\mu m$ | $L_c = V_{chip}^{1/3}$ | Reference length for shape-invariant scaling. |
| `total_vessel_length_per_chip_characteristic_length` | unitless | $L_{total}/L_c$ | Vessel extent normalized to chip linear scale. |
| `sprouts_per_chip_volume_um_inverse3` | $\mu m^{-3}$ | $N_{sprout}/V_{chip}$ | Sprout density per chip volume. |
| `junctions_per_chip_volume_um_inverse3` | $\mu m^{-3}$ | $N_{junction}/V_{chip}$ | Junction density per chip volume. |
| `median_junction_dist_nearest_junction_per_characteristic_length` | unitless | `median_junction_dist_nearest_junction_um / L_c` | Size-invariant typical branchpoint spacing. |
| `p90_minus_p10_junction_dist_nearest_junction_per_characteristic_length` | unitless | `p90_minus_p10_junction_dist_nearest_junction_um / L_c` | Size-invariant heterogeneity of branchpoint spacing. |
| `median_sprout_dist_nearest_endpoint_per_characteristic_length` | unitless | `median_sprout_dist_nearest_endpoint_um / L_c` | Size-invariant typical endpoint spacing. |
| `p90_minus_p10_sprout_dist_nearest_endpoint_per_characteristic_length` | unitless | `p90_minus_p10_sprout_dist_nearest_endpoint_um / L_c` | Size-invariant heterogeneity of endpoint spacing. |
| `median_cs_area_over_characteristic_area` | unitless | `median_sprout_and_branch_median_cs_area_um2 / A_c` | Size-invariant typical vessel caliber. |
| `p90_minus_p10_cs_area_over_characteristic_area` | unitless | `p90_minus_p10_sprout_and_branch_median_cs_area_um2 / A_c` | Size-invariant heterogeneity of vessel caliber. |
| `total_internal_pore_density_per_vessel_volume_um_inverse3` | $\mu m^{-3}$ | `total_internal_pore_count / V_{vessel}` | Pore-event burden relative to vascular biomass. |

## Pore Inclusion/Exclusion Rules

Pores are detected slice-wise as internal holes in `binary_fill_holes(vessel_slice) & ~vessel_slice`.

Validity filters used by the current code:
- minimum pore area: `min_pore_area_um2 = 16.0`
- maximum pore area: `max_pore_area_fraction_of_slice = 0.10` (10% of slice area)

This suppresses tiny noise and very large likely-artifactual cavities.

## Missing-Value Behavior (NaN)

Some columns are `NaN` when the required structure is absent, for example:
- no valid edges/nodes for tortuosity and nearest-distance distributions,
- no valid pores for pore-size/radius distribution summaries,
- divisions where denominator is effectively zero.

This is expected behavior and indicates the metric is undefined for that sample, not a computation crash.

## Recommended Use

- Use `global_metrics_df` as the authoritative per-sample feature row.
- For cross-chip comparisons, prefer the shape-invariant normalized columns when chip volumes differ.
- Keep distance mode (`skeleton` or `euclidean`) fixed for all samples in one analysis.
