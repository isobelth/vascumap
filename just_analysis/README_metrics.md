# Vascular Metrics Reference

This file documents the streamlined outputs in `bel_skeletonisation.ipynb` intended for cross-chip comparison and pipeline integration.

Primary tables:
- `global_metrics_df`: compact sample-level output (recommended for downstream analysis)
- `vessel_metrics_df`: per-edge measurements (for QC and drill-down)
- `junction_metrics_df`: per-node measurements (for QC and drill-down)

The design principle is one biologically meaningful metric per concept, with shape normalization built in.

## Normalization and Units

Let:
- $V_{chip}$ = chip volume in $\mu m^3$
- $V_{vessel}$ = vessel volume in $\mu m^3$
- $L_c = V_{chip}^{1/3}$ = characteristic chip length in $\mu m$
- $A_c = L_c^2$ = characteristic chip area in $\mu m^2$

Voxel size is `2 x 2 x 2 um`, so each voxel is `8 um^3`.

## Compact Output Metrics (Global)

### 1) Volume Occupancy
- **Metric:** `vessel_volume_fraction`
- **Math:** $V_{vessel}/V_{chip}$
- **Biological meaning:** fraction of chip occupied by vasculature.
- **Why included:** core “how much vasculature formed” endpoint.

### 2) Network Density (Length)
- **Metric:** `vessel_length_per_chip_volume_um_inverse2`
- **Math:** $L_{total}/V_{chip}$
- **Biological meaning:** vessel packing density in 3D.
- **Why included:** normalizes for chip size and depth differences.

### 3) Sprout Density
- **Metric:** `sprouts_per_chip_volume_um_inverse3`
- **Math:** $N_{sprout}/V_{chip}$
- **Biological meaning:** terminal exploratory phenotype density.
- **Why included:** captures angiogenic branching front independent of chip size.

### 4) Junction Density
- **Metric:** `junctions_per_chip_volume_um_inverse3`
- **Math:** $N_{junction}/V_{chip}$
- **Biological meaning:** branching complexity density.
- **Why included:** compact topological summary of network connectivity.

### 5) Tortuosity
- **Metric:** `median_sprout_and_branch_tortuosity`
- **Math (per edge):** $\tau = L_{path}/L_{endpoints}$; reported as median over edges.
- **Biological meaning:** vessel winding/straightness.
- **Why included:** robust morphology descriptor of vessel remodeling.

### 6) Relative Vessel Thickness
- **Metric:** `median_cs_area_over_characteristic_area`
- **Math:** $\mathrm{median}(A_{cs})/A_c$
- **Biological meaning:** typical vessel thickness (cross-sectional size) relative to chip scale.
- **Why included:** prevents direct vessel-thickness comparison bias from differing device sizes.

### 7) Relative Junction Spacing
- **Metric:** `median_junction_dist_nearest_junction_per_characteristic_length`
- **Math:** $\mathrm{median}(d_{jj})/L_c$
- **Biological meaning:** typical branchpoint spacing at network scale.
- **Why included:** interpretable spacing metric that is size-invariant.

### 8) Relative Endpoint Spacing
- **Metric:** `median_sprout_dist_nearest_endpoint_per_characteristic_length`
- **Math:** $\mathrm{median}(d_{ee})/L_c$
- **Biological meaning:** how clustered terminal tips are.
- **Why included:** complements sprout count by adding spatial context.

### 9) Fractal Dimension
- **Metric:** `fractal_dimension`
- **Math:** slope from box-count relation $\log N(s)$ vs $\log(1/s)$.
- **Biological meaning:** degree of space-filling architecture of the segmented vascular volume.
- **Why included:** compact complexity descriptor beyond simple counts.

### 10) Lacunarity
- **Metric:** `lacunarity`
- **Math:** variance-to-mean-squared measure of box mass distribution.
- **Biological meaning:** heterogeneity / patchiness of the segmented vascular volume (how unevenly vessels occupy space).
- **Why included:** distinguishes patterns with similar fractal dimension but different texture.
- **Computed on:** the binary segmentation (`binary_filled_holes`), not on the skeleton.

### 11) Internal Pore Fraction
- **Metric:** `internal_pore_area_fraction_in_filled_vascular_area`
- **Math:** total valid pore area / total filled vascular area (slice-wise).
- **Biological meaning:** porousness of enclosed vascular voids.
- **Why included:** direct lumenal/discontinuity phenotype summary.

### 12) Internal Pore Density
- **Metric:** `total_internal_pore_density_per_vessel_volume_um_inverse3`
- **Math:** $N_{pores}/V_{vessel}$
- **Biological meaning:** number of pores per amount of vascular tissue.
- **Why included:** count normalized by actual vessel mass (not chip size).

### 13) Typical Pore Size
- **Metric:** `median_internal_pore_area_um2`
- **Math:** median of pore areas.
- **Biological meaning:** central tendency of enclosed void size.
- **Why included:** robust descriptor of typical pore scale.

### 14) Pore Size Heterogeneity
- **Metric:** `p90_minus_p10_internal_pore_area_um2`
- **Math:** $P90(\mathrm{pore\ area}) - P10(\mathrm{pore\ area})$.
- **Biological meaning:** robust spread of pore sizes across the distribution.
- **Why included:** captures heterogeneity with one number, without relying on SD.

### 15) Typical Pore Radius
- **Metric:** `median_internal_pore_max_inscribed_radius_um`
- **Math:** median of max inscribed radius per pore.
- **Biological meaning:** intrinsic pore size scale independent of shape irregularity.
- **Why included:** complements area with a radius-scale interpretation.

### 16) Pore Radius Heterogeneity
- **Metric:** `p90_minus_p10_internal_pore_max_inscribed_radius_um`
- **Math:** $P90(\mathrm{max\ inscribed\ radius}) - P10(\mathrm{max\ inscribed\ radius})$.
- **Biological meaning:** robust spread of pore radius scale.
- **Why included:** pairs with median radius to summarize shape variability minimally.

## Heterogeneity Policy (Median + Spread)

Where a metric comes from a distribution of per-edge, per-node, or per-pore values, the compact output now reports:
- a **median** (typical behavior), and
- a **p90-p10 spread** (distribution width).

This applies to:
- tortuosity,
- vessel cross-sectional area,
- nearest-junction spacing,
- nearest-endpoint spacing,
- pore area,
- and pore max-inscribed radius.

Count/density/fraction metrics (for example volume fraction or junction density) are already aggregate quantities, so they do not have separate median/spread counterparts.

## Junction Distance Mode: Euclidean vs Along-Skeleton

`compute_junction_metrics` supports two definitions for nearest-neighbor spacing:

- **Euclidean (`distance_mode='euclidean'`)**  
	Straight-line distance between node coordinates in physical space.
	- Pros: simple, fast, and easy to interpret geometrically.
	- Cons: can underestimate separation when paths are highly tortuous.

- **Along skeleton (`distance_mode='skeleton'`)**  
	Shortest-path distance constrained to the vascular graph (sum of edge path lengths).
	- Pros: follows biologically traversable vessel paths; better for network-routing interpretations.
	- Cons: computationally heavier and dependent on skeleton/graph quality.

Use one mode consistently across all samples in a study; mixing modes breaks comparability.

## Pore Exclusion Rule

All pore analyses exclude any pore whose area exceeds **10% of total slice area**.

- Rule: exclude pores with `area > 0.10 * slice_area`.
- This filter is applied to headline pore metrics, per-slice pore stats, and hole-visualization layers.
- Purpose: remove large non-biological cavities or boundary artifacts that can dominate pore summaries.

## Why Median + P90-P10

For integration and model features, median summaries are retained because they are robust to segmentation artifacts and rare extreme structures. The $P90-P10$ companion is included because it:
- captures both lower and upper distribution behavior in one scalar,
- is less sensitive than SD to a few extreme outliers,
- and keeps the descriptor count minimal (median + one heterogeneity term per concept).

## Is SD useful?

Standard deviation (SD) can be useful when your biological question is explicitly about heterogeneity (for example, treatment-driven increase in variability rather than shift in central tendency).

For this compact pipeline output, SD is excluded by default because:
- it expands feature count substantially,
- it is more sensitive to segmentation outliers,
- and median + $P90-P10$ captures most biologically relevant spread in a robust way.

If needed, SD can still be computed from `vessel_metrics_df` / `junction_metrics_df` in a downstream analysis notebook.

## Practical Notes

- Use `global_metrics_df` for inter-sample statistics and modeling.
- Use `vessel_metrics_df` / `junction_metrics_df` for diagnostics and mechanism exploration.
- Keep one distance mode (Euclidean vs skeleton geodesic) consistent across all compared samples.
