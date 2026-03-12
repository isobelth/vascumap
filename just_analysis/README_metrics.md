# Vascular Metrics Reference

This document reflects the current lean output set in `bel_skeletonisation.ipynb`.

Outputs:
- `global_metrics_df`: one-row summary for sample-level comparison
- `vessel_metrics_df`: one row per cleaned vessel edge
- `junction_metrics_df`: one row per cleaned graph node

The aim is to keep a compact set of mostly non-redundant descriptors that still captures the main biological properties of each vascular network.

## Normalization Basis

- Chip volume is defined as the full volume of the `vasculature_segmentation` image.
- The voxel size is `2 x 2 x 2 um`.
- One voxel therefore corresponds to `8 um^3`.
- All retained headline outputs are now reported in physical units only.
- Chip volume is stored as `chip_volume_um3`.
- Vessel occupancy volume is stored as `vessel_volume_um3`.
- Length quantities are stored in `um`.
- Cross-sectional areas are stored in `um^2`.
- Density terms are reported in their derived physical units.

## 1) Detailed Tables

### Vessel Metrics (`vessel_metrics_df`)

Each row corresponds to one cleaned graph edge.

Key per-edge fields retained in the detailed table:
- `length`: segment extent in `um`
- `tortuosity`: segment winding
- `median_cs_area`: typical local caliber in `um^2`
- `volume`: segment volume proxy in `um^3`
- `is_sprout`: terminal sprout versus internal branch

### Junction Metrics (`junction_metrics_df`)

Each row corresponds to one cleaned graph node.

Key per-node fields retained in the detailed table:
- `number_of_vessel_per_node`: local connectivity
- `dist_nearest_junction`: local branchpoint spacing in `um`
- `dist_nearest_endpoint`: local endpoint spacing in `um`
- `num_junction_neighbors`: local junction crowding
- `num_endpoint_neighbors`: local sprout crowding

## 2) Lean Global Metric Set (`global_metrics_df`)

These are the current headline outputs used to define and compare vasculatures.

| Parameter | Why it is kept |
|---|---|
| `chip_volume_um3` | Records the analysed image volume in physical units. |
| `vessel_volume_um3` | Records total occupied vessel volume in physical units. |
| `vessel_volume_fraction` | Measures vascular occupancy normalized by chip volume. |
| `total_vessel_length_um` | Captures the overall network extent in `um`. |
| `vessel_length_per_chip_volume_um_inverse2` | Measures vessel length density in `um / um^3 = um^-2`. |
| `sprouts_per_vessel_length_um_inverse` | Captures terminal sprouting normalized by network length in `um^-1`. |
| `junctions_per_vessel_length_um_inverse` | Captures branching density normalized by network extent in `um^-1`. |
| `median_sprout_and_branch_tortuosity` | Represents the typical winding of vessel segments. |
| `median_sprout_and_branch_median_cs_area_um2` | Represents the typical vessel caliber in `um^2`. |
| `median_junction_dist_nearest_junction_um` | Represents the typical spacing between branchpoints in `um`. |
| `median_sprout_dist_nearest_endpoint_um` | Represents the typical spacing in sprout-dense terminal regions in `um`. |
| `fractal_dimension` | Describes space-filling architectural complexity. |
| `lacunarity` | Describes heterogeneity and gappiness of the vascular pattern. |

## 3) Interpretation Guide

- `vessel_volume_fraction` answers: how much of the chip volume is occupied by vessels?
- `vessel_length_per_chip_volume_um_inverse2` answers: how densely packed is the network per unit chip volume?
- `sprouts_per_vessel_length_um_inverse` answers: how terminal or exploratory is the network per unit vessel length?
- `junctions_per_vessel_length_um_inverse` answers: how strongly branched is the network per unit vessel length?
- `median_sprout_and_branch_tortuosity` answers: how straight or winding are typical vessels?
- `median_sprout_and_branch_median_cs_area_um2` answers: how thick are typical vessels?
- `median_junction_dist_nearest_junction_um` answers: how tightly spaced are branchpoints?
- `median_sprout_dist_nearest_endpoint_um` answers: how tightly spaced are terminal endpoints?
- `fractal_dimension` answers: how space-filling is the network architecture?
- `lacunarity` answers: how uneven or patchy is the vascular distribution?

## 4) Why This Set Is Leaner

- Raw counts such as total sprouts, total branches, and total junctions were removed from the headline output because they are strongly size-dependent and overlap conceptually with the normalized density terms.
- Large grouped summary blocks were removed because many of them described nearly the same biological property from slightly different angles.
- The retained set keeps one main descriptor each for occupancy, density, branching, sprouting, caliber, tortuosity, spacing, and architecture.

## 5) Units and Comparison Notes

- The segmentation voxel size is `2 x 2 x 2 um`.
- `chip_volume_um3` and `vessel_volume_um3` are reported directly in physical volume units.
- `total_vessel_length_um` and the nearest-distance metrics are reported directly in `um`.
- `median_sprout_and_branch_median_cs_area_um2` is reported in `um^2`.
- `vessel_length_per_chip_volume_um_inverse2` has units of `um^-2`.
- `sprouts_per_vessel_length_um_inverse` and `junctions_per_vessel_length_um_inverse` have units of `um^-1`.

This means the retained comparison metrics are now fully physical-unit outputs rather than voxel-unit proxies.
