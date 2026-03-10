# Vascular Metrics Reference (Slimmed Set)

This document reflects the current slimmed metric outputs in `bel_skeletonisation.ipynb`.

Outputs:
- `global_metrics` (one-row summary for sample-level comparison)
- `vessel_metrics_df` (one row per vessel edge)
- `junction_metrics_df` (one row per graph node)

The goal is to keep mostly non-redundant descriptors for contrasting segmentations.

## 1) Vessel Metrics (`vessel_metrics_df`)

Each row corresponds to one graph edge.

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `z`, `y`, `x` | Arrays of voxel coordinates along the edge polyline. | 3D vessel centerline trajectory. |
| `volume` | Estimated physical segment volume in um^3, computed as `mean(cross_section_area_um2 along edge) * length_um`. | Estimated vessel segment volume in physical units. |
| `length` | Polyline arclength in um (centerline points scaled by voxel size). | Physical vessel segment extent. |
| `tortuosity` | `length / endpoint_chord_length`, clipped to `[0, 5]`. | Curviness/winding of the segment. Near 1 is straighter. |
| `is_sprout` | True if either endpoint node has degree 1. | Distinguishes terminal sprouts from internal branch segments. |
| `median_cs_area` | Median of EDT-derived cross-sectional area estimate along segment (um^2). | Median cross-sectional area of vessel. |
| `std_cs_area` | Standard deviation of cross-sectional area estimate along segment (um^2). | Heterogeneity in cross-sectional area of vessel. |

## 2) Junction Metrics (`junction_metrics_df`)

Each row corresponds to one node.

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `z`, `y`, `x` | Node voxel coordinates. | Spatial location of sprout tip or branch junction. |
| `number_of_vessel_per_node` | Graph node degree. | Local branching connectivity. |
| `node_type` | `sprout` for degree-1 endpoint, otherwise `junction`. | Terminal tip vs branching locus identity. |
| `dist_nearest_junction` | Dijkstra geodesic distance along skeleton edge polylines in um to nearest other junction. | Distance to nearest junction (as travelled along the vessel). |
| `dist_nearest_endpoint` | Dijkstra geodesic distance along skeleton edge polylines in um to nearest endpoint. | Distance to nearest sprout (as travelled along the vessel). |

## 3) Global Metrics (`global_metrics`)

### 3.1 Core global fields

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `total_vessel_length` | Sum of all cleaned edge arclengths. | Overall length of vasculature. |
| `total_number_of_sprouts` | Count of edges touching at least one endpoint node. | Overall length of sprout segments. |
| `total_number_of_junctions` | Count of non-endpoint nodes. | Overall branching complexity. |
| `fractal_dimension` | Box-counting slope computed on the skeleton mask. | Space-filling complexity of skeletonized vascular architecture. |
| `skeleton_lacunarity` | Mean box-mass heterogeneity statistic computed on the skeleton mask. | Spatial heterogeneity/patchiness of the skeletonized vascular network (topology-focused). |
| `vessel_lacunarity` | Mean box-mass heterogeneity statistic computed on the full vessel mask. | Spatial heterogeneity/patchiness of vessel occupancy including vessel thickness/caliber effects. |
| `lacunarity` | Backward-compatible alias of `skeleton_lacunarity`. | Same biological meaning as `skeleton_lacunarity`. |

### 3.2 Lean derived summary fields

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `median_sprout_and_branch_volume` | Median `volume` across all vessel edges. | Typical segment volume across whole network. |
| `median_sprout_and_branch_tortuosity` | Median `tortuosity` across all vessel edges. | Typical vessel winding of network. |
| `median_sprout_and_branch_median_cs_area` | Median `median_cs_area` across all vessel edges. | Typical vessel caliber across network. |
| `median_branch_tortuosity` | Median `tortuosity` restricted to branch edges (`is_sprout == False`). | Typical winding of internal branch segments. |
| `median_sprout_tortuosity` | Median `tortuosity` restricted to sprout edges (`is_sprout == True`). | Typical winding of terminal sprouts. |
| `median_junction_and_sprout_dist_nearest_junction` | Median `dist_nearest_junction` across all nodes. | Typical along-skeleton spacing to nearest branchpoint. |
| `median_junction_and_sprout_dist_nearest_endpoint` | Median `dist_nearest_endpoint` across all nodes. | Typical along-skeleton spacing to nearest endpoint. |
| `median_junction_dist_nearest_junction` | Median `dist_nearest_junction` restricted to junction nodes. | Typical junction-to-junction spacing. |
| `median_sprout_dist_nearest_endpoint` | Median `dist_nearest_endpoint` restricted to sprout nodes. | Typical sprout-to-nearest-tip spacing in terminal domain. |

## 4) Why Skeleton And Vessel Lacunarity Differ Biologically

- `skeleton_lacunarity` emphasizes network architecture (branching, spacing, and connectivity) while suppressing caliber/thickness effects.
- `vessel_lacunarity` reflects both architecture and caliber variation because thicker or thinner vessel regions change occupancy mass.
- If two segmentations have similar branching topology but different vessel widths, `skeleton_lacunarity` may stay similar while `vessel_lacunarity` diverges.
- If two segmentations have similar widths but different branching organization, both can change, but `skeleton_lacunarity` is usually more sensitive to topological rearrangement.

### 4.1 Mathematical difference

- `skeleton_lacunarity` is computed on a binary skeleton volume (mostly 1-voxel-wide centerlines), so box mass mainly reflects centerline density and arrangement.
- `vessel_lacunarity` is computed on the full vessel mask, so box mass reflects both arrangement and local vessel thickness/occupancy.
- Because lacunarity is based on variance-to-mean-squared of box masses, any factor that increases mass unevenness across boxes can increase it.
: for skeleton this is mostly topology and spacing; for vessel this is topology plus caliber/coverage.

### 4.2 Biological interpretation guide

- High `skeleton_lacunarity` with moderate `vessel_lacunarity`:
: likely irregular branching/spacing pattern without extreme caliber variability.
- High `vessel_lacunarity` with lower `skeleton_lacunarity`:
: likely uneven vessel thickness or regional occupancy differences with relatively similar centerline architecture.
- Both high:
: architecture and occupancy are both heterogeneous (irregular topology and uneven caliber/coverage).
- Both low:
: more homogeneous, evenly distributed network architecture and occupancy.

### 4.3 Why keep both

- Use `skeleton_lacunarity` when you want architecture-first comparisons that are less sensitive to segmentation thickness differences.
- Use `vessel_lacunarity` when total occupancy pattern and caliber heterogeneity are biologically meaningful outcomes.
- Reporting both helps disentangle "where vessels run" (topology) from "how much vessel mass is present" (occupancy/caliber).


## 5) Units and comparison notes

- Distances (`length`, nearest-distance metrics) are reported in um using voxel size (2.0, 2.0, 2.0).
- Cross-sectional areas are reported in um^2.
- Vessel volume is reported as an estimated um^3 quantity from area-times-length.
