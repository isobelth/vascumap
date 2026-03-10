# Vascular Metrics Reference

This document explains the outputs produced in `bel_skeletonisation.ipynb`:
- `global_metrics` (one-row summary)
- `vessel_metrics_df` (one row per edge/vessel segment)
- `junction_metrics_df` (one row per node)

It gives both:
- Mathematical meaning (how the value is computed)
- Biological meaning (how to interpret vessel biology)

## 1) Vessel Metrics (`vessel_metrics_df`)

Each row corresponds to one graph edge (between two nodes).

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `z`, `y`, `x` | Arrays of voxel coordinates along the edge polyline. | 3D trajectory of the vessel centerline segment. |
| `volume` | `sum(area_image along edge points)`; area-integral surrogate in voxel units. | Relative blood vessel volume represented by that segment. |
| `length` | Polyline arclength: sum of Euclidean distances between consecutive edge points. | Physical extent of vessel segment. Longer values imply longer vessel runs. |
| `shortest_path` | Euclidean distance between first and last edge point. | Straight-line endpoint separation; local span of the segment. |
| `tortuosity` | `length / (shortest_path + 1e-8)`, clipped to `[0, 5]`. | Vessel winding/curviness. Near 1 means straighter vessels; larger means more tortuous paths. |
| `is_sprout` | Boolean: true if either endpoint node has degree 1. | Flags terminal/end vessel pieces (sprouts) vs branch-to-branch segments. |
| `mean_cs_area` | Mean of cross-sectional area proxy values along edge voxels. | Typical vessel caliber along this segment. |
| `median_cs_area` | Median of cross-sectional area proxy along edge voxels. | Robust typical caliber less sensitive to local outliers. |
| `std_cs_area` | Standard deviation of cross-sectional area proxy along edge voxels. | Caliber variability along the segment (uniform vs irregular diameter). |
| `node1_degree`, `node2_degree` | Graph degree of each endpoint node. | Topological context: endpoint vs branchpoint connectivity at segment ends. |
| `orientation_z`, `orientation_y`, `orientation_x` | Components of normalized vector from first to last edge point. | Preferred segment direction in 3D tissue space. |

## 2) Junction Metrics (`junction_metrics_df`)

Each row corresponds to one graph node.

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `z`, `y`, `x` | Node voxel coordinates. | Spatial location of sprout tip or branch junction. |
| `number_of_vessel_per_node` | Node degree in graph. | Local branching complexity at that point (how many vessel segments meet). |
| `node_type` | Categorical label: `sprout` (degree-1 endpoint) or `junction` (non-endpoint). | Distinguishes terminal tips from branching points. |
| `dist_nearest_junction` | Dijkstra geodesic distance along the skeleton graph (edge polyline arclengths) to nearest other junction node. | How tightly packed branchpoints are along vessel centerlines (not straight-line distance). |
| `dist_nearest_endpoint` | Dijkstra geodesic distance along the skeleton graph (edge polyline arclengths) to nearest endpoint/sprout node. | Along-vessel proximity from a node to terminal tips. |
| `num_junction_neighbors` | Count of junction nodes whose along-skeleton geodesic distance is within threshold (`distance_threshold`, currently 250). | Density of nearby branchpoints measured along connected vessel paths. |
| `num_endpoint_neighbors` | Count of endpoint nodes whose along-skeleton geodesic distance is within threshold (`distance_threshold`, currently 250). | Density of nearby sprout tips measured along connected vessel paths. |

## 3) Global Metrics (`global_metrics`)

### 3.1 Core global fields

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `total_vessel_length` | Sum of all edge arclengths in cleaned graph. | Total vascular extent in the sample. |
| `total_number_of_sprouts` | Number of edges touching at least one sprout endpoint. | Overall terminal outgrowth prevalence. |
| `total_number_of_branches` | Number of edges whose endpoints are both non-sprout nodes. | Amount of internal branch-to-branch structure. |
| `total_number_of_junctions` | Count of non-sprout nodes. | Number of true branching loci. |
| `fractal_dimension` | Box-counting slope of occupied boxes for vessel mask. | Space-filling complexity of vessel pattern. |
| `lacunarity` | Mean box-mass heterogeneity statistic for vessel mask. | Texture heterogeneity; high values indicate patchy/uneven vascular coverage. |
| `gap_lacunarity` | Lacunarity on `internal_gap_mask = support_mask & (~vessel_mask)`. | Heterogeneity of avascular pockets within support region. |
| `support_fractal_dimension` | Box-counting FD of support mask. | Complexity of analysis support region geometry. |
| `support_lacunarity` | Lacunarity of support mask occupancy. | Heterogeneity of support region occupancy. |

Note for current no-hull/no-support setup:
- There is no separate support mask in code.
- `gap_lacunarity`, `support_fractal_dimension`, and `support_lacunarity` are currently set to `NaN` by design.

### 3.2 Aggregated fields produced by `add_grouped_stats`

The notebook automatically creates grouped summary fields from numeric columns.

Pattern:
- `mean_<group>_<metric>`
- `std_<group>_<metric>`
- `median_<group>_<metric>`

Groups used for vessel-level aggregation:
- `branch`
- `sprout`
- `sprout_and_branch` (combined set)

Metrics aggregated from vessel table:
- `volume`, `length`, `shortest_path`, `tortuosity`, `node1_degree`, `node2_degree`

Groups used for junction-level aggregation:
- `junction`
- `sprout`
- `junction_and_sprout` (combined set)

Metrics aggregated from junction table:
- `number_of_vessel_per_node`, `dist_nearest_junction`, `dist_nearest_endpoint`, `num_junction_neighbors`, `num_endpoint_neighbors`

## 4) The 3 specifically confusing parameters

### `mean_sprout_and_branch_volume`

- Mathematical: arithmetic mean of `vessel_metrics_df['volume']` across all vessel edges (sprouts + branches together).
- Biological: average vessel segment volume in the whole network, combining terminal sprouts and internal branches.

### `median_sprout_num_junction_neighbors`

- Mathematical: median of `junction_metrics_df['num_junction_neighbors']` restricted to rows where `node_type == 'sprout'`; neighbors are counted by along-skeleton geodesic distance threshold.
- Biological: typical number of branchpoints reachable from sprout tips within a fixed along-vessel travel distance.

### `mean_junction_and_sprout_num_endpoint_neighbors`

- Mathematical: mean of `junction_metrics_df['num_endpoint_neighbors']` across all nodes (junctions + sprouts); counts are based on along-skeleton geodesic distance threshold.
- Biological: average local endpoint-tip neighborhood density measured along connected vessel paths across the whole network.

## 5) Units and practical interpretation

- Most distance-like terms (`length`, `shortest_path`, nearest-distance fields) are in voxel units unless you rescale by voxel size.
- Area/volume proxies are in voxel-derived units from EDT-based approximations.
- Neighbor counts are unitless integers.
- For cross-sample comparison, keep voxel size and preprocessing constants fixed.
