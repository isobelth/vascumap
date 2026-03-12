# Vascular Metrics Reference

This document reflects the current outputs in `bel_skeletonisation.ipynb`.

Outputs:
- `global_metrics_df`: one-row sample-level summary for comparison between samples/chips
- `vessel_metrics_df`: one row per cleaned vessel edge
- `junction_metrics_df`: one row per cleaned graph node

The current normalization strategy is designed for comparing vasculature grown on chips with different image sizes.

## Normalization Basis

- Chip volume is defined as the full volume of the `vasculature_segmentation` image.
- The voxel size is assumed to be `2 x 2 x 2 um`.
- Therefore, one voxel corresponds to `8 um^3`.
- In the notebook, `chip_volume_voxels = np.prod(vasculature_segmentation.shape)`.
- Physical chip volume can therefore be computed as:

	`chip_volume_um3 = chip_volume_voxels * 8`

- The current notebook stores the denominator in voxel units, so all `*_per_chip_volume` outputs are currently reported per voxel-volume of the analysed image.
- Because every chip uses the same voxel size, those density metrics remain directly comparable between chips.

## 1) Vessel Metrics (`vessel_metrics_df`)

Each row corresponds to one cleaned graph edge.

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `z`, `y`, `x` | Arrays of voxel coordinates along the edge polyline. | 3D vessel centerline trajectory. |
| `volume` | Sum of local cross-sectional area estimates sampled along the edge. | Relative vessel segment volume proxy. |
| `length` | Polyline arclength measured from centerline voxels. | Vessel segment extent. |
| `shortest_path` | Straight-line Euclidean distance between edge endpoints. | End-to-end displacement of the segment. |
| `tortuosity` | `length / shortest_path`, clipped to `[0, 5]`. | Curviness or winding of the segment. Near 1 is straighter. |
| `is_sprout` | `True` if either endpoint node has degree 1. | Distinguishes terminal sprouts from internal branches. |
| `mean_cs_area` | Mean cross-sectional area estimate sampled along edge. | Average vessel caliber along segment. |
| `median_cs_area` | Median cross-sectional area estimate sampled along edge. | Typical vessel caliber along segment. |
| `std_cs_area` | Standard deviation of cross-sectional area estimate along edge. | Caliber heterogeneity along segment. |
| `node1_degree`, `node2_degree` | Degree of each endpoint node in graph. | Local connectivity at the segment ends. |
| `orientation_z`, `orientation_y`, `orientation_x` | Normalized direction vector from first to last edge point. | Gross vessel orientation in 3D. |

## 2) Junction Metrics (`junction_metrics_df`)

Each row corresponds to one node in the cleaned graph.

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `z`, `y`, `x` | Node voxel coordinates. | Spatial location of sprout tip or branch junction. |
| `number_of_vessel_per_node` | Graph node degree. | Local branching connectivity. |
| `node_type` | `sprout` for degree-1 endpoint, otherwise `junction`. | Terminal tip versus branching locus. |
| `dist_nearest_junction` | Euclidean distance in voxel coordinates to nearest junction node. | Local spacing to nearest branchpoint. |
| `dist_nearest_endpoint` | Euclidean distance in voxel coordinates to nearest sprout node. | Local spacing to nearest endpoint. |
| `num_junction_neighbors` | Number of junction nodes within the specified distance threshold. | Local branchpoint crowding. |
| `num_endpoint_neighbors` | Number of endpoint nodes within the specified distance threshold. | Local sprout crowding. |

## 3) Global Metrics (`global_metrics_df`)

### 3.1 Raw sample-level totals

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `chip_volume_voxels` | `np.prod(vasculature_segmentation.shape)`. | Total analysed chip image volume. |
| `vessel_volume_voxels` | Number of nonzero voxels in `binary_filled_holes`. | Total vessel occupancy in the analysed image. |
| `vessel_volume_fraction` | `vessel_volume_voxels / chip_volume_voxels`. | Fraction of analysed chip volume occupied by vasculature. |
| `total_vessel_length` | Sum of all cleaned edge lengths. | Overall network extent. |
| `total_number_of_sprouts` | Count of sprout edges. | Total number of terminal vessel segments. |
| `total_number_of_branches` | Count of non-sprout edges. | Total number of internal branch segments. |
| `total_number_of_junctions` | Count of non-endpoint nodes. | Overall branching complexity. |
| `fractal_dimension` | Box-counting scaling exponent. | Space-filling complexity of vascular architecture. |
| `lacunarity` | Box-mass heterogeneity statistic. | Spatial heterogeneity or gappiness of the vascular pattern. |

### 3.2 Size-normalized comparison metrics

These are the most useful outputs for comparing chips with different analysed image sizes.

| Parameter | Mathematical significance | Biological significance |
|---|---|---|
| `vessel_length_per_chip_volume` | `total_vessel_length / chip_volume_voxels`. | Network length density within the analysed chip volume. |
| `sprouts_per_chip_volume` | `total_number_of_sprouts / chip_volume_voxels`. | Sprout density per analysed chip volume. |
| `branches_per_chip_volume` | `total_number_of_branches / chip_volume_voxels`. | Branch density per analysed chip volume. |
| `junctions_per_chip_volume` | `total_number_of_junctions / chip_volume_voxels`. | Junction density per analysed chip volume. |
| `sprouts_per_vessel_length` | `total_number_of_sprouts / total_vessel_length`. | Frequency of terminal sprouts per unit network extent. |
| `branches_per_vessel_length` | `total_number_of_branches / total_vessel_length`. | Frequency of branch segments per unit network extent. |
| `junctions_per_vessel_length` | `total_number_of_junctions / total_vessel_length`. | Frequency of junctions per unit network extent. |

Interpretation:
- Use `*_per_chip_volume` when you want density relative to the full analysed chip image.
- Use `*_per_vessel_length` when you want topology normalized by how much vasculature is present.
- Use `vessel_volume_fraction` when you want occupancy normalized by chip size.

### 3.3 Grouped summary fields

The notebook also appends grouped summary statistics from `vessel_metrics_df` and `junction_metrics_df`.

For vessel-edge metrics, fields are generated in the form:
- `mean_sprout_and_branch_<metric>`
- `std_sprout_and_branch_<metric>`
- `median_sprout_and_branch_<metric>`
- `mean_branch_<metric>`, `std_branch_<metric>`, `median_branch_<metric>`
- `mean_sprout_<metric>`, `std_sprout_<metric>`, `median_sprout_<metric>`

For node-level metrics, fields are generated in the form:
- `mean_junction_and_sprout_<metric>`
- `std_junction_and_sprout_<metric>`
- `median_junction_and_sprout_<metric>`
- `mean_junction_<metric>`, `std_junction_<metric>`, `median_junction_<metric>`
- `mean_sprout_<metric>`, `std_sprout_<metric>`, `median_sprout_<metric>`

These grouped fields are still useful, but for cross-chip comparison the most robust first-pass readouts are usually:
- `vessel_volume_fraction`
- `vessel_length_per_chip_volume`
- `sprouts_per_chip_volume`
- `branches_per_chip_volume`
- `junctions_per_chip_volume`
- `sprouts_per_vessel_length`
- `branches_per_vessel_length`
- `junctions_per_vessel_length`

## 4) Units and Comparison Notes

- The segmentation image voxel size is `2 x 2 x 2 um`.
- One voxel corresponds to `8 um^3`.
- `chip_volume_voxels` and `vessel_volume_voxels` are stored in voxel counts.
- To convert either to physical volume, multiply by `8`.
- Length-based quantities are currently computed from voxel coordinates in the notebook, so they are reported in voxel-length units unless the code is further scaled by voxel size.
- This means the current volume-normalized metrics are correct for comparison between chips acquired at the same voxel size, but not yet expressed in absolute physical units.

If absolute physical units are required everywhere, the next step would be to scale all length-based outputs by voxel size and all volume-normalized outputs by `8 um^3` per voxel.
