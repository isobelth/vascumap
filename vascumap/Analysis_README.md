# VascuMap Analysis Outputs — Reference and Biological Interpretation

VascuMap outputs three CSV files for each image processed:

| File | Row granularity | Number of columns | Intended use |
|---|---|---|---|
| `{name_prefix}_analysis_metrics.csv` | One row per image | ID columns + **23 curated features** | The recommended panel for use in the lab. This CSV contains a subset of biologically interpretable, shape-invariant descriptors. |
| `{name_prefix}_all_morphological_params.csv` | One row per image | ID columns + **all** computed morphological metrics (~180) | Contains every global metric the pipeline computes, including disaggregated mean / std / median / p90\u2212p10 statistics split by branch vs sprout vs combined, and per-junction-type connectivity stats. Useful to have on file for future analyses, or useful for future GNN embedding. |
| `{name_prefix}_branch_metrics.csv` | One row per skeleton **edge** (vessel segment) | ~25 columns including `node_start`/`node_end` integer IDs | The **graph table**: each row is one branch/sprout in the cleaned vessel graph, with its endpoint coordinates, length, calibre, tortuosity and orientation. Suitable as edge features for a future **graph neural network embedding**, or for plotting per-branch distributions. Probably not useful for individual lab users! |

The first three columns of every CSV are identical and identify the image:
`image_name`, `source_file`, `image_index`.

The curated `analysis_metrics.csv` is just a subset of
`all_morphological_params.csv`, curated for easier plotting/readability
in the lab.

---

## Scope and notation

All values are derived from the **final cleaned segmentation, skeleton, and
graph** (after smoothing, hole-filling, pruning, mid-node removal, border /
exclusion zone trimming, and isolated-node removal). The default voxel size
is `(2, 2, 2) µm`.

Symbols used throughout:

- $V_{chip}$: chip / imaged volume in $\mu m^3$. When an organoid is imaged,
- its masked volume is **subtracted**
  from $V_{chip}$ so it does not count as available space.
- $V_{hull}$: convex-hull volume of the segmented vasculature in $\mu m^3$.
  When an organoid is masked, the volume of the organoid in the convex hull
  is subtracted from the hull volume. In symbols:
  $V_{hull} = V_{hull,\,raw} - V_{organoid \cap hull}$.
- $V_{vessel}$: total vessel volume (number of vessel-positive voxels times
  voxel volume) in $\mu m^3$. The organoid region is set to zero in the
  segmentation before this is counted, so excluded voxels never contribute.
- $L_{total}$: total centerline length of segmented vasculature, summed over all graph
  edges, in $\mu m$.
- $N_{junction}$: number of non-sprout graph nodes (i.e. branch points).
- $N_{sprout}$: number of graph edges incident to at least one sprout (degree-1)
  node.
- "$P90 - P10$" denotes the spread between the 90th and 10th percentiles of a
  distribution. Selected as a robust, outlier-tolerant measure of the spread of values.
- "Sprout" = a degree-1 endpoint (a tip). "Branch" = a non-tip edge connecting
  two junctions. "Sprout-and-branch" = the union (every edge in the graph).

---

## Curated Analysis Metrics

There are 23 features in `*_analysis_metrics.csv`. They were chosen to be:

1. **Shape-invariant.** None of these features scales with the chip size.
   They are either dimensionless ratios (e.g. vessel volume fraction),
   per-unit-length densities (e.g. sprouts per micron of vessel),
   per-unit-volume densities normalised by the convex hull (e.g. branches
   per hull volume), or intrinsic per-vessel/per-junction quantities (e.g.
   typical branch length in microns). This means two images of the same
   biology cropped to different sizes should give very similar values to
   prevent PCA from just trivially separating different chip sizes.
2. **Biologically interpretable.** Features can be thought of in terms of
   the underlying biology.
3. **Manageable dimensionality.** Twenty-three features is hopefully
   sufficient to allow PCA clustering at smaller sample sizes while still
   carrying separable density, geometry, topology, connectivity, and
   orientation channels.

All volume densities are normalised to **$V_{hull}$ with the organoid
region subtracted** (see *Convex hull and exclusion regions* below); this
is the biologically available gel space inside the vascular envelope.

### The 23 curated features

#### Density (7 features)

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `vessel_volume_fraction` | $V_{vessel}/V_{hull}$ | The fraction of the convex hull around the vasculature that is actually filled with vessel tissue. A global readout of how densely vascularised the sample is, regardless of how big the imaged region was. **Excluded organoid volume is subtracted from the denominator** so the fraction reflects only the gel space available for vessels. | Ratio of two volumes, both intrinsic to the sample. |
| `branch_length_per_hull_volume_um_inverse2` | $L_{branch}/V_{hull}$ | Total amount of **non-sprout** vessel "wire" per unit envelope volume. Sprouts are excluded from the numerator because their length depends on tip-pruning depth and is reported separately under sprout-only branch geometry. Higher means more or finer connecting vasculature. | Length divided by volume → intensive (does not scale with FOV). |
| `sprouts_per_vessel_length_um_inverse` | $N_{sprout}/L_{total}$ | How frequently you encounter a sprout (a tip) per micron of vessel. A direct readout of **angiogenic sprouting intensity** normalised to network size. | Count per unit length. |
| `junctions_per_vessel_length_um_inverse` | $N_{junction}/L_{total}$ | How frequently you encounter a branch point per micron of vessel. A direct readout of **branching intensity** independent of how much vessel you imaged. | Count per unit length. |
| `sprouts_per_hull_volume_um_inverse3` | $N_{sprout}/V_{hull}$ | Sprout count per unit envelope volume. Together with `sprouts_per_vessel_length_um_inverse`, separates "a few sprouts on a sparse network" from "many sprouts on a dense network". | Count per volume. |
| `junctions_per_hull_volume_um_inverse3` | $N_{junction}/V_{hull}$ | Junction count per unit envelope volume — the spatial density of branch points within the gel space. | Count per volume. |
| `branches_per_hull_volume_um_inverse3` | $N_{branch}/V_{hull}$ | Non-sprout edge count per unit envelope volume. Together with `junctions_per_hull_volume_um_inverse3`, parameterises mesh fineness in two complementary ways (count of vertices vs count of edges). | Count per volume. |

The per-vessel-length and per-hull-volume densities are kept side-by-side
because they encode genuinely different information: the per-length
variants describe **architecture** (how many sprouts/junctions per unit of
vessel you have built), while the per-volume variants describe **spatial
occupancy** (how many sprouts/junctions are packed into the gel space).
Two samples can match on one and differ on the other.

#### Topology (2 features)

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `skeleton_fractal_dimension` | Box-counting slope $D$ of the cleaned skeleton mask (see *Mathematical caveats* below) | A scale-free complexity index of the centerline network. Higher values ($D \to 2$) mean the network fills space more densely with branches; lower values ($D \to 1$) mean it looks more like a sparse tree. | Defined as a scaling exponent, intrinsically scale-free. |
| `skeleton_lacunarity` | Variance-to-mean statistic of box-mass distribution on the skeleton | A measure of "patchiness" or unevenness in how branches are distributed in space. Low values mean the network is evenly spread; high values mean there are dense clumps separated by empty regions. Two networks can share fractal dimension but differ strongly in lacunarity. | Constructed from a normalised mass distribution. |

#### Branch geometry — branches only (4 features)

These describe the geometry of **non-sprout** edges (segments between two
junction nodes). Sprouts are split into a separate subsection below
because their length and calibre depend on where the skeletonisation
pipeline chose to terminate the tip (see *Tortuosity caveats*), so it is
cleaner to interpret them as their own family.

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `median_branch_length_um` | Median of per-edge centerline lengths $L_{path}$ across non-sprout edges ($\mu m$) | The **typical length of a connecting vessel segment** between two branch points. Larger values = longer, less subdivided vessels; smaller values = a finely subdivided, mesh-like network. | An intrinsic length of one structural unit. |
| `p90_minus_p10_branch_length_um` | $P90(L_{path}) - P10(L_{path})$ across non-sprout edges ($\mu m$) | How heterogeneous the connecting-segment lengths are. A small spread means a uniform mesh; a large spread means the sample has a mixture of short capillary segments and long arterial-like runs. | Difference of two percentiles of an intrinsic length. |
| `median_branch_median_cs_area_um2` | Median across non-sprout edges of each edge's median sampled cross-sectional area ($\mu m^2$) | The **typical vessel calibre** (cross-section area) — a thickness proxy — for established (non-tip) vasculature. | Per-vessel measurement, independent of how many vessels are imaged. |
| `p90_minus_p10_branch_median_cs_area_um2` | $P90 - P10$ of the per-edge median cross-section area across non-sprout edges ($\mu m^2$) | Heterogeneity of vessel calibre across the connecting network. Large spread = mixed vessel sizes; small spread = uniformly sized vessels. | Spread of an intrinsic per-vessel quantity. |

#### Branch geometry — sprouts only (4 features)

The same four geometric descriptors restricted to sprout edges (those
incident to a degree-1 tip). These are reported separately because their
absolute values depend on the skeletonisation pruning depth — interpret
them in a *relative* sense (comparing conditions processed with the same
pipeline) rather than as absolute anatomical sizes.

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `median_sprout_length_um` | Median of $L_{path}$ across sprout edges ($\mu m$) | The **typical length a sprout extends** before terminating. Higher values in matched-pipeline comparisons indicate longer-reaching tip cells. | Intrinsic length of one anatomical unit. |
| `p90_minus_p10_sprout_length_um` | $P90(L_{path}) - P10(L_{path})$ across sprout edges ($\mu m$) | Heterogeneity of sprout lengths — does the sample contain a mixture of short and long sprouts, or are all sprouts of similar length? | Difference of two percentiles of an intrinsic length. |
| `median_sprout_median_cs_area_um2` | Median across sprout edges of each edge's median sampled cross-sectional area ($\mu m^2$) | **Typical sprout calibre** — sprouts are usually thinner than connecting vessels, and trends here track tip-cell maturation. | Per-sprout measurement. |
| `p90_minus_p10_sprout_median_cs_area_um2` | $P90 - P10$ of the per-edge median cross-section area across sprout edges ($\mu m^2$) | Heterogeneity of sprout calibre. | Spread of an intrinsic per-sprout quantity. |

#### Tortuosity — branch-only (2 features)

Sprouts (degree-1 tips) are excluded from these statistics. Both pieces of
tortuosity — the centerline length $L_{path}$ and the chord
$L_{endpoints}$ — do depend on where the skeletonisation algorithm chose
to terminate the tip; they are *not* independent of the algorithm. The
problem with the *ratio* $\tau = L_{path}/L_{endpoints}$ is that it is
dramatically more sensitive to that choice than either piece alone:

- For a near-straight sprout, $L_{path} \approx L_{endpoints}$, so $\tau
  \approx 1$ no matter where the tip is cut. Fine.
- For a curved sprout, the chord $L_{endpoints}$ is a 3-D vector between
  the junction and the tip. Shifting the tip by a few voxels along a
  curving centerline can swing that chord across an inflection, changing
  its *direction* and length disproportionately to the small change in
  $L_{path}$. Geometrically the chord has $90^{\circ}$-ish failure modes
  that the arc-length does not. The ratio therefore inherits that
  instability while the path length itself only changes by the few voxels
  that were added or removed.

In short, $L_{path}$ is a **measurement** of the sprout (small pruning
changes → small measurement changes), whereas $\tau$ for a sprout is a
**ratio whose sensitivity to pruning is unbounded**. We therefore report
sprout length as a biological readout (next subsection) but restrict
tortuosity statistics to fully-formed connecting branches whose two
endpoints are both real junctions.

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `median_branch_tortuosity` | Median of $\tau = L_{path}/L_{endpoints}$ across non-sprout edges; $\tau$ clipped to $[1, 50]$ | The **typical curviness of a vessel**. $\tau = 1$ means perfectly straight; $\tau > 1$ means winding. | Ratio of two lengths; dimensionless. |
| `p90_minus_p10_branch_tortuosity` | $P90(\tau) - P10(\tau)$ across non-sprout edges | How heterogeneous the curviness is across branches — a small spread means uniformly straight (or uniformly winding) vessels, a large spread means a mix of straight and tortuous vessels coexist. We use a percentile spread rather than the standard deviation here so that the small number of clipped-at-50 outliers (see *Tortuosity clipping*, below) cannot dominate the statistic. | Difference of two percentiles of a dimensionless quantity. |

#### Junction connectivity (2 features)

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `median_junction_degree` | Median graph degree of non-sprout nodes (number of edges meeting at a typical branch point) | The **typical branching factor** at a junction. A value near 3 means most branch points are simple Y-junctions (the canonical vascular branching pattern). Higher values indicate more complex multi-way meeting points. | Pure graph-theoretic count at one node; FOV-independent. |
| `p90_minus_p10_junction_degree` | $P90 - P10$ of junction degree | How variable the branching pattern is across the network. A homogeneous capillary bed will have very small spread; a network with frequent multi-way "hubs" will have larger spread. Using $P90 - P10$ rather than std is consistent with the other spread metrics and avoids inflation by rare very-high-degree nodes. | Spread of a count; dimensionless. |

#### Orientation (2 features)

Orientation is measured relative to the device long axis. The pipeline
infers the long axis automatically from the image footprint: if the image
is wider than it is tall ($x$-extent $\ge$ $y$-extent) it uses the $x$
axis; otherwise it uses the $y$ axis. This means rotated acquisitions are
handled correctly without any manual configuration. The angle reported is
the acute angle in the $xy$ plane between each branch's end-to-end
direction and that auto-detected long axis, expressed in degrees in
$[0, 90]$.

| Column | Math | Biological meaning | Why it is shape-invariant |
|---|---|---|---|
| `median_sprout_and_branch_orientation_deg` | Median branch orientation in degrees | The **dominant alignment** of vessels with respect to the device. A value near $45°$ means no preferred direction (isotropic); values near $0°$ or $90°$ indicate strong alignment with or perpendicular to the device axis (e.g. flow-induced alignment in a perfused chamber). | Per-branch angle; geometry independent of FOV size. |
| `p90_minus_p10_sprout_and_branch_orientation_deg` | $P90 - P10$ of branch orientation in degrees | Network **anisotropy**: how concentrated the orientations are around their median. A small spread means almost all vessels point in the same direction; a large spread (towards $80°$) means orientations are nearly uniform. | Spread of a dimensionless angle. |

Nearest-neighbour spacing metrics (`*_dist_nearest_*`) are not part of the
curated panel — they remain available in the full
`*_all_morphological_params.csv` (both along-skeleton and Euclidean
variants).

### Excluded from the curated panel — and why

The following columns appear in the full
`*_all_morphological_params.csv` file but are deliberately omitted from the
curated panel:

- `chip_volume_um3`, `convex_hull_volume_um3`, `vessel_volume_um3`,
  `total_vessel_length_um`, `total_branch_length_um` — **raw size/length
  quantities**. These scale directly with the cropped field of view and would
  dominate any unsupervised analysis with a "big-vs-small image" axis instead
  of biology.
- `total_number_of_sprouts`, `total_number_of_branches`,
  `total_number_of_junctions`, `total_number_of_edges`,
  `total_number_of_nodes`, `total_number_of_floating_sprouts` — **raw
  counts**. Same problem as raw size: their density-normalised equivalents
  (per length or per volume) are kept instead.
- `floating_sprouts_per_hull_volume_um_inverse3` — kept in the full file
  (see *Floating sprouts* under *Mathematical caveats* below) but excluded
  from the curated panel because it can be dominated by segmentation noise
  rather than biology.
- `median_internal_pore_area_um2`,
  `p90_minus_p10_internal_pore_area_um2`,
  `median_internal_pore_max_inscribed_radius_um`, … — **pore features** are
  largely captured by the distribution of vessel cross-sectional area in
  combination with branch spacing, and have noisier estimates because they
  are computed slice-by-slice. Excluded by design; they remain available in
  the full audit file if needed.
- All `mean_*` and `std_*` per-aggregate variants of features already
  represented by their `median_*` and `p90_minus_p10_*` siblings, in the
  interest of robustness to outliers.

---

## Per-branch metrics — the GNN-ready table

`*_branch_metrics.csv` has **one row per edge** in the cleaned vessel graph.
Together with the integer node IDs (`node_start`, `node_end`) it forms a
ready-made edge list for a graph neural network: each row carries the
geometric features of one vessel segment, and the IDs let you reconstruct
the graph topology.

| Column | Units | Meaning |
|---|---|---|
| `node_start`, `node_end` | int | Integer IDs of the two graph nodes this edge connects. Use these to build the GNN edge index. |
| `is_sprout` | bool | True if either endpoint is a degree-1 sprout (i.e. this edge is a sprout/tip), False if it is a connecting branch between two junctions. |
| `start_z`, `start_y`, `start_x`, `end_z`, `end_y`, `end_x` | voxels | Endpoint coordinates in voxel index space. |
| `start_z_um`, `start_y_um`, `start_x_um`, `end_z_um`, `end_y_um`, `end_x_um` | $\mu m$ | The same endpoints in physical units (voxel index times voxel size). |
| `path_length_um` | $\mu m$ | The **curved length** of the vessel segment, summed along its centerline polyline. This is the biologically relevant distance "along the pipe". |
| `endpoint_distance_um` | $\mu m$ | The **straight-line distance** between the two endpoints. Compare with `path_length_um` to assess curvature. |
| `tortuosity` | unitless | $\tau = L_{path}/L_{endpoints}$, clipped to $[1, 50]$. A direct curviness score: $\tau = 1$ is perfectly straight, $\tau \gg 1$ is very winding. |
| `mean_cs_area_um2`, `median_cs_area_um2`, `std_cs_area_um2` | $\mu m^2$ | Cross-sectional area sampled along the segment from the local distance transform; mean / median / standard deviation of those samples. The median is more robust to local segmentation noise. |
| `mean_width_um`, `median_width_um` | $\mu m$ | Equivalent circular widths $w = \sqrt{4 A / \pi}$ derived from the corresponding cross-section area. Interpret as the diameter of a circle with the same area as the local vessel cross-section. |
| `branch_volume_um3` | $\mu m^3$ | Approximate volume of the vessel segment, computed as `mean_cs_area_um2 × path_length_um`. |
| `orientation_to_device_axis_deg` | degrees in $[0, 90]$ | Acute angle between the segment's endpoint-to-endpoint vector (in the $xy$ plane) and the device long axis. |

---

## All Morphological Parameters — full audit reference

`*_all_morphological_params.csv` contains a superset of every metric
the pipeline computes. It includes:

1. **All of the curated 19 features above** (so you never need to re-run the
   pipeline to switch between curated and full views).
2. **All of the per-image global metrics** that the pipeline computes
   internally — see the table below.
3. **Disaggregated branch statistics**: for each of the per-branch metrics
   `volume_um3`, `length_um`, `endpoint_distance_um`, `tortuosity`,
   `mean_cs_area_um2`, `median_cs_area_um2`, `std_cs_area_um2`,
   `mean_width_um`, `median_width_um`, `orientation_deg`, the file contains
   `mean_*`, `std_*`, `median_*`, and `p90_minus_p10_*` aggregated over
   (a) **branch-only** edges (`*_branch_*`), (b) **sprout-only** edges
   (`*_sprout_*`), and (c) **all edges combined**
   (`*_sprout_and_branch_*`). For example: `mean_branch_length_um`,
   `std_sprout_tortuosity`, `median_sprout_and_branch_mean_width_um`,
   `p90_minus_p10_branch_tortuosity`, etc. The `p90_minus_p10_*` family is
   an outlier-robust measure of spread (the difference between the 90th and
   10th percentile) and is preferred over `std_*` whenever the underlying
   distribution has heavy tails or hard-clipped values.
4. **Disaggregated junction statistics**: for each of the per-junction metrics
   `degree`, `dist_nearest_junction_um`, `dist_nearest_endpoint_um`,
   `num_junction_neighbors`, `num_endpoint_neighbors`, the file contains
   `mean_*`, `std_*`, `median_*`, and `p90_minus_p10_*` aggregated over
   (a) **junction nodes** (`*_junction_*`), (b) **sprout-tip nodes**
   (`*_sprout_tip_*`), and (c) **all nodes combined** (`*_all_nodes_*`).
   For example: `mean_junction_degree`,
   `std_sprout_tip_dist_nearest_endpoint_um`,
   `p90_minus_p10_all_nodes_degree`.

### Headline global-metric dictionary

The columns below are emitted directly by the pipeline (they are the
"global" entries on top of which the disaggregated stats are layered):

| Column | Units | Mathematical meaning | Biological interpretation |
|---|---:|---|---|
| `chip_volume_um3` | $\mu m^3$ | $V_{chip}$, the imaged chip volume (minus any excluded organoid region). | Physical assay/imaged volume used internally for normalisation. **Excluded from curated panel** because it depends on FOV. |
| `convex_hull_volume_um3` | $\mu m^3$ | $V_{hull}$, volume of the 3D convex hull of vessel-positive voxels. | The "envelope" the vasculature occupies. **Excluded from curated panel** for the same reason. |
| `vessel_volume_um3` | $\mu m^3$ | $V_{vessel}$, total vessel-positive volume. | Total vascular biomass. **Excluded from curated panel.** |
| `vessel_volume_fraction` | unitless | $V_{vessel}/V_{hull}$ | **Curated.** Fraction of hull occupied by vessels. |
| `total_vessel_length_um` | $\mu m$ | $L_{total}$ from summed edge polyline lengths (all edges, sprouts + branches). | Total vascular extent by centerline length. **Excluded from curated panel.** |
| `total_branch_length_um` | $\mu m$ | $L_{branch}$, sum of polyline lengths over **non-sprout** edges only. | Total length of established (non-tip) vasculature. **Excluded from curated panel** (size-dependent); used internally as the numerator of `branch_length_per_hull_volume_um_inverse2`. |
| `branch_length_per_hull_volume_um_inverse2` | $\mu m^{-2}$ | $L_{branch}/V_{hull}$ | **Curated.** 3D length density of non-sprout vasculature. |
| `sprouts_per_vessel_length_um_inverse` | $\mu m^{-1}$ | $N_{sprout}/L_{total}$ | **Curated.** Sprouting intensity per unit vessel length. |
| `junctions_per_vessel_length_um_inverse` | $\mu m^{-1}$ | $N_{junction}/L_{total}$ | **Curated.** Branching intensity per unit vessel length. |
| `sprouts_per_hull_volume_um_inverse3` | $\mu m^{-3}$ | $N_{sprout}/V_{hull}$ | **Curated.** Sprout density per unit envelope volume. |
| `junctions_per_hull_volume_um_inverse3` | $\mu m^{-3}$ | $N_{junction}/V_{hull}$ | **Curated.** Junction density per unit envelope volume. |
| `branches_per_hull_volume_um_inverse3` | $\mu m^{-3}$ | $N_{branch}/V_{hull}$ | **Curated.** Non-sprout edge density per unit envelope volume. |
| `floating_sprouts_per_hull_volume_um_inverse3` | $\mu m^{-3}$ | (Number of all-sprout connected components)$/V_{hull}$ | Density of fully-detached sprout fragments — components of the cleaned graph all of whose nodes are degree-1 tips (i.e. vessel pieces with no junction). May reflect either real biological detachments or segmentation noise. **Excluded from curated panel.** See *Floating sprouts* below. |
| `skeleton_fractal_dimension` | unitless | Box-counting slope of the cleaned graph-derived skeleton mask. | **Curated.** Geometric complexity of the centerline network. |
| `skeleton_lacunarity` | unitless | Gap/heterogeneity statistic on the same skeleton mask. | **Curated.** Spatial patchiness / unevenness of the centerline network. |
| `median_sprout_and_branch_orientation_deg` | degrees | Median per-edge orientation to the auto-detected device long axis. | **Curated.** Dominant vessel alignment. |
| `p90_minus_p10_sprout_and_branch_orientation_deg` | degrees | Spread $P90 - P10$ of per-edge orientation. | **Curated.** Anisotropy of vessel alignment. |
| `median_sprout_and_branch_tortuosity` | unitless | Median per-edge tortuosity over all edges. | Typical winding across the whole network. *(The curated panel keeps the branch-only variant `median_branch_tortuosity` instead.)* |
| `p90_minus_p10_sprout_and_branch_tortuosity` | unitless | Spread $P90 - P10$ of per-edge tortuosity. | Heterogeneity of winding across the whole network. |
| `median_sprout_and_branch_median_cs_area_um2` | $\mu m^2$ | Median of per-edge median cross-sectional area. | **Curated.** Typical vessel calibre. |
| `p90_minus_p10_sprout_and_branch_median_cs_area_um2` | $\mu m^2$ | Spread of per-edge median cross-sectional area. | **Curated.** Heterogeneity of vessel calibre. |
| `median_sprout_and_branch_length_um` | $\mu m$ | Median per-edge centerline length. | **Curated.** Typical segment length. |
| `p90_minus_p10_sprout_and_branch_length_um` | $\mu m$ | Spread of per-edge centerline length. | **Curated.** Heterogeneity of segment lengths. |
| `median_junction_skeleton_dist_nearest_junction_um` | $\mu m$ | Median over branch points of the shortest along-skeleton distance to the nearest other branch point (whole-network headline metric). | Characteristic mesh size. *(Per-junction Euclidean equivalents are emitted by the disaggregated stats as `median_junction_dist_nearest_junction_um` etc.)* |
| `p90_minus_p10_junction_skeleton_dist_nearest_junction_um` | $\mu m$ | Spread of along-skeleton nearest-junction distances. | Heterogeneity of branch-point spacing. |
| `median_sprout_tip_skeleton_dist_nearest_endpoint_um` | $\mu m$ | Median along-skeleton nearest-endpoint distance among sprout tips. | Typical tip-to-tip spacing. |
| `p90_minus_p10_sprout_tip_skeleton_dist_nearest_endpoint_um` | $\mu m$ | Spread of along-skeleton nearest-endpoint distances. | Heterogeneity of tip-to-tip spacing. |
| `average_vessel_volume_um3` | $\mu m^3$ | Mean of `branch_volume_um3` over all edges. | Typical per-segment vessel volume. **Excluded from curated panel** because it is largely a product of typical length and typical calibre, both already kept. |
| `median_internal_pore_area_um2` | $\mu m^2$ | Median valid pore area across detected slice-wise pores. | Typical pore size. **Excluded from curated panel** (see exclusions section). |
| `p90_minus_p10_internal_pore_area_um2` | $\mu m^2$ | Spread $P90 - P10$ of pore area. | Heterogeneity of pore size. **Excluded from curated panel.** |
| `total_number_of_sprouts` | count | $N_{sprout}$. | Raw sprout count. **Excluded from curated panel** (FOV-dependent). |
| `total_number_of_branches` | count | Number of non-sprout edges. | Raw branch count. **Excluded.** |
| `total_number_of_junctions` | count | $N_{junction}$. | Raw junction count. **Excluded.** |
| `total_number_of_edges` | count | Total number of edges in the cleaned graph. | Raw edge count. **Excluded.** |
| `total_number_of_nodes` | count | Total number of nodes in the cleaned graph. | Raw node count. **Excluded.** |
| `total_number_of_floating_sprouts` | count | Number of connected components of the cleaned graph all of whose nodes are degree-1 sprouts (i.e. small vessel fragments with no junction). | Raw count of detached sprout fragments. **Excluded** (FOV- and noise-dependent); see *Floating sprouts* below. |

The disaggregated `mean_*` / `std_*` / `median_*` / `p90_minus_p10_*` ×
`branch` / `sprout` / `sprout_and_branch` × geometry-column families (and
the corresponding junction-side families) follow a consistent naming
pattern, so any column not listed above can be decoded by reading its name
left-to-right. The `<aggregate>` token is one of `mean`, `std`, `median`,
or `p90_minus_p10` (the difference between the 90th and 10th percentile,
i.e. an outlier-robust spread measure).

> `<aggregate>_<edge subset>_<per-branch column>` → e.g.
> `std_branch_mean_width_um` is the *standard deviation* of *non-sprout edge*
> *mean equivalent widths*; `p90_minus_p10_sprout_and_branch_length_um` is
> the *P90 − P10 spread* of *all-edge centerline lengths*.

> `<aggregate>_<node subset>_<per-junction column>` → e.g.
> `mean_sprout_tip_dist_nearest_junction_um` is the *mean* over *sprout-tip
> nodes* of the *Euclidean distance to the nearest branch-point junction*.

---

## Mathematical caveats and visual intuition

### Tortuosity definition and clipping

For every edge in the cleaned graph the pipeline computes

$$\tau = \frac{L_{path}}{L_{endpoints} + \varepsilon}, \qquad \tau \leftarrow \mathrm{clip}(\tau,\, 1,\, 50)$$

where $L_{path}$ is the integrated centerline arc-length, $L_{endpoints}$ is
the straight-line distance between the two endpoints, and $\varepsilon =
10^{-8}\ \mu m$ guards against a literal zero denominator. The clipping
bounds exist for two distinct reasons:

- **Lower bound $\tau \ge 1$.** Geometrically a curve can never be shorter
  than the straight line between its endpoints, so $\tau < 1$ is
  unphysical. In practice $\tau$ can drop slightly below $1$ from
  floating-point round-off when summing many short polyline segments on a
  near-straight branch; clamping at $1$ removes that numerical artefact
  without distorting any real curvature.
- **Upper bound $\tau \le 50$.** When a branch forms a near-closed loop the
  two endpoints can come arbitrarily close in space, so $L_{endpoints} \to
  0$ and $\tau$ explodes. A handful of such degenerate edges would
  otherwise dominate any mean / standard-deviation aggregate and dwarf the
  signal from real winding vessels. Capping at $50$ leaves any biologically
  realistic value untouched (typical vessel tortuosity is in $[1, 3]$,
  pathologically tortuous tumour vessels can reach $\sim 10$) while
  bounding pathological cases. This is also why the curated panel reports
  `p90_minus_p10_branch_tortuosity` rather than the standard deviation:
  percentile-based spread is insensitive to whether a few clipped values
  sit at the cap, whereas $\mathrm{std}$ would be inflated by them.

### Fractal dimension

Computed by box-counting on the cleaned graph-derived skeleton mask. It
spans from line-like (1D) to area-like (2D) behaviour; treat it as a
scale-dependent **complexity index** of the centerline network — higher
generally means more branching and better space-filling. Because it is
computed from the skeleton, it emphasises **architecture rather than vessel
thickness**: two networks of the same shape but different vessel calibres
will have nearly identical fractal dimension.

  <img src="README_images/fractal_dimension.png" width="75%" />

  <img src="README_images/equivalent_fractal_dimension.png" width="75%" />

### Lacunarity

Computed on the same cleaned skeleton mask. **Lower** values indicate a
**more evenly distributed** centerline network; **higher** values indicate
stronger spatial **clustering / patchiness** of branches.

A common confusion: *"isn't this measuring the same thing as the spread in
vessel cross-sectional area?"* No — calibre and pore spread are
**size-distribution** metrics, while skeleton lacunarity is a
**spatial-organisation** metric. You can match one and change the other.

  <img src="README_images/lacunarity.png" width="75%" />

### Convex hull and exclusion regions

When the pipeline is given an `exclusion_mask_xy` (e.g. an organoid mask),
it is treated as **unavailable space** at every step that involves a
denominator volume:

1. The exclusion region is z-extruded to a 3D mask and **zeroed in the
   segmentation** before any vessel statistics are computed.
2. Its z-extruded volume is **subtracted from $V_{chip}$**.
3. For $V_{hull}$, we first compute the geometric convex hull of the vessel
   point cloud (which has organoid voxels already zeroed), then test which
   voxels of the z-extruded exclusion region fall *inside* that hull and
   subtract their volume:

$$V_{hull} \; = \; V_{hull,\,raw} \; - \; V_{exclusion \,\cap\, hull}.$$

This matters because an organoid that sits *inside* the vascular envelope
would otherwise inflate the denominator of `vessel_volume_fraction` and
`branch_length_per_hull_volume_um_inverse2`, biasing them downward in
proportion to the organoid size. With the correction in place,
`vessel_volume_fraction = V_{vessel}/(V_{gel \cap hull})` — exactly the
biologically meaningful quantity "fraction of the gel space inside the
vascular envelope that is occupied by vessels".

If no exclusion mask is supplied, $V_{hull}$ reduces to the raw convex-hull
volume.

**Implementation note (performance).** The inside/outside test uses a
`scipy.spatial.Delaunay` triangulation of the hull vertices and is
vectorised over all candidate exclusion voxels. To keep the cost low on
large fields of view, the exclusion voxels are first prefiltered by the
hull's axis-aligned bounding box, so only voxels that *could* lie inside
the hull are passed to the inside-test. On a realistic
$60 \times 1024 \times 1024$ image with a $\sim 5\,$M-voxel z-extruded
organoid, the entire correction adds well under a second on top of the
underlying `ConvexHull` computation that the pipeline already performs.

### Distance convention

There are multiple ways to measure "distance to the nearest …". As
illustrated below, the nearest sprout tip to the highlighted branch point
could be one of two options depending on the metric chosen.

  <img src="README_images/distance_metrics.png" width="75%" />

The pipeline currently sets `junction_distance_mode = 'skeleton'`:

- **`skeleton`** (default): nearest-neighbour distances are graph
  shortest-path lengths along vessel centerlines — the biologically
  traversable route.
- **`euclidean`**: nearest-neighbour distances are straight-line
  distances in physical space.

The per-junction Euclidean distances stored in `junction_metrics_df` (and
hence in the disaggregated junction stats) are computed independently and
are always Euclidean. To prevent name collisions, the whole-network
headline distance metrics carry an explicit mode infix in their column
name: `*_skeleton_dist_*` for along-skeleton distances and
`*_euclidean_dist_*` for straight-line distances. The disaggregated
per-junction columns retain the shorter `*_dist_nearest_*` form (always
Euclidean).

### Floating sprouts

A **floating sprout** is a connected component of the cleaned graph in
which every node is a degree-1 tip — i.e. a small vessel fragment that is
not attached to any junction. Previously the cleaning step only removed
*fully isolated* nodes (degree 0), so two-node sprout-edge components
(both endpoints degree 1, connected by one edge) survived into the metrics
but were never explicitly counted. The pipeline now reports
`total_number_of_floating_sprouts` and
`floating_sprouts_per_hull_volume_um_inverse3` in
`*_all_morphological_params.csv`.

These components can represent either real biological detachments (e.g.
short isolated capillary fragments) or segmentation noise (a single
voxel-thin bridge missed by the binarisation). They are reported as a
separate audit metric and are deliberately not part of the curated panel.
