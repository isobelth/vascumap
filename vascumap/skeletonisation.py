"""
Skeletonisation and vascular-network analysis module.

Extracted from bel_skeletonisation.ipynb for pipeline integration.
"""

import numpy as np
import pandas as pd

import cupy as cp
import cupyx.scipy.ndimage as ndi_gpu
import dask.array as da
from skimage.morphology import skeletonize as skeletonize_3d
import sknw
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull, Delaunay
from scipy.ndimage import distance_transform_edt as edt
from scipy import ndimage as ndi
from scipy.ndimage import maximum_filter
from utils import cupy_chunk_processing
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def safe_divide(numerator, denominator):
    """Divide numerator by denominator, returning NaN if denominator is zero or negative."""
    denominator = float(denominator)
    if denominator <= 0:
        return np.nan
    return float(numerator) / denominator


def safe_median(values):
    """Compute the median of values, returning NaN if no finite values are available."""
    arr = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def safe_percentile_spread(values, low=10, high=90):
    """Return the spread between the low and high percentiles of values."""
    arr = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    q = np.percentile(arr, [low, high])
    return float(q[1] - q[0])


def trim_segmentation(segmentation, fill_threshold=0.75):
    """Trim top/bottom slices where the fill fraction exceeds *fill_threshold*.

    Only peels from the outer edges — middle slices are never removed.

    Returns:
        (trimmed_segmentation, keep_start, keep_stop)  where keep_stop is
        exclusive, suitable for slicing ``arr[keep_start:keep_stop]``.
    """
    slice_fill = segmentation.astype(bool).mean(axis=(1, 2))
    keep_start = 0
    while keep_start < len(slice_fill) and slice_fill[keep_start] > fill_threshold:
        keep_start += 1
    keep_end = len(slice_fill) - 1
    while keep_end >= keep_start and slice_fill[keep_end] > fill_threshold:
        keep_end -= 1
    keep_stop = keep_end + 1
    return segmentation[keep_start:keep_stop], keep_start, keep_stop


# ---------------------------------------------------------------------------
# Skeleton / graph helpers
# ---------------------------------------------------------------------------

def measure_edge_length(coordinates):
    """Sum the Euclidean lengths of polyline segments defined by coordinates."""
    differences = np.diff(coordinates, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return np.sum(segment_lengths)


def prune_graph(graph, area_3d, edt_cutoff=0.25, length_cutoff=25):
    """Iteratively remove short or thin terminal branches from the vessel graph."""
    while True:
        endpoint_nodes = [node for node, degree in graph.degree() if degree == 1]
        values = []
        for node in endpoint_nodes:
            neighbors = list(graph.neighbors(node))

            if len(neighbors) == 1:
                neighbor = neighbors[0]
                edge_data = graph.get_edge_data(neighbor, node)
                edge_length = measure_edge_length(edge_data['pts'])
                branch_edt = area_3d[edge_data['pts'][:, 0], edge_data['pts'][:, 1], edge_data['pts'][:, 2]]
                branch_edt_interp = np.interp(np.linspace(0, 1, 100), np.linspace(0, 1, branch_edt.size), branch_edt)

                if np.mean(branch_edt_interp[:50]) < np.mean(branch_edt_interp[-50:]):
                    neighbor_coords = edge_data['pts'][-1]
                    part_oi = branch_edt_interp[:20]
                else:
                    neighbor_coords = edge_data['pts'][0]
                    part_oi = branch_edt_interp[-20:]

                neighbor_edt = area_3d[neighbor_coords[0], neighbor_coords[1], neighbor_coords[2]]
                value = np.mean(part_oi) / (neighbor_edt + 1e-6)

                if value > edt_cutoff or edge_length <= length_cutoff:
                    graph.remove_node(node)
                    values.append(value)

        if len(values) == 0:
            break
    return graph


def remove_mid_node(graph):
    """Remove degree-2 nodes by merging their two edges into one."""
    while True:
        nodes_to_process = [n for n, d in graph.degree() if d == 2]
        if not nodes_to_process:
            break

        processed_in_iteration = False
        for i in nodes_to_process:
            if not graph.has_node(i) or graph.degree(i) != 2:
                continue

            neighbors = list(graph.neighbors(i))
            if len(neighbors) != 2:
                continue

            n1, n2 = neighbors[0], neighbors[1]
            if n1 == n2 or graph.has_edge(n1, n2):
                continue

            edge1 = graph.get_edge_data(i, n1)
            edge2 = graph.get_edge_data(i, n2)
            pts1 = np.atleast_2d(edge1['pts'])
            pts2 = np.atleast_2d(edge2['pts'])
            node_coord = graph.nodes[i]['pts'].astype(np.int32)

            s1, e1 = pts1[0], pts1[-1]
            s2, e2 = pts2[0], pts2[-1]

            dists = cdist([s1, e1], [s2, e2])
            min_row, min_col = np.unravel_index(np.argmin(dists), dists.shape)

            if min_row == 0 and min_col == 0:
                combined_line = np.concatenate([pts1[::-1], [node_coord], pts2], axis=0)
            elif min_row == 1 and min_col == 1:
                combined_line = np.concatenate([pts1, [node_coord], pts2[::-1]], axis=0)
            elif min_row == 0 and min_col == 1:
                combined_line = np.concatenate([pts2[::-1], [node_coord], pts1], axis=0)
            else:
                combined_line = np.concatenate([pts1, [node_coord], pts2], axis=0)

            new_weight = edge1.get('weight', 0) + edge2.get('weight', 0)
            graph.add_edge(n1, n2, weight=new_weight, pts=combined_line)
            graph.remove_node(i)
            processed_in_iteration = True

        if not processed_in_iteration:
            break
    return graph


def collect_border_vicinity_edges(graph, image_shape, vicinity_xy=50, inplace=False):
    """Remove graph edges that pass within vicinity_xy pixels of the image XY border."""
    border_vicinity_edges = set()
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if any((
                pt[1] < vicinity_xy or pt[1] > image_shape[1] - 1 - vicinity_xy or
                pt[2] < vicinity_xy or pt[2] > image_shape[2] - 1 - vicinity_xy
                ) for pt in pts):
                border_vicinity_edges.add((u, v))
        except KeyError:
            continue

    # B3: skip copy when caller has already done it
    g = graph if inplace else graph.copy()
    edges_to_remove = [edge for edge in border_vicinity_edges if g.has_edge(*edge)]
    g.remove_edges_from(edges_to_remove)

    isolated_nodes = [node for node in g.nodes() if g.degree[node] == 0]
    if isolated_nodes:
        g.remove_nodes_from(isolated_nodes)

    return g


def collect_exclusion_zone_edges(graph, exclusion_mask_xy, inplace=False):
    """Remove graph edges that pass through the exclusion zone (e.g. organoid region)."""
    exclusion_edges = set()
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if any(exclusion_mask_xy[int(pt[1]), int(pt[2])] for pt in pts):
                exclusion_edges.add((u, v))
        except (KeyError, IndexError):
            continue

    # B3: skip copy when caller has already done it
    g = graph if inplace else graph.copy()
    edges_to_remove = [e for e in exclusion_edges if g.has_edge(*e)]
    g.remove_edges_from(edges_to_remove)

    isolated = [n for n in g.nodes() if g.degree[n] == 0]
    if isolated:
        g.remove_nodes_from(isolated)

    return g


def compute_cross_sectional_areas(mask, skeleton, binary_edt, voxel_size_um=(2.0, 2.0, 2.0)):
    """Compute per-skeleton-voxel cross-sectional areas from EDT and 2D EDT."""
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    edt_2d = edt(np.max(mask, axis=0), sampling=tuple(voxel_size_um[1:]))
    area_3d = np.zeros_like(binary_edt, dtype=float)
    z_idx, y_idx, x_idx = np.where(skeleton > 0)

    minor_axis = 2 * binary_edt[z_idx, y_idx, x_idx]
    major_axis = 2 * edt_2d[y_idx, x_idx]

    areas = np.pi * (major_axis / 2) * (minor_axis / 2)
    area_3d[z_idx, y_idx, x_idx] = areas
    return area_3d


def fractal_dimension_and_lacunarity(binary, min_box_size=1, max_box_size=None, n_samples=12):
    """Estimate fractal dimension and lacunarity of a binary volume via box-counting."""
    pts = np.argwhere(binary > 0)
    if pts.size == 0:
        return np.nan, np.nan

    if max_box_size is None:
        max_box_size = int(np.floor(np.log2(np.min(binary.shape))))

    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2)).astype(np.int64)
    scales = np.unique(scales)
    scales = scales[scales > 0]

    log_inv_scale = []
    log_N = []
    lac_vals = []

    for s in scales:
        box_ids = pts // s
        unique_box_ids, counts = np.unique(box_ids, axis=0, return_counts=True)
        N = unique_box_ids.shape[0]
        if N < 2:
            continue

        log_inv_scale.append(np.log(1.0 / s))
        log_N.append(np.log(N))

        mu = counts.mean()
        lac_vals.append((counts.var() / (mu * mu)) if mu > 0 else np.nan)

    if len(log_N) < 2:
        return np.nan, np.nan

    fd = float(np.polyfit(log_inv_scale, log_N, 1)[0])
    lac = float(np.nanmean(lac_vals)) if len(lac_vals) else np.nan
    return fd, lac


def graph2image(graph, shape):
    """Rasterise graph edges back into a binary 3D volume of the given shape."""
    pruned_skeleton = np.zeros(shape)
    for u, v in graph.edges():
        coords = graph.get_edge_data(u, v)['pts']

        clipped_coords = np.zeros_like(coords)
        clipped_coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
        clipped_coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)
        clipped_coords[:, 2] = np.clip(coords[:, 2], 0, shape[2] - 1)

        pruned_skeleton[clipped_coords[:, 0], clipped_coords[:, 1], clipped_coords[:, 2]] = 1

    return pruned_skeleton


def orientation_to_device_axis_deg(pts_um, device_axis='x'):
    """Return acute branch orientation angle (deg) relative to device axis in XY plane.

    The device segmentation step rectifies/crops to the device frame, so the
    device long axis is treated as the +X axis in this aligned space.
    """
    if pts_um is None or len(pts_um) < 2:
        return np.nan

    vec = np.asarray(pts_um[-1] - pts_um[0], dtype=float)
    # XY plane in array coordinates is (x, y) -> indices (2, 1)
    vec_xy = np.asarray([vec[2], vec[1]], dtype=float)
    norm = float(np.linalg.norm(vec_xy))
    if not np.isfinite(norm) or norm <= 1e-8:
        return np.nan

    unit = vec_xy / norm
    axis_xy = np.array([1.0, 0.0], dtype=float) if str(device_axis).lower() == 'x' else np.array([0.0, 1.0], dtype=float)
    dot = float(np.clip(np.abs(np.dot(unit, axis_xy)), -1.0, 1.0))
    # Acute angle to axis: 0° aligned, 90° orthogonal
    return float(np.degrees(np.arccos(dot)))


def compute_branch_metrics_df(graph, area_image, voxel_size_um=(2.0, 2.0, 2.0), device_axis='x'):
    """Compute per-branch metrics DataFrame from a cleaned vessel graph."""
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    rows = []

    for u, v in graph.edges():
        try:
            pts = np.asarray(graph[u][v]['pts'])
            if pts.ndim != 2 or pts.shape[0] < 2:
                continue

            pts = pts.astype(int)
            pts_um = pts.astype(float) * voxel_size_um[None, :]
            seg_len_um = np.linalg.norm(np.diff(pts_um, axis=0), axis=1)
            path_length_um = float(np.sum(seg_len_um))
            endpoint_distance_um = float(np.linalg.norm(pts_um[-1] - pts_um[0]))

            seg_areas_um2 = np.asarray(area_image[pts[:, 0], pts[:, 1], pts[:, 2]], dtype=float)
            mean_cs_area_um2 = float(np.nanmean(seg_areas_um2)) if seg_areas_um2.size else np.nan
            median_cs_area_um2 = float(np.nanmedian(seg_areas_um2)) if seg_areas_um2.size else np.nan
            std_cs_area_um2 = float(np.nanstd(seg_areas_um2)) if seg_areas_um2.size else np.nan

            # Equivalent circular width from area
            valid_area = seg_areas_um2[np.isfinite(seg_areas_um2) & (seg_areas_um2 > 0)]
            if valid_area.size:
                eq_widths_um = np.sqrt((4.0 * valid_area) / np.pi)
                mean_width_um = float(np.nanmean(eq_widths_um))
                median_width_um = float(np.nanmedian(eq_widths_um))
            else:
                mean_width_um = np.nan
                median_width_um = np.nan

            # Volume approximation from path length and mean cross-sectional area
            branch_volume_um3 = float(mean_cs_area_um2 * path_length_um) if np.isfinite(mean_cs_area_um2) else np.nan

            # Tortuosity: ratio of path length to straight-line distance
            tortuosity = float(np.clip(path_length_um / (endpoint_distance_um + 1e-8), 1.0, 50.0))

            row = {
                'node_start': int(u),
                'node_end': int(v),
                'is_sprout': bool(graph.nodes[u].get('sprout', False) or graph.nodes[v].get('sprout', False)),
                'start_z_idx': int(pts[0, 0]),
                'start_y_idx': int(pts[0, 1]),
                'start_x_idx': int(pts[0, 2]),
                'end_z_idx': int(pts[-1, 0]),
                'end_y_idx': int(pts[-1, 1]),
                'end_x_idx': int(pts[-1, 2]),
                'start_z': float(pts_um[0, 0]),
                'start_y': float(pts_um[0, 1]),
                'start_x': float(pts_um[0, 2]),
                'end_z': float(pts_um[-1, 0]),
                'end_y': float(pts_um[-1, 1]),
                'end_x': float(pts_um[-1, 2]),
                'path_length': path_length_um,
                'endpoint_distance': endpoint_distance_um,
                'tortuosity': tortuosity,
                'mean_cs_area': mean_cs_area_um2,
                'median_cs_area': median_cs_area_um2,
                'std_cs_area': std_cs_area_um2,
                'mean_width': mean_width_um,
                'median_width': median_width_um,
                'branch_volume': branch_volume_um3,
                'orientation_to_device_axis': orientation_to_device_axis_deg(pts_um, device_axis=device_axis),
            }
            rows.append(row)
        except (KeyError, IndexError, ValueError):
            continue

    if not rows:
        return pd.DataFrame(columns=[
            'node_start', 'node_end', 'is_sprout',
            'start_z_idx', 'start_y_idx', 'start_x_idx', 'end_z_idx', 'end_y_idx', 'end_x_idx',
            'start_z', 'start_y', 'start_x', 'end_z', 'end_y', 'end_x',
            'path_length', 'endpoint_distance', 'tortuosity',
            'mean_cs_area', 'median_cs_area', 'std_cs_area',
            'mean_width', 'median_width', 'branch_volume',
            'orientation_to_device_axis',
        ])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-junction metrics
# ---------------------------------------------------------------------------

def compute_junction_metrics_df(graph, voxel_size_um=(2.0, 2.0, 2.0), distance_threshold_um=500.0):
    """Compute per-junction/endpoint metrics DataFrame from a cleaned vessel graph.

    Returns one row per graph node with coordinates, degree, type, nearest-
    neighbour distances (Euclidean, in µm), and neighbourhood counts.
    """
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    empty_cols = [
        'node_id', 'z', 'y', 'x',
        'degree', 'is_sprout_tip', 'is_junction',
        'dist_nearest_junction', 'dist_nearest_endpoint',
        'num_junction_neighbors', 'num_endpoint_neighbors',
    ]
    nodes = list(graph.nodes())
    if not nodes:
        return pd.DataFrame(columns=empty_cols)

    positions_px = np.array([graph.nodes[n]['pts'] for n in nodes], dtype=float)
    positions_um = positions_px * voxel_size_um[None, :]

    degrees = np.array([graph.degree[n] for n in nodes])
    is_sprout_tip = np.array([bool(graph.nodes[n].get('sprout', False)) for n in nodes])
    is_junction = ~is_sprout_tip

    n_nodes = len(nodes)
    dist_nearest_junction = np.full(n_nodes, np.nan)
    dist_nearest_endpoint = np.full(n_nodes, np.nan)
    num_junction_neighbors = np.zeros(n_nodes, dtype=int)
    num_endpoint_neighbors = np.zeros(n_nodes, dtype=int)

    junction_idx = np.where(is_junction)[0]
    endpoint_idx = np.where(is_sprout_tip)[0]

    if n_nodes >= 2:
        dist_matrix = cdist(positions_um, positions_um)

        if len(junction_idx) > 0 and len(endpoint_idx) > 0:
            jj = dist_matrix[np.ix_(junction_idx, junction_idx)]
            np.fill_diagonal(jj, np.inf)
            if jj.shape[1] > 1:
                dist_nearest_junction[junction_idx] = np.min(jj, axis=1)
                num_junction_neighbors[junction_idx] = np.sum(jj <= distance_threshold_um, axis=1)

            je = dist_matrix[np.ix_(junction_idx, endpoint_idx)]
            dist_nearest_endpoint[junction_idx] = np.min(je, axis=1)
            num_endpoint_neighbors[junction_idx] = np.sum(je <= distance_threshold_um, axis=1)

            ej = dist_matrix[np.ix_(endpoint_idx, junction_idx)]
            dist_nearest_junction[endpoint_idx] = np.min(ej, axis=1)
            num_junction_neighbors[endpoint_idx] = np.sum(ej <= distance_threshold_um, axis=1)

            ee = dist_matrix[np.ix_(endpoint_idx, endpoint_idx)]
            np.fill_diagonal(ee, np.inf)
            if ee.shape[1] > 1:
                dist_nearest_endpoint[endpoint_idx] = np.min(ee, axis=1)
                num_endpoint_neighbors[endpoint_idx] = np.sum(ee <= distance_threshold_um, axis=1)

        elif len(junction_idx) > 1:
            jj = dist_matrix[np.ix_(junction_idx, junction_idx)]
            np.fill_diagonal(jj, np.inf)
            dist_nearest_junction[junction_idx] = np.min(jj, axis=1)
            num_junction_neighbors[junction_idx] = np.sum(jj <= distance_threshold_um, axis=1)

        elif len(endpoint_idx) > 1:
            ee = dist_matrix[np.ix_(endpoint_idx, endpoint_idx)]
            np.fill_diagonal(ee, np.inf)
            dist_nearest_endpoint[endpoint_idx] = np.min(ee, axis=1)
            num_endpoint_neighbors[endpoint_idx] = np.sum(ee <= distance_threshold_um, axis=1)

    return pd.DataFrame({
        'node_id': nodes,
        'z': positions_um[:, 0],
        'y': positions_um[:, 1],
        'x': positions_um[:, 2],
        'degree': degrees,
        'is_sprout_tip': is_sprout_tip,
        'is_junction': is_junction,
        'dist_nearest_junction': dist_nearest_junction,
        'dist_nearest_endpoint': dist_nearest_endpoint,
        'num_junction_neighbors': num_junction_neighbors,
        'num_endpoint_neighbors': num_endpoint_neighbors,
    })


# ---------------------------------------------------------------------------
# Comprehensive morphological parameters (all legacy + current metrics)
# ---------------------------------------------------------------------------

def compute_all_morphological_params(global_metrics, branch_metrics_df, junction_metrics_df):
    """Build single-row DataFrame with all morphological parameters.

    Includes everything in the simplified CSV plus full disaggregated
    statistics: mean/std/median × branch/sprout/combined for vessel metrics,
    and mean/std/median × junction/sprout_tip/combined for junction metrics.
    """
    params = dict(global_metrics)

    # ── Per-branch disaggregated stats ────────────────────────────────────
    if branch_metrics_df is not None and not branch_metrics_df.empty:
        is_sprout = branch_metrics_df['is_sprout'].values

        vessel_agg_columns = {
            'volume': 'branch_volume',
            'length': 'path_length',
            'endpoint_distance': 'endpoint_distance',
            'tortuosity': 'tortuosity',
            'mean_cs_area': 'mean_cs_area',
            'median_cs_area': 'median_cs_area',
            'std_cs_area': 'std_cs_area',
            'mean_width': 'mean_width',
            'median_width': 'median_width',
            'orientation': 'orientation_to_device_axis',
        }

        for param_name, col_name in vessel_agg_columns.items():
            if col_name not in branch_metrics_df.columns:
                continue
            values = branch_metrics_df[col_name].to_numpy(dtype=float)
            branch_vals = values[~is_sprout]
            sprout_vals = values[is_sprout]

            for agg, fn in [
                ('mean', np.nanmean),
                ('std', np.nanstd),
                ('median', np.nanmedian),
                ('spread', safe_percentile_spread),
            ]:
                params[f'{agg}_branch_{param_name}'] = float(fn(branch_vals)) if len(branch_vals) > 0 else np.nan
                params[f'{agg}_sprout_{param_name}'] = float(fn(sprout_vals)) if len(sprout_vals) > 0 else np.nan
                params[f'{agg}_sprout_and_branch_{param_name}'] = float(fn(values)) if len(values) > 0 else np.nan

    # ── Per-junction disaggregated stats ──────────────────────────────────
    if junction_metrics_df is not None and not junction_metrics_df.empty:
        is_junction = junction_metrics_df['is_junction'].values
        is_sprout_tip = junction_metrics_df['is_sprout_tip'].values

        junction_agg_columns = [
            'degree',
            'dist_nearest_junction',
            'dist_nearest_endpoint',
            'num_junction_neighbors',
            'num_endpoint_neighbors',
        ]

        for col_name in junction_agg_columns:
            if col_name not in junction_metrics_df.columns:
                continue
            values = junction_metrics_df[col_name].to_numpy(dtype=float)
            junc_vals = values[is_junction]
            tip_vals = values[is_sprout_tip]

            for agg, fn in [
                ('mean', np.nanmean),
                ('std', np.nanstd),
                ('median', np.nanmedian),
                ('spread', safe_percentile_spread),
            ]:
                params[f'{agg}_junction_{col_name}'] = float(fn(junc_vals)) if len(junc_vals) > 0 else np.nan
                params[f'{agg}_sprout_tip_{col_name}'] = float(fn(tip_vals)) if len(tip_vals) > 0 else np.nan
                params[f'{agg}_all_nodes_{col_name}'] = float(fn(values)) if len(values) > 0 else np.nan

    return pd.DataFrame([params])


# ---------------------------------------------------------------------------
# Curated, shape-invariant analysis-metrics panel (PCA / clustering)
# ---------------------------------------------------------------------------

# Curated subset of `all_morphological_params_df` chosen to be:
#   1. Shape-invariant (no field-of-view-dependent quantities)
#   2. Biologically interpretable
#   3. Small enough for PCA / clustering with modest sample sizes
#
# Every column listed here MUST also appear in the full
# `all_morphological_params_df`, so the curated panel is a strict subset of
# the full audit file. See `vascumap/Analysis_README.md` for the per-feature
# mathematical and biological description.
ANALYSIS_METRICS_COLUMNS = [
    # Density (5)
    'vessel_volume_fraction',
    'branch_length_per_hull_volume',
    'sprouts_per_hull_volume',
    'junctions_per_hull_volume',
    'branches_per_hull_volume',
    # Topology (2)
    'skeleton_fractal_dimension',
    'skeleton_lacunarity',
    # Branch geometry — branches only (4)
    'median_branch_length',
    'spread_branch_length',
    'median_branch_median_cs_area',
    'spread_branch_median_cs_area',
    # Tortuosity — branch-only (2)
    'median_branch_tortuosity',
    'spread_branch_tortuosity',
    # Junction connectivity (2)
    'median_junction_degree',
    'spread_junction_degree',
    # Orientation — combined sprouts + branches (2)
    'median_sprout_and_branch_orientation',
    'spread_sprout_and_branch_orientation',
]


def build_curated_analysis_metrics_df(all_params_df):
    """Return a single-row DataFrame containing the curated PCA/clustering panel.

    Selects `ANALYSIS_METRICS_COLUMNS` from `all_params_df`. Any missing
    column (e.g. for an empty graph) is filled with NaN so the schema is
    stable across images.
    """
    if all_params_df is None or all_params_df.empty:
        return pd.DataFrame([{c: np.nan for c in ANALYSIS_METRICS_COLUMNS}])

    row = {}
    for col in ANALYSIS_METRICS_COLUMNS:
        row[col] = all_params_df[col].iloc[0] if col in all_params_df.columns else np.nan
    return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Network headline metrics
# ---------------------------------------------------------------------------

def summarize_network_headline_metrics(graph, area_image, voxel_size_um=(2.0, 2.0, 2.0), distance_mode='skeleton', device_axis='x'):
    """Aggregate per-branch metrics into network-level headline statistics.

    Distance-based summaries are emitted with a mode-tagged infix
    (`_skeleton_dist_` or `_euclidean_dist_`) so they never collide with the
    per-junction Euclidean disaggregated stats produced from
    `compute_junction_metrics_df` (which use plain `dist_nearest_*` keys).
    """
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    mode_tag = str(distance_mode).lower()
    if mode_tag not in ('skeleton', 'euclidean'):
        mode_tag = 'skeleton'
    summary = {
        'median_sprout_and_branch_orientation': np.nan,
        'spread_sprout_and_branch_orientation': np.nan,
        'median_sprout_and_branch_median_cs_area': np.nan,
        'spread_sprout_and_branch_median_cs_area': np.nan,
        'median_sprout_and_branch_length': np.nan,
        'spread_sprout_and_branch_length': np.nan,
        f'median_junction_{mode_tag}_dist_nearest_junction': np.nan,
        f'spread_junction_{mode_tag}_dist_nearest_junction': np.nan,
        f'median_sprout_{mode_tag}_dist_nearest_endpoint': np.nan,
        f'spread_sprout_{mode_tag}_dist_nearest_endpoint': np.nan,
    }

    orientations_deg = []
    median_cs_areas = []
    branch_lengths = []
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if len(pts) < 2:
                continue
            pts_um = np.asarray(pts, dtype=float) * voxel_size_um[None, :]
            orientations_deg.append(orientation_to_device_axis_deg(pts_um, device_axis=device_axis))

            segment_areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            median_cs_areas.append(float(np.nanmedian(segment_areas)))

            branch_lengths.append(float(np.sum(np.linalg.norm(np.diff(pts_um, axis=0), axis=1))))
        except (KeyError, IndexError):
            continue

    if len(orientations_deg) > 0:
        summary['median_sprout_and_branch_orientation'] = safe_median(orientations_deg)
        summary['spread_sprout_and_branch_orientation'] = safe_percentile_spread(orientations_deg)
    if len(median_cs_areas) > 0:
        summary['median_sprout_and_branch_median_cs_area'] = safe_median(median_cs_areas)
        summary['spread_sprout_and_branch_median_cs_area'] = safe_percentile_spread(median_cs_areas)
    if len(branch_lengths) > 0:
        summary['median_sprout_and_branch_length'] = safe_median(branch_lengths)
        summary['spread_sprout_and_branch_length'] = safe_percentile_spread(branch_lengths)

    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return summary

    positions = np.array([graph.nodes[n]['pts'] for n in nodes])
    positions_um = positions.astype(float) * voxel_size_um[None, :]
    node_type = np.array(['sprout' if graph.nodes[n]['sprout'] else 'junction' for n in nodes])

    if distance_mode == 'skeleton':
        graph_weighted = graph.copy()
        for u, v, data in graph_weighted.edges(data=True):
            pts = data.get('pts', None)
            if pts is None or len(pts) < 2:
                graph_weighted.edges[u, v]['path_length_um'] = np.inf
                continue
            pts_um = np.asarray(pts, dtype=float) * voxel_size_um[None, :]
            seg_lengths = np.linalg.norm(np.diff(pts_um, axis=0), axis=1)
            graph_weighted.edges[u, v]['path_length_um'] = float(np.sum(seg_lengths))

        # B2: only run Dijkstra from nodes of each relevant type and record the
        # nearest-neighbour distance — avoids building an O(N²) distance matrix.
        junction_nodes = [n for n, t in zip(nodes, node_type) if t == 'junction']
        sprout_nodes   = [n for n, t in zip(nodes, node_type) if t == 'sprout']
        junction_set   = set(junction_nodes)
        sprout_set     = set(sprout_nodes)

        def _nearest_same_type(source_list, same_set):
            nearest = []
            for source in source_list:
                lengths = nx.single_source_dijkstra_path_length(
                    graph_weighted, source, weight='path_length_um'
                )
                others = [d for target, d in lengths.items()
                          if target in same_set and target != source and np.isfinite(d)]
                if others:
                    nearest.append(min(others))
            return nearest

        junction_nearest = _nearest_same_type(junction_nodes, junction_set)
        sprout_nearest   = _nearest_same_type(sprout_nodes,   sprout_set)
    else:
        # Euclidean: still build sub-distance matrices but only for same-type nodes
        junction_mask  = node_type == 'junction'
        sprout_mask    = node_type == 'sprout'
        pos_junc       = positions_um[junction_mask]
        pos_sprout     = positions_um[sprout_mask]

        def _nearest_euclidean(pos):
            if len(pos) < 2:
                return []
            d = cdist(pos, pos)
            np.fill_diagonal(d, np.inf)
            near = np.min(d, axis=1)
            return list(near[np.isfinite(near)])

        junction_nearest = _nearest_euclidean(pos_junc)
        sprout_nearest   = _nearest_euclidean(pos_sprout)

    if junction_nearest:
        summary[f'median_junction_{mode_tag}_dist_nearest_junction'] = safe_median(junction_nearest)
        summary[f'spread_junction_{mode_tag}_dist_nearest_junction'] = safe_percentile_spread(junction_nearest)

    if sprout_nearest:
        summary[f'median_sprout_{mode_tag}_dist_nearest_endpoint'] = safe_median(sprout_nearest)
        summary[f'spread_sprout_{mode_tag}_dist_nearest_endpoint'] = safe_percentile_spread(sprout_nearest)

    return summary


# ---------------------------------------------------------------------------
# Internal pore metrics
# ---------------------------------------------------------------------------

def compute_internal_pore_headline_metrics(
    mask,
    voxel_size_um=(2.0, 2.0, 2.0),
    min_pore_area_um2=16.0,
    max_pore_area_fraction_of_slice=0.15,
    use_gpu_edt=True,
    exclusion_mask_xy=None,
):
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    _, y_um, x_um = voxel_size_um
    pixel_area_um2 = float(y_um * x_um)

    total_pore_area_um2 = 0.0
    total_filled_area_um2 = 0.0
    pore_areas_all = []
    pore_radii_all = []

    # B1: two-pass approach — collect pore slices first, batch all GPU EDT transfers,
    # then accumulate metrics.  Reduces N×(CPU→GPU + GPU→CPU) to 1×(CPU→GPU + GPU→CPU).
    pore_slice_data = []  # list of dicts: {z, pores, labeled, valid_label_ids, valid_areas, filled_area_um2}

    for z in range(mask.shape[0]):
        vessel_slice = mask[z].astype(bool)
        filled_slice = ndi.binary_fill_holes(vessel_slice)
        internal_pores = filled_slice & ~vessel_slice

        if exclusion_mask_xy is not None:
            internal_pores = internal_pores & ~exclusion_mask_xy
            filled_slice   = filled_slice   & ~exclusion_mask_xy

        filled_area_um2 = float(np.count_nonzero(filled_slice)) * pixel_area_um2
        total_filled_area_um2 += filled_area_um2

        if not np.any(internal_pores):
            continue

        labeled, n_labels = ndi.label(internal_pores, structure=np.ones((3, 3), dtype=np.uint8))
        if n_labels == 0:
            continue

        area_counts   = np.bincount(labeled.ravel(), minlength=n_labels + 1)[1:].astype(np.float64)
        area_um2_all  = area_counts * pixel_area_um2

        slice_area_um2   = float(vessel_slice.size) * pixel_area_um2
        max_pore_area_um2 = float(max_pore_area_fraction_of_slice) * slice_area_um2

        label_ids  = np.arange(1, n_labels + 1, dtype=np.int32)
        valid_mask = (area_um2_all >= min_pore_area_um2) & (area_um2_all <= max_pore_area_um2)
        if not np.any(valid_mask):
            continue

        pore_slice_data.append({
            'z':               z,
            'pores':           internal_pores,
            'labeled':         labeled,
            'valid_label_ids': label_ids[valid_mask],
            'valid_areas':     area_um2_all[valid_mask],
        })

    if not pore_slice_data:
        return {
            'median_internal_pore_area': np.nan,
            'spread_internal_pore_area': np.nan,
        }

    # ── GPU batch EDT ──────────────────────────────────────────────────────
    if use_gpu_edt:
        pore_stack    = np.stack([d['pores'] for d in pore_slice_data], axis=0)  # (N, H, W) uint8
        pore_gpu      = cp.asarray(pore_stack.astype(np.uint8))
        dist_list_gpu = [
            ndi_gpu.distance_transform_edt(pore_gpu[i], sampling=(float(y_um), float(x_um)))
            for i in range(pore_gpu.shape[0])
        ]
        dist_stack    = cp.asnumpy(cp.stack(dist_list_gpu, axis=0))  # (N, H, W) – single transfer back
        del pore_gpu, dist_list_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        dist_stack = np.stack([
            edt(d['pores'], sampling=(y_um, x_um)) for d in pore_slice_data
        ], axis=0)

    # ── Accumulate metrics ────────────────────────────────────────────────
    for idx, d in enumerate(pore_slice_data):
        dist_map_um    = dist_stack[idx]
        valid_radii    = np.asarray(
            ndi.maximum(dist_map_um, labels=d['labeled'], index=d['valid_label_ids']),
            dtype=float,
        )
        pore_areas_all.append(d['valid_areas'])
        pore_radii_all.append(valid_radii)
        total_pore_area_um2 += float(np.sum(d['valid_areas']))

    all_areas = np.concatenate(pore_areas_all)
    all_radii = np.concatenate(pore_radii_all)

    return {
        'median_internal_pore_area': float(np.median(all_areas)),
        'spread_internal_pore_area': float(np.percentile(all_areas, 90) - np.percentile(all_areas, 10)),
    }


def build_internal_pore_label_volumes(
    mask,
    voxel_size_um=(2.0, 2.0, 2.0),
    max_pore_area_fraction_of_slice=0.15,
):
    """Build per-slice pore label and distance volumes for napari visualisation.

    Returns:
        (holes, hole_labels_per_slice, hole_distance_per_slice_um) as numpy arrays.
    """
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    holes = np.zeros_like(mask, dtype=np.uint8)
    hole_labels_per_slice = np.zeros_like(mask, dtype=np.int32)
    hole_distance_per_slice_um = np.zeros_like(mask, dtype=np.float32)

    for z in range(mask.shape[0]):
        vessel_slice = mask[z].astype(bool)
        filled_slice = ndi.binary_fill_holes(vessel_slice)
        internal_pores = filled_slice & ~vessel_slice

        labeled, n_labels = ndi.label(internal_pores, structure=np.ones((3, 3), dtype=np.uint8))
        if n_labels == 0:
            continue

        area_counts = np.bincount(labeled.ravel(), minlength=n_labels + 1).astype(np.float64)
        max_pore_area_px = float(max_pore_area_fraction_of_slice) * float(vessel_slice.size)
        valid_label_mask = (area_counts > 0) & (area_counts <= max_pore_area_px)
        valid_label_mask[0] = False

        filtered_pores_slice = valid_label_mask[labeled]
        holes[z] = filtered_pores_slice.astype(np.uint8)

        relabeled_slice, _ = ndi.label(filtered_pores_slice, structure=np.ones((3, 3), dtype=np.uint8))
        hole_labels_per_slice[z] = relabeled_slice.astype(np.int32)

        if np.any(filtered_pores_slice):
            hole_distance_per_slice_um[z] = edt(
                filtered_pores_slice,
                sampling=tuple(voxel_size_um[1:]),
            ).astype(np.float32)

    return holes, hole_labels_per_slice, hole_distance_per_slice_um


# ---------------------------------------------------------------------------
# Skeleton overview visualisation
# ---------------------------------------------------------------------------

def generate_skeleton_overview_plot(segmentation, analysis_results, title="", save_path=None,
                                    brightfield_stack=None, organoid_mask_xy=None,
                                    brightfield_full=None, device_corners_xy=None,
                                    organoid_mask_full_xy=None):
    """Generate and optionally save the 6-panel skeleton/graph overview plot.

    Panels: (0) full XY plane with device geometry and tumour overlay,
    (1) cropped brightfield (no overlay), (2) reconstructed vasculature sum projection,
    (3) full skeleton overlay, (4) clean skeleton overlay,
    (5) pruned clean graph.
    """

    skeleton = analysis_results['skeleton']
    clean_skeleton = analysis_results['skeleton_from_graph']
    clean_graph = analysis_results['clean_graph']
    binary_edt = analysis_results['binary_edt']

    nz = segmentation.shape[0]
    seg_bool = segmentation.astype(bool)
    seg_max = np.mean(seg_bool, axis=0).astype(np.float32)

    # Zero out tumour/organoid region on all segmentation-based panels
    if organoid_mask_xy is not None:
        org_mask = np.asarray(organoid_mask_xy, dtype=bool)
        if org_mask.shape == seg_max.shape:
            seg_max = seg_max.copy()
            seg_max[org_mask] = 0.0

    background = np.stack([seg_max * 0.40] * 3, axis=-1)

    _SKEL_COLOUR = np.array([0.0, 1.0, 1.0])  # cyan

    def _make_overlay(seg_bg, arr):
        thick = maximum_filter(np.sum(arr.astype(np.float32), axis=0), size=3)
        rgb = np.stack([seg_bg * 0.40] * 3, axis=-1)
        mask = thick > 0
        rgb[mask] = _SKEL_COLOUR  # full-brightness cyan wherever skeleton is present
        return np.clip(rgb, 0, 1)

    overlay_skel = _make_overlay(seg_max, skeleton)
    overlay_clean = _make_overlay(seg_max, clean_skeleton)

    def _edge_diameters(g):
        diams = []
        for u, v in g.edges():
            try:
                pts = np.clip(g[u][v]['pts'].astype(int),
                              [0, 0, 0],
                              [s - 1 for s in binary_edt.shape])
                radii = binary_edt[pts[:, 0], pts[:, 1], pts[:, 2]]
                diams.append(float(np.median(radii)) * 2.0)
            except (KeyError, IndexError):
                diams.append(0.0)
        return diams

    clean_diams = _edge_diameters(clean_graph)

    all_diams = [d for d in clean_diams if d > 0]
    vmin = np.percentile(all_diams, 5) * 0.5 if all_diams else 0
    vmax = np.percentile(all_diams, 95) if all_diams else 1
    norm_g = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm_g, cmap=cm.magma)

    def _edge_orientations(g):
        orientations = []
        for u, v in g.edges():
            try:
                pts = g[u][v]['pts'].astype(float)
                orientations.append(orientation_to_device_axis_deg(pts))
            except (KeyError, IndexError):
                orientations.append(np.nan)
        return orientations

    clean_orientations = _edge_orientations(clean_graph)

    valid_orient = [o for o in clean_orientations if np.isfinite(o)]
    norm_orient = Normalize(vmin=0, vmax=90)
    sm_orient = cm.ScalarMappable(norm=norm_orient, cmap=cm.magma)

    def _draw_graph(ax, g, edge_values, scalar_mappable, graph_title):
        ax.imshow(background)
        for (u, v), val in zip(g.edges(), edge_values):
            try:
                pts = g[u][v]['pts'].astype(int)
                color = scalar_mappable.to_rgba(val) if np.isfinite(val) else (0.5, 0.5, 0.5, 1.0)
                ax.plot(pts[:, 2], pts[:, 1],
                        color=color, linewidth=2.0, solid_capstyle='round')
            except (KeyError, IndexError):
                continue
        nx_x, nx_y, nc = [], [], []
        for node in g.nodes():
            pos = g.nodes[node]['pts']
            if pos.ndim > 1:
                pos = pos[0]
            nx_x.append(pos[2])
            nx_y.append(pos[1])
            nc.append('limegreen' if g.degree(node) == 1 else 'white')
        if nx_x:
            ax.scatter(nx_x, nx_y, c=nc, s=12, alpha=0.9, zorder=5)
        ax.set_title(f'{graph_title}\n({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)',
                     fontsize=13)
        ax.set_aspect('equal', adjustable='box')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=6, figsize=(36, 16))

    # ── Panel 0: full XY plane with device corners + tumour overlay ──────────
    if brightfield_full is not None:
        bf_full = np.asarray(brightfield_full, dtype=np.float32)
        if bf_full.ndim == 3:
            bf_full = np.mean(bf_full, axis=-1)
        bfmin, bfmax = bf_full.min(), bf_full.max()
        bf_norm = (bf_full - bfmin) / max(float(bfmax - bfmin), 1e-6)
        rgb_full = np.stack([bf_norm] * 3, axis=-1)
        if organoid_mask_full_xy is not None:
            omask = np.asarray(organoid_mask_full_xy, dtype=bool)
            if omask.shape == bf_full.shape:
                rgb_full[omask] = rgb_full[omask] * 0.5 + np.array([1.0, 0.0, 0.0]) * 0.5
        ax[0].imshow(np.clip(rgb_full, 0, 1))
        if device_corners_xy is not None:
            corners = np.asarray(device_corners_xy)
            closed = np.vstack([corners, corners[0:1]])
            ax[0].plot(closed[:, 0], closed[:, 1], color='yellow', linewidth=2)
        p0_title = 'Full plane + device geometry'
        if organoid_mask_full_xy is not None:
            p0_title += '\n(tumour mask, red)'
        ax[0].set_title(p0_title, fontsize=13)
    else:
        ax[0].axis('off')

    # ── Panel 1: cropped brightfield, no overlay ─────────────────────────────
    if brightfield_stack is not None:
        bf_2d = np.mean(brightfield_stack.astype(np.float32), axis=0)
        bf_min, bf_max = bf_2d.min(), bf_2d.max()
        bf_norm2 = (bf_2d - bf_min) / max(float(bf_max - bf_min), 1e-6)
        ax[1].imshow(bf_norm2, cmap='gray')
        ax[1].set_title('Cropped brightfield', fontsize=13)
    else:
        ax[1].axis('off')

    # ── Panel 2: reconstructed vasculature sum projection ────────────────────
    ax[2].imshow(seg_max, cmap='gray')
    ax[2].set_title(f'Reconstructed vasculature  ({nz} z-slices)', fontsize=13)

    # ── Panel 3: clean skeleton overlaid on segmentation ─────────────────────
    ax[3].imshow(overlay_clean)
    ax[3].set_title(f'Clean skeleton  ({int(clean_skeleton.sum()):,} voxels)', fontsize=13)

    # ── Panel 4: pruned clean graph (coloured by diameter) ──────────────────
    _draw_graph(ax[4], clean_graph, clean_diams, sm, 'Clean graph — diameter')

    # ── Panel 5: pruned clean graph (coloured by orientation) ─────────────────
    _draw_graph(ax[5], clean_graph, clean_orientations, sm_orient, 'Clean graph — orientation')

    fig.colorbar(sm, ax=ax[4], fraction=0.02, pad=0.02, label='Vessel diameter (\u00b5m)')
    fig.colorbar(sm_orient, ax=ax[5], fraction=0.02, pad=0.02, label='Orientation to device axis (\u00b0)')
    for a in ax:
        a.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clean_and_analyse(
    vasculature_segmentation,
    voxel_size_um=(2.0, 2.0, 2.0),
    junction_distance_mode='skeleton',
    exclusion_mask_xy=None,
):
    """Run full skeletonisation, graph extraction, pruning, and metric computation.

    Args:
        vasculature_segmentation: 3-D binary mask (z, y, x).
        voxel_size_um: Isotropic or anisotropic voxel size in micrometers.
        junction_distance_mode: 'skeleton' for Dijkstra path-length or 'euclidean'.
        exclusion_mask_xy: Optional 2-D bool mask (y, x). True pixels are excluded
            from all analysis (e.g. organoid region). Subtracted from chip volume,
            zeroed in the segmentation, and excluded from pore detection.

    Returns:
        dict with keys: global_metrics, global_metrics_df, clean_segmentation,
        binary_edt, skeleton, graph, area_image, pruned_graph, clean_graph,
        skeleton_from_graph.
    """
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    voxel_volume_um3 = float(np.prod(voxel_size_um))
    chunk_size = (vasculature_segmentation.shape[0], 512, 512)
    max_internal_pore_area_fraction_of_slice = 0.10

    # Determine the device long axis from the image footprint (y=shape[1], x=shape[2]).
    # If x-extent >= y-extent the image is landscape → long axis is x; otherwise y.
    _h, _w = vasculature_segmentation.shape[1], vasculature_segmentation.shape[2]
    device_axis = 'x' if _w >= _h else 'y'

    # ---- apply exclusion mask (e.g. organoid region) ----
    if exclusion_mask_xy is not None:
        exclusion_mask_xy = np.asarray(exclusion_mask_xy, dtype=bool)
        vasculature_segmentation = vasculature_segmentation.copy()
        vasculature_segmentation[:, exclusion_mask_xy] = 0

    # ---- clean segmentation ----
    binary_smoothed = cupy_chunk_processing(
        volume=vasculature_segmentation.astype(np.float32),
        processing_func=ndi_gpu.gaussian_filter,
        sigma=3,
        chunk_size=chunk_size,
    ) > 0.5
    clean_segmentation = cupy_chunk_processing(
        volume=binary_smoothed,
        processing_func=ndi_gpu.binary_fill_holes,
        chunk_size=chunk_size,
    ).astype(np.float32)
    binary_edt = cupy_chunk_processing(
        volume=clean_segmentation,
        processing_func=ndi_gpu.distance_transform_edt,
        sampling=tuple(voxel_size_um),
        chunk_size=chunk_size,
    )

    # ---- skeletonise and graph ----
    dask_volume = da.from_array(clean_segmentation.astype(bool), chunks=chunk_size)
    skeleton = da.overlap.map_overlap(
        dask_volume,
        skeletonize_3d,
        depth=(2, 2, 2),
        boundary='none',
        dtype=bool,
    ).compute(scheduler='threads')
    graph = sknw.build_sknw(skeleton)
    area_image = compute_cross_sectional_areas(
        vasculature_segmentation,
        skeleton,
        binary_edt,
        voxel_size_um=voxel_size_um,
    )

    for n in list(graph.nodes()):
        graph.nodes[n]['pts'] = graph.nodes[n]['pts'][0]
        graph.nodes[n]['sprout'] = graph.degree(n) == 1

    pruned_graph = prune_graph(graph=graph, area_3d=area_image, edt_cutoff=0.20, length_cutoff=25)
    clean_graph = remove_mid_node(pruned_graph)
    # B3: copy once here; both trimming functions operate inplace on that single copy
    clean_graph = clean_graph.copy()
    collect_border_vicinity_edges(clean_graph, vasculature_segmentation.shape, inplace=True)
    if exclusion_mask_xy is not None:
        collect_exclusion_zone_edges(clean_graph, exclusion_mask_xy, inplace=True)
    isolated_nodes = [node for node in clean_graph.nodes() if clean_graph.degree[node] == 0]
    if isolated_nodes:
        clean_graph.remove_nodes_from(isolated_nodes)

    skeleton_from_graph = graph2image(clean_graph, vasculature_segmentation.shape).astype(np.uint8)
    for node in clean_graph.nodes():
        clean_graph.nodes[node]['sprout'] = clean_graph.degree(node) == 1

    # ---- metrics ----
    global_metrics = {}
    chip_volume_um3 = float(np.prod(vasculature_segmentation.shape)) * voxel_volume_um3
    # Subtract excluded region from chip volume
    if exclusion_mask_xy is not None:
        excluded_voxels = int(np.count_nonzero(exclusion_mask_xy)) * vasculature_segmentation.shape[0]
        chip_volume_um3 -= float(excluded_voxels) * voxel_volume_um3
    vessel_volume_um3 = float(np.count_nonzero(clean_segmentation)) * voxel_volume_um3

    # ---- convex hull volume of segmented region ----
    # The convex hull is the geometric envelope of the vessel point cloud.
    # If an exclusion region (e.g. an organoid) sits inside that envelope,
    # its volume must be subtracted so it is not counted as available
    # space in `vessel_volume_fraction = V_vessel / V_hull` or in any
    # `*_per_chip_volume*` density that divides by `V_hull`.
    seg_pts = np.argwhere(clean_segmentation > 0)
    if len(seg_pts) >= 4:
        try:
            seg_pts_um = seg_pts * voxel_size_um[None, :]
            hull = ConvexHull(seg_pts_um)
            convex_hull_volume_um3 = hull.volume

            if exclusion_mask_xy is not None and np.any(exclusion_mask_xy):
                # Volume of the (z-extruded) exclusion region that falls
                # *inside* the convex hull. Prefilter to the hull's bbox so
                # find_simplex only runs on candidate voxels.
                hull_min_um = seg_pts_um[hull.vertices].min(axis=0)
                hull_max_um = seg_pts_um[hull.vertices].max(axis=0)
                z_um_full = np.arange(vasculature_segmentation.shape[0]) * voxel_size_um[0]
                z_in = np.where((z_um_full >= hull_min_um[0]) & (z_um_full <= hull_max_um[0]))[0]
                ys, xs = np.where(exclusion_mask_xy)
                y_um = ys * voxel_size_um[1]
                x_um = xs * voxel_size_um[2]
                xy_in = np.where(
                    (y_um >= hull_min_um[1]) & (y_um <= hull_max_um[1])
                    & (x_um >= hull_min_um[2]) & (x_um <= hull_max_um[2])
                )[0]
                if z_in.size and xy_in.size:
                    hull_delaunay = Delaunay(seg_pts_um[hull.vertices])
                    z_rep = np.repeat(z_in, xy_in.size)
                    y_rep = np.tile(ys[xy_in], z_in.size)
                    x_rep = np.tile(xs[xy_in], z_in.size)
                    excl_coords_um = np.stack([z_rep, y_rep, x_rep], axis=1).astype(float) \
                        * voxel_size_um[None, :]
                    inside = hull_delaunay.find_simplex(excl_coords_um) >= 0
                    excluded_inside_hull_um3 = float(np.count_nonzero(inside)) * voxel_volume_um3
                    convex_hull_volume_um3 = max(
                        convex_hull_volume_um3 - excluded_inside_hull_um3, 0.0,
                    )
        except Exception:
            convex_hull_volume_um3 = chip_volume_um3
    else:
        convex_hull_volume_um3 = chip_volume_um3

    if clean_graph.number_of_edges() > 0:
        fd, lacunarity = fractal_dimension_and_lacunarity(skeleton_from_graph > 0)
        # All-edge length (sprouts + branches) and branch-only length.
        per_edge_length_um = []
        per_edge_is_sprout = []
        for u, v in clean_graph.edges():
            edge_len = float(np.linalg.norm(
                np.diff(clean_graph[u][v]['pts'].astype(float) * voxel_size_um[None, :], axis=0),
                axis=1,
            ).sum())
            per_edge_length_um.append(edge_len)
            per_edge_is_sprout.append(
                clean_graph.nodes[u]['sprout'] or clean_graph.nodes[v]['sprout']
            )
        per_edge_length_um = np.asarray(per_edge_length_um, dtype=float)
        per_edge_is_sprout = np.asarray(per_edge_is_sprout, dtype=bool)
        total_vessel_length_um = float(per_edge_length_um.sum())
        total_branch_length_um = float(per_edge_length_um[~per_edge_is_sprout].sum())
        branchpoints_count = sum(1 for u in clean_graph.nodes() if not clean_graph.nodes[u]['sprout'])
        sprouts_count = int(per_edge_is_sprout.sum())
        branches_count = int((~per_edge_is_sprout).sum())
        # Floating components: connected components in which every node is a
        # sprout (degree-1 endpoint). These are vessel fragments not attached
        # to the branching network — e.g. a single edge whose two endpoints
        # are both degree-1.
        floating_sprouts_count = sum(
            1 for cc in nx.connected_components(clean_graph)
            if all(clean_graph.nodes[n]['sprout'] for n in cc)
        )
    else:
        fd, lacunarity = np.nan, np.nan
        total_vessel_length_um = 0.0
        total_branch_length_um = 0.0
        branchpoints_count = 0
        sprouts_count = 0
        branches_count = 0
        floating_sprouts_count = 0

    global_metrics['chip_volume'] = chip_volume_um3
    global_metrics['convex_hull_volume'] = convex_hull_volume_um3
    global_metrics['vessel_volume'] = vessel_volume_um3
    global_metrics['vessel_volume_fraction'] = safe_divide(vessel_volume_um3, convex_hull_volume_um3)
    global_metrics['total_vessel_length'] = float(total_vessel_length_um)
    global_metrics['total_branch_length'] = float(total_branch_length_um)
    # Length-density: branch-only length per hull volume (sprouts excluded).
    global_metrics['branch_length_per_hull_volume'] = safe_divide(
        total_branch_length_um, convex_hull_volume_um3,
    )
    global_metrics['sprouts_per_vessel_length'] = safe_divide(sprouts_count, total_vessel_length_um)
    global_metrics['junctions_per_vessel_length'] = safe_divide(branchpoints_count, total_vessel_length_um)
    global_metrics['skeleton_fractal_dimension'] = fd
    global_metrics['skeleton_lacunarity'] = lacunarity
    global_metrics['median_sprout_and_branch_orientation'] = np.nan
    global_metrics['spread_sprout_and_branch_orientation'] = np.nan
    global_metrics['median_sprout_and_branch_median_cs_area'] = np.nan
    global_metrics['spread_sprout_and_branch_median_cs_area'] = np.nan
    global_metrics['median_sprout_and_branch_length'] = np.nan
    global_metrics['spread_sprout_and_branch_length'] = np.nan
    global_metrics['median_junction_skeleton_dist_nearest_junction'] = np.nan
    global_metrics['spread_junction_skeleton_dist_nearest_junction'] = np.nan
    global_metrics['median_sprout_skeleton_dist_nearest_endpoint'] = np.nan
    global_metrics['spread_sprout_skeleton_dist_nearest_endpoint'] = np.nan

    branch_metrics_df = compute_branch_metrics_df(
        clean_graph,
        area_image,
        voxel_size_um=voxel_size_um,
        device_axis=device_axis,
    )

    if clean_graph.number_of_nodes() > 0 and clean_graph.number_of_edges() > 0:
        global_metrics.update(
            summarize_network_headline_metrics(
                clean_graph,
                area_image,
                voxel_size_um=voxel_size_um,
                distance_mode=junction_distance_mode,
                device_axis=device_axis,
            )
        )

    global_metrics['average_vessel_volume'] = (
        float(np.nanmean(branch_metrics_df['branch_volume'].to_numpy(dtype=float)))
        if not branch_metrics_df.empty
        else np.nan
    )

    pore_global_metrics = compute_internal_pore_headline_metrics(
        clean_segmentation.astype(bool),
        voxel_size_um=voxel_size_um,
        min_pore_area_um2=16.0,
        max_pore_area_fraction_of_slice=max_internal_pore_area_fraction_of_slice,
        use_gpu_edt=True,
        exclusion_mask_xy=exclusion_mask_xy,
    )
    global_metrics.update(pore_global_metrics)

    # ---- extra density metrics (per hull volume) ----
    global_metrics['sprouts_per_hull_volume'] = safe_divide(sprouts_count, convex_hull_volume_um3)
    global_metrics['junctions_per_hull_volume'] = safe_divide(branchpoints_count, convex_hull_volume_um3)
    global_metrics['branches_per_hull_volume'] = safe_divide(branches_count, convex_hull_volume_um3)
    global_metrics['floating_sprouts_per_hull_volume'] = safe_divide(
        floating_sprouts_count, convex_hull_volume_um3,
    )

    # ---- raw counts (for comprehensive output) ----
    global_metrics['total_number_of_sprouts'] = int(sprouts_count)
    global_metrics['total_number_of_branches'] = int(branches_count)
    global_metrics['total_number_of_junctions'] = int(branchpoints_count)
    global_metrics['total_number_of_floating_sprouts'] = int(floating_sprouts_count)
    global_metrics['total_number_of_edges'] = int(clean_graph.number_of_edges())
    global_metrics['total_number_of_nodes'] = int(clean_graph.number_of_nodes())

    # ---- junction metrics ----
    junction_metrics_df = compute_junction_metrics_df(
        clean_graph,
        voxel_size_um=voxel_size_um,
        distance_threshold_um=500.0,
    )

    # ---- comprehensive all-params DataFrame ----
    all_morphological_params_df = compute_all_morphological_params(
        global_metrics, branch_metrics_df, junction_metrics_df,
    )

    # ---- curated, shape-invariant analysis-metrics panel (PCA / clustering) ----
    analysis_metrics_df = build_curated_analysis_metrics_df(all_morphological_params_df)

    global_metrics_df = pd.DataFrame([global_metrics])
    return {
        'global_metrics': global_metrics,
        'global_metrics_df': global_metrics_df,
        'branch_metrics_df': branch_metrics_df,
        'junction_metrics_df': junction_metrics_df,
        'all_morphological_params_df': all_morphological_params_df,
        'analysis_metrics_df': analysis_metrics_df,
        'voxel_size_um': voxel_size_um,
        'chunk_size': chunk_size,
        'clean_segmentation': clean_segmentation,
        'binary_edt': binary_edt,
        'skeleton': skeleton,
        'graph': graph,
        'area_image': area_image,
        'pruned_graph': pruned_graph,
        'clean_graph': clean_graph,
        'skeleton_from_graph': skeleton_from_graph,
    }
