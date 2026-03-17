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
from scipy.ndimage import distance_transform_edt as edt
from scipy import ndimage as ndi
from utils import cupy_chunk_processing


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def safe_divide(numerator, denominator):
    denominator = float(denominator)
    if denominator <= 0:
        return np.nan
    return float(numerator) / denominator


def safe_median(values):
    arr = np.asarray(pd.to_numeric(values, errors='coerce'), dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return np.nan
    return float(np.median(arr))


def safe_percentile_spread(values, low=10, high=90):
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
    differences = np.diff(coordinates, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return np.sum(segment_lengths)


def prune_graph(graph, area_3d, edt_cutoff=0.25, length_cutoff=25):
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


def collect_border_vicinity_edges(graph, image_shape, vicinity_xy=50):
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

    trimmed_subgraph = graph.copy()
    edges_to_remove = [edge for edge in border_vicinity_edges if trimmed_subgraph.has_edge(*edge)]
    trimmed_subgraph.remove_edges_from(edges_to_remove)

    isolated_nodes = [node for node in trimmed_subgraph.nodes() if trimmed_subgraph.degree[node] == 0]
    if isolated_nodes:
        trimmed_subgraph.remove_nodes_from(isolated_nodes)

    return trimmed_subgraph


def collect_exclusion_zone_edges(graph, exclusion_mask_xy):
    """Remove graph edges that pass through the exclusion zone (e.g. organoid region)."""
    exclusion_edges = set()
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if any(exclusion_mask_xy[int(pt[1]), int(pt[2])] for pt in pts):
                exclusion_edges.add((u, v))
        except (KeyError, IndexError):
            continue

    trimmed = graph.copy()
    edges_to_remove = [e for e in exclusion_edges if trimmed.has_edge(*e)]
    trimmed.remove_edges_from(edges_to_remove)

    isolated = [n for n in trimmed.nodes() if trimmed.degree[n] == 0]
    if isolated:
        trimmed.remove_nodes_from(isolated)

    return trimmed


def compute_cross_sectional_areas(mask, skeleton, binary_edt, voxel_size_um=(2.0, 2.0, 2.0)):
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
    pruned_skeleton = np.zeros(shape)
    for u, v in graph.edges():
        coords = graph.get_edge_data(u, v)['pts']

        clipped_coords = np.zeros_like(coords)
        clipped_coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
        clipped_coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)
        clipped_coords[:, 2] = np.clip(coords[:, 2], 0, shape[2] - 1)

        pruned_skeleton[clipped_coords[:, 0], clipped_coords[:, 1], clipped_coords[:, 2]] = 1

    return pruned_skeleton


# ---------------------------------------------------------------------------
# Network headline metrics
# ---------------------------------------------------------------------------

def summarize_network_headline_metrics(graph, area_image, voxel_size_um=(2.0, 2.0, 2.0), distance_mode='skeleton'):
    voxel_size_um = np.asarray(voxel_size_um, dtype=float)
    summary = {
        'median_sprout_and_branch_tortuosity': np.nan,
        'p90_minus_p10_sprout_and_branch_tortuosity': np.nan,
        'median_sprout_and_branch_median_cs_area_um2': np.nan,
        'p90_minus_p10_sprout_and_branch_median_cs_area_um2': np.nan,
        'median_junction_dist_nearest_junction_um': np.nan,
        'p90_minus_p10_junction_dist_nearest_junction_um': np.nan,
        'median_sprout_dist_nearest_endpoint_um': np.nan,
        'p90_minus_p10_sprout_dist_nearest_endpoint_um': np.nan,
    }

    tortuosities = []
    median_cs_areas = []
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if len(pts) < 2:
                continue
            pts_um = np.asarray(pts, dtype=float) * voxel_size_um[None, :]
            segment_lengths_um = np.linalg.norm(np.diff(pts_um, axis=0), axis=1)
            length_um = float(np.sum(segment_lengths_um))
            shortest_path_um = float(np.linalg.norm(pts_um[0] - pts_um[-1]))
            tortuosities.append(np.clip(length_um / (shortest_path_um + 1e-8), 0, 5))

            segment_areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            median_cs_areas.append(float(np.nanmedian(segment_areas)))
        except (KeyError, IndexError):
            continue

    if len(tortuosities) > 0:
        summary['median_sprout_and_branch_tortuosity'] = safe_median(tortuosities)
        summary['p90_minus_p10_sprout_and_branch_tortuosity'] = safe_percentile_spread(tortuosities)
    if len(median_cs_areas) > 0:
        summary['median_sprout_and_branch_median_cs_area_um2'] = safe_median(median_cs_areas)
        summary['p90_minus_p10_sprout_and_branch_median_cs_area_um2'] = safe_percentile_spread(median_cs_areas)

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

        node_to_idx = {n: i for i, n in enumerate(nodes)}
        dist_matrix = np.full((len(nodes), len(nodes)), np.inf, dtype=float)
        for source in nodes:
            i = node_to_idx[source]
            lengths = nx.single_source_dijkstra_path_length(
                graph_weighted, source, weight='path_length_um'
            )
            for target, d in lengths.items():
                j = node_to_idx[target]
                dist_matrix[i, j] = float(d)
    else:
        dist_matrix = cdist(positions_um, positions_um)

    endpoint_mask = (node_type == 'sprout')
    branch_point_mask = (node_type == 'junction')

    if np.any(branch_point_mask):
        sub_dist = dist_matrix[np.ix_(branch_point_mask, branch_point_mask)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        nearest = np.min(sub_dist, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if nearest.size > 0:
            summary['median_junction_dist_nearest_junction_um'] = safe_median(nearest)
            summary['p90_minus_p10_junction_dist_nearest_junction_um'] = safe_percentile_spread(nearest)

    if np.any(endpoint_mask):
        sub_dist = dist_matrix[np.ix_(endpoint_mask, endpoint_mask)].copy()
        np.fill_diagonal(sub_dist, np.inf)
        nearest = np.min(sub_dist, axis=1)
        nearest = nearest[np.isfinite(nearest)]
        if nearest.size > 0:
            summary['median_sprout_dist_nearest_endpoint_um'] = safe_median(nearest)
            summary['p90_minus_p10_sprout_dist_nearest_endpoint_um'] = safe_percentile_spread(nearest)

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

    for z in range(mask.shape[0]):
        vessel_slice = mask[z].astype(bool)
        filled_slice = ndi.binary_fill_holes(vessel_slice)
        internal_pores = filled_slice & ~vessel_slice

        # Exclude organoid / exclusion region from pore and filled-area accounting
        if exclusion_mask_xy is not None:
            internal_pores = internal_pores & ~exclusion_mask_xy
            filled_slice = filled_slice & ~exclusion_mask_xy

        filled_area_um2 = float(np.count_nonzero(filled_slice)) * pixel_area_um2
        total_filled_area_um2 += filled_area_um2

        if not np.any(internal_pores):
            continue

        labeled, n_labels = ndi.label(internal_pores, structure=np.ones((3, 3), dtype=np.uint8))
        if n_labels == 0:
            continue

        area_counts = np.bincount(labeled.ravel(), minlength=n_labels + 1)[1:].astype(np.float64)
        area_um2_all = area_counts * pixel_area_um2

        slice_area_um2 = float(vessel_slice.size) * pixel_area_um2
        max_pore_area_um2 = float(max_pore_area_fraction_of_slice) * slice_area_um2

        label_ids = np.arange(1, n_labels + 1, dtype=np.int32)
        valid_mask = (area_um2_all >= min_pore_area_um2) & (area_um2_all <= max_pore_area_um2)
        if not np.any(valid_mask):
            continue

        valid_label_ids = label_ids[valid_mask]
        valid_areas = area_um2_all[valid_mask]

        if use_gpu_edt:
            dist_map_um = cp.asnumpy(
                ndi_gpu.distance_transform_edt(
                    cp.asarray(internal_pores, dtype=cp.uint8),
                    sampling=(float(y_um), float(x_um)),
                )
            )
        else:
            dist_map_um = edt(internal_pores, sampling=(y_um, x_um))

        valid_radii = np.asarray(
            ndi.maximum(dist_map_um, labels=labeled, index=valid_label_ids),
            dtype=float,
        )

        pore_areas_all.append(valid_areas)
        pore_radii_all.append(valid_radii)
        total_pore_area_um2 += float(np.sum(valid_areas))

    if len(pore_areas_all) == 0:
        return {
            'total_internal_pore_count': 0,
            'internal_pore_area_fraction_in_filled_vascular_area': 0.0,
            'median_internal_pore_area_um2': np.nan,
            'p90_minus_p10_internal_pore_area_um2': np.nan,
            'median_internal_pore_max_inscribed_radius_um': np.nan,
            'p90_minus_p10_internal_pore_max_inscribed_radius_um': np.nan,
        }

    all_areas = np.concatenate(pore_areas_all)
    all_radii = np.concatenate(pore_radii_all)

    return {
        'total_internal_pore_count': int(all_areas.size),
        'internal_pore_area_fraction_in_filled_vascular_area': float(total_pore_area_um2 / max(total_filled_area_um2, 1e-12)),
        'median_internal_pore_area_um2': float(np.median(all_areas)),
        'p90_minus_p10_internal_pore_area_um2': float(np.percentile(all_areas, 90) - np.percentile(all_areas, 10)),
        'median_internal_pore_max_inscribed_radius_um': float(np.median(all_radii)),
        'p90_minus_p10_internal_pore_max_inscribed_radius_um': float(np.percentile(all_radii, 90) - np.percentile(all_radii, 10)),
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

def generate_skeleton_overview_plot(segmentation, analysis_results, title="", save_path=None):
    """Generate and optionally save the 4-panel skeleton/graph overview plot.

    Panels: (1) full skeleton overlay, (2) clean skeleton overlay,
    (3) graph from clean skeleton with diameter colouring,
    (4) pruned clean graph with diameter colouring.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from scipy.ndimage import maximum_filter

    skeleton = analysis_results['skeleton']
    clean_skeleton = analysis_results['skeleton_from_graph']
    clean_graph = analysis_results['clean_graph']
    binary_edt = analysis_results['binary_edt']

    nz = segmentation.shape[0]
    seg_bool = segmentation.astype(bool)
    seg_max = np.mean(seg_bool, axis=0).astype(np.float32)
    background = np.stack([seg_max * 0.40] * 3, axis=-1)

    def _make_overlay(seg_bg, arr):
        thick = maximum_filter(np.sum(arr.astype(np.float32), axis=0), size=3)
        norm = thick / max(thick.max(), 1)
        rgb = np.stack([seg_bg * 0.40] * 3, axis=-1)
        mask = norm > 0
        rgb[mask] = cm.get_cmap('cool')(norm[mask])[:, :3]
        return np.clip(rgb, 0, 1)

    overlay_skel = _make_overlay(seg_max, skeleton)
    overlay_clean = _make_overlay(seg_max, clean_skeleton)

    # Build a graph from the clean skeleton image for display
    raw_graph = sknw.build_sknw(clean_skeleton.astype(bool))
    for _n in list(raw_graph.nodes()):
        if raw_graph.nodes[_n]['pts'].ndim > 1:
            raw_graph.nodes[_n]['pts'] = raw_graph.nodes[_n]['pts'][0]

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

    raw_diams = _edge_diameters(raw_graph)
    clean_diams = _edge_diameters(clean_graph)

    all_diams = [d for d in raw_diams + clean_diams if d > 0]
    vmin = np.percentile(all_diams, 5) * 0.5 if all_diams else 0
    vmax = np.percentile(all_diams, 95) if all_diams else 1
    norm_g = Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm_g, cmap=cm.magma)

    def _draw_graph(ax, g, diams, graph_title):
        ax.imshow(background)
        for (u, v), diam in zip(g.edges(), diams):
            try:
                pts = g[u][v]['pts'].astype(int)
                ax.plot(pts[:, 2], pts[:, 1],
                        color=sm.to_rgba(diam), linewidth=2.0, solid_capstyle='round')
            except (KeyError, IndexError):
                continue
        nx_x, nx_y, nc = [], [], []
        for node in g.nodes():
            pos = g.nodes[node]['pts']
            if pos.ndim > 1:
                pos = pos[0]
            nx_x.append(pos[2])
            nx_y.append(pos[1])
            nc.append('lime' if g.degree(node) == 1 else 'white')
        if nx_x:
            ax.scatter(nx_x, nx_y, c=nc, s=20, alpha=0.8, zorder=5)
        ax.set_title(f'{graph_title}\n({g.number_of_nodes()} nodes, {g.number_of_edges()} edges)',
                     fontsize=13)
        ax.set_aspect('equal', adjustable='box')

    plt.style.use('dark_background')
    fig, ax = plt.subplots(ncols=4, figsize=(24, 16))

    ax[0].imshow(overlay_skel)
    ax[0].set_title(f'Skeleton  ({int(skeleton.sum()):,} voxels,  {nz} z-slices)', fontsize=13)

    ax[1].imshow(overlay_clean)
    ax[1].set_title(f'Clean skeleton  ({int(clean_skeleton.sum()):,} voxels,  {nz} z-slices)', fontsize=13)

    _draw_graph(ax[2], raw_graph, raw_diams, 'Graph (clean skeleton)')
    _draw_graph(ax[3], clean_graph, clean_diams, 'Clean graph (pruned)')

    fig.colorbar(sm, ax=ax[-1], fraction=0.02, pad=0.02, label='Vessel diameter (\u00b5m)')
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
    clean_graph = collect_border_vicinity_edges(clean_graph, vasculature_segmentation.shape)
    if exclusion_mask_xy is not None:
        clean_graph = collect_exclusion_zone_edges(clean_graph, exclusion_mask_xy)
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

    if clean_graph.number_of_edges() > 0:
        fd, lacunarity = fractal_dimension_and_lacunarity(skeleton_from_graph > 0)
        total_vessel_length_um = np.sum([
            np.linalg.norm(
                np.diff(clean_graph[u][v]['pts'].astype(float) * voxel_size_um[None, :], axis=0),
                axis=1,
            ).sum()
            for u, v in clean_graph.edges()
        ])
        branchpoints_count = sum(1 for u in clean_graph.nodes() if not clean_graph.nodes[u]['sprout'])
        sprouts_count = sum(
            1 for u, v in clean_graph.edges()
            if clean_graph.nodes[u]['sprout'] or clean_graph.nodes[v]['sprout']
        )
    else:
        fd, lacunarity = np.nan, np.nan
        total_vessel_length_um = 0.0
        branchpoints_count = 0
        sprouts_count = 0

    global_metrics['chip_volume_um3'] = chip_volume_um3
    global_metrics['vessel_volume_um3'] = vessel_volume_um3
    global_metrics['vessel_volume_fraction'] = safe_divide(vessel_volume_um3, chip_volume_um3)
    global_metrics['total_vessel_length_um'] = float(total_vessel_length_um)
    global_metrics['vessel_length_per_chip_volume_um_inverse2'] = safe_divide(total_vessel_length_um, chip_volume_um3)
    global_metrics['sprouts_per_vessel_length_um_inverse'] = safe_divide(sprouts_count, total_vessel_length_um)
    global_metrics['junctions_per_vessel_length_um_inverse'] = safe_divide(branchpoints_count, total_vessel_length_um)
    global_metrics['skeleton_fractal_dimension'] = fd
    global_metrics['skeleton_lacunarity'] = lacunarity
    global_metrics['median_sprout_and_branch_tortuosity'] = np.nan
    global_metrics['p90_minus_p10_sprout_and_branch_tortuosity'] = np.nan
    global_metrics['median_sprout_and_branch_median_cs_area_um2'] = np.nan
    global_metrics['p90_minus_p10_sprout_and_branch_median_cs_area_um2'] = np.nan
    global_metrics['median_junction_dist_nearest_junction_um'] = np.nan
    global_metrics['p90_minus_p10_junction_dist_nearest_junction_um'] = np.nan
    global_metrics['median_sprout_dist_nearest_endpoint_um'] = np.nan
    global_metrics['p90_minus_p10_sprout_dist_nearest_endpoint_um'] = np.nan

    if clean_graph.number_of_nodes() > 0 and clean_graph.number_of_edges() > 0:
        global_metrics.update(
            summarize_network_headline_metrics(
                clean_graph,
                area_image,
                voxel_size_um=voxel_size_um,
                distance_mode=junction_distance_mode,
            )
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

    # ---- extra density metrics ----
    global_metrics['sprouts_per_chip_volume_um_inverse3'] = safe_divide(sprouts_count, chip_volume_um3)
    global_metrics['junctions_per_chip_volume_um_inverse3'] = safe_divide(branchpoints_count, chip_volume_um3)
    global_metrics['total_internal_pore_density_per_vessel_volume_um_inverse3'] = safe_divide(
        global_metrics.get('total_internal_pore_count', np.nan),
        vessel_volume_um3,
    )

    global_metrics_df = pd.DataFrame([global_metrics])
    return {
        'global_metrics': global_metrics,
        'global_metrics_df': global_metrics_df,
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
