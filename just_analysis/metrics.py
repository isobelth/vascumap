import numpy as np
import pandas as pd
import math
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull
from scipy.ndimage.morphology import distance_transform_edt as edt
from skeletonization import measure_edge_length

def compute_cross_sectional_areas(mask, skeleton_image, binary_edt):
    """
    Compute local cross-sectional areas of vessels at skeleton points.

    Approximates area as an ellipse: Area = pi * (major_axis/2) * (minor_axis/2).
    Minor axis is derived from local 3D EDT (radius).
    Major axis is derived from 2D projection EDT (radius).

    Args:
        mask (np.ndarray): Binary vessel mask.
        skeleton_image (np.ndarray): Binary skeleton image.
        binary_edt (np.ndarray): 3D Euclidean Distance Transform.

    Returns:
        np.ndarray: 3D volume with area values at skeleton points.
    """
    edt_2d = edt(np.max(mask, axis=0))
    area_3d = np.zeros_like(binary_edt, dtype=float)
    z_idx, y_idx, x_idx = np.where(skeleton_image > 0)
    
    minor_axis = 2 * binary_edt[z_idx, y_idx, x_idx] 
    major_axis = 2 * edt_2d[y_idx, x_idx] 
    
    areas = np.pi * (major_axis/2) * (minor_axis/2)
    area_3d[z_idx, y_idx, x_idx] = areas
    return area_3d

def shortest_path(coordinates):
    """Calculate Euclidean distance between first and last points."""
    return np.linalg.norm(coordinates[0] - coordinates[-1])

def compute_vessel_metrics(graph, area_image, vessel_metrics_df):
    """
    Compute geometric and topological metrics for each vessel segment (edge).

    Metrics include: length, tortuosity, volume, cross-sectional area (mean, median, std),
    orientation, connectivity.

    Args:
        graph (nx.Graph): Input graph.
        area_image (np.ndarray): 3D volume with cross-sectional area values.
        vessel_metrics_df (pd.DataFrame): (Unused in this function, returns new DF).

    Returns:
        pd.DataFrame: DataFrame containing metrics for each edge.
    """
    zs, ys, xs = [], [], []
    volumes, lengths, shortest_paths = [], [], []
    tortuosities, is_sprouts = [], []
    mean_cs_areas, median_cs_areas, std_cs_areas = [], [], []
    node1_degrees, node2_degrees = [], []
    orientation_zs, orientation_ys, orientation_xs = [], [], []

    edges = list(graph.edges())
    if not edges: 
        cols = ['z', 'y', 'x', 'volume', 'length', 'shortest_path', 'tortuosity', 
                'is_sprout', 'mean_cs_area', 'median_cs_area', 'std_cs_area',
                'node1_degree', 'node2_degree', 'orientation_z', 'orientation_y', 'orientation_x']
        return pd.DataFrame(columns=cols)

    node_degrees_dict = dict(graph.degree())

    for u, v in edges:
        try:
            pts = graph[u][v]['pts']
            if len(pts) < 2: 
                continue 

            zs.append(pts[:, 0])
            ys.append(pts[:, 1])
            xs.append(pts[:, 2])

            segment_areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            mean_cs_areas.append(np.nanmean(segment_areas))
            median_cs_areas.append(np.nanmedian(segment_areas))
            std_cs_areas.append(np.nanstd(segment_areas))
            
            volumes.append(np.nansum(segment_areas))

            l = measure_edge_length(pts)
            lengths.append(l)
            sp = shortest_path(pts)
            shortest_paths.append(sp)
            
            tort = np.clip(l / (sp + 1e-8), 0, 5) 
            tortuosities.append(tort)

            deg_u = node_degrees_dict.get(u, 0) 
            deg_v = node_degrees_dict.get(v, 0)
            node1_degrees.append(deg_u)
            node2_degrees.append(deg_v)

            is_sprouts.append(graph.nodes[u]['sprout'] or graph.nodes[v]['sprout'])

            direction_vector = pts[-1] - pts[0]
            norm = np.linalg.norm(direction_vector)
            if norm > 1e-8: 
                normalized_vector = direction_vector / norm
            else:
                normalized_vector = np.array([np.nan, np.nan, np.nan])

            orientation_zs.append(normalized_vector[0])
            orientation_ys.append(normalized_vector[1])
            orientation_xs.append(normalized_vector[2])

        except (KeyError, IndexError):
            continue

    valid_edges = [(u,v) for i,(u,v) in enumerate(edges) if i < len(volumes)] 
    vessel_metrics_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(valid_edges, names=['node1', 'node2']))

    vessel_metrics_df['z'] = zs
    vessel_metrics_df['y'] = ys
    vessel_metrics_df['x'] = xs

    vessel_metrics_df['volume'] = volumes
    vessel_metrics_df['length'] = lengths
    vessel_metrics_df['shortest_path'] = shortest_paths
    vessel_metrics_df['tortuosity'] = tortuosities
    vessel_metrics_df['is_sprout'] = is_sprouts
    
    vessel_metrics_df['mean_cs_area'] = mean_cs_areas
    vessel_metrics_df['median_cs_area'] = median_cs_areas
    vessel_metrics_df['std_cs_area'] = std_cs_areas
    vessel_metrics_df['node1_degree'] = node1_degrees
    vessel_metrics_df['node2_degree'] = node2_degrees
    vessel_metrics_df['orientation_z'] = orientation_zs
    vessel_metrics_df['orientation_y'] = orientation_ys
    vessel_metrics_df['orientation_x'] = orientation_xs
 
    return vessel_metrics_df

def compute_junction_metrics(graph, junction_metrics_df, distance_threshold=50):
    """
    Compute metrics for graph nodes (junctions and endpoints).

    Metrics include: type (junction vs sprout tip), degree, distance to nearest neighbor of same/different type.

    Args:
        graph (nx.Graph): Input graph.
        junction_metrics_df (pd.DataFrame): (Unused, returns new DF).
        distance_threshold (int, optional): Threshold for neighbor counting. Defaults to 50.

    Returns:
        pd.DataFrame: DataFrame containing metrics for each node.
    """
    nodes = list(graph.nodes())
    if not nodes: 
            cols = ['z', 'y', 'x', 'number_of_vessel_per_node', 'is_sprout_tip', 'is_junction',
                    'dist_nearest_junction', 'dist_nearest_endpoint',
                    'num_junction_neighbors', 'num_endpoint_neighbors']
            return pd.DataFrame(index=nodes, columns=cols) 

    positions = np.array([graph.nodes[n]['pts'] for n in nodes])

    junction_metrics_df = pd.DataFrame(index=nodes) 
    junction_metrics_df['z'] = positions[:, 0]
    junction_metrics_df['y'] = positions[:, 1]
    junction_metrics_df['x'] = positions[:, 2]

    degrees = np.array([graph.degree[n] for n in nodes])
    junction_metrics_df['number_of_vessel_per_node'] = degrees

    junction_metrics_df['is_sprout_tip'] = [graph.nodes[u]['sprout'] for u in nodes]
    junction_metrics_df['is_junction'] = [not graph.nodes[u]['sprout'] for u in nodes] 

    endpoint_mask = junction_metrics_df['is_sprout_tip'].values
    branch_point_mask = junction_metrics_df['is_junction'].values

    endpoint_indices = np.where(endpoint_mask)[0]
    branch_point_indices = np.where(branch_point_mask)[0]

    has_endpoints = len(endpoint_indices) > 0
    has_branch_points = len(branch_point_indices) > 0

    inf_fill = np.inf
    nan_fill = np.nan
    junction_metrics_df['dist_nearest_junction'] = nan_fill
    junction_metrics_df['dist_nearest_endpoint'] = nan_fill
    junction_metrics_df['num_junction_neighbors'] = 0
    junction_metrics_df['num_endpoint_neighbors'] = 0

    if has_endpoints and has_branch_points:
        dist_matrix = cdist(positions, positions)

        dists_from_branches = dist_matrix[branch_point_mask, :]
        branch_to_branch = dists_from_branches[:, branch_point_mask]
        np.fill_diagonal(branch_to_branch, np.inf) 
        junction_metrics_df.loc[branch_point_mask, 'dist_nearest_junction'] = np.min(branch_to_branch, axis=1)
        junction_metrics_df.loc[branch_point_mask, 'num_junction_neighbors'] = np.sum(branch_to_branch <= distance_threshold, axis=1)
        
        branch_to_endpoint = dists_from_branches[:, endpoint_mask]
        junction_metrics_df.loc[branch_point_mask, 'dist_nearest_endpoint'] = np.min(branch_to_endpoint, axis=1)
        junction_metrics_df.loc[branch_point_mask, 'num_endpoint_neighbors'] = np.sum(branch_to_endpoint <= distance_threshold, axis=1)

        dists_from_endpoints = dist_matrix[endpoint_mask, :]
        endpoint_to_branch = dists_from_endpoints[:, branch_point_mask]
        junction_metrics_df.loc[endpoint_mask, 'dist_nearest_junction'] = np.min(endpoint_to_branch, axis=1)
        junction_metrics_df.loc[endpoint_mask, 'num_junction_neighbors'] = np.sum(endpoint_to_branch <= distance_threshold, axis=1)
        
        endpoint_to_endpoint = dists_from_endpoints[:, endpoint_mask]
        np.fill_diagonal(endpoint_to_endpoint, np.inf) 
        junction_metrics_df.loc[endpoint_mask, 'dist_nearest_endpoint'] = np.min(endpoint_to_endpoint, axis=1)
        junction_metrics_df.loc[endpoint_mask, 'num_endpoint_neighbors'] = np.sum(endpoint_to_endpoint <= distance_threshold, axis=1)

    elif has_branch_points: 
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf) 
        junction_metrics_df['dist_nearest_junction'] = np.min(dist_matrix, axis=1)
        junction_metrics_df['num_junction_neighbors'] = np.sum(dist_matrix <= distance_threshold, axis=1)

    elif has_endpoints: 
        dist_matrix = cdist(positions, positions)
        np.fill_diagonal(dist_matrix, np.inf) 
        junction_metrics_df['dist_nearest_endpoint'] = np.min(dist_matrix, axis=1)
        junction_metrics_df['num_endpoint_neighbors'] = np.sum(dist_matrix <= distance_threshold, axis=1)

    cols_to_check = ['dist_nearest_junction', 'dist_nearest_endpoint',
                     'num_junction_neighbors', 'num_endpoint_neighbors']
    for col in cols_to_check:
        if col not in junction_metrics_df.columns:
             fill_val = np.inf if 'dist' in col else 0
             junction_metrics_df[col] = fill_val

    return junction_metrics_df

def fractal_dimension_and_lacunarity(array, max_box_size=None, min_box_size=1, n_samples=20, n_offsets=0, plot=False):
    """
    Calculate fractal dimension and lacunarity of a binary array using box counting.

    Args:
        array (np.ndarray): Binary input array.
        max_box_size (int, optional): Max box size. Defaults to min dimension log2.
        min_box_size (int, optional): Min box size. Defaults to 1.
        n_samples (int, optional): Number of scales to sample. Defaults to 20.
        n_offsets (int, optional): Number of grid offsets to average over (for robustness). Defaults to 0.
        plot (bool, optional): Whether to plot the fit. Defaults to False.

    Returns:
        tuple: (fractal_dimension, lacunarity)
    """
    if max_box_size == None:
        max_box_size = int(np.floor(np.log2(np.min(array.shape))))
    scales = np.floor(np.logspace(max_box_size, min_box_size, num=n_samples, base=2))
    scales = np.unique(scales)
    locs = np.array(np.where(array > 0))
    Ns = []
    lacunarity_values = []
    for scale in scales:
        touched = []
        if n_offsets == 0:
            offsets = [0]
        else:
            offsets = np.linspace(0, scale, n_offsets)
        for offset in offsets:
            bin_edges = [np.arange(0-offset, i+offset, scale) for i in array.shape]
            H1, _ = np.histogramdd(locs.T, bins=bin_edges)
            touched.append(np.sum(H1>0))
        non_zero_hist = H1[H1 > 0]
        lacunarity_values.append(np.var(non_zero_hist) / (np.mean(non_zero_hist) ** 2))
        Ns.append(min(touched))
    Ns = np.array(Ns)
    unique_Ns, indices = np.unique(Ns, return_index=True)
    unique_Ns = unique_Ns[unique_Ns > 0]
    unique_scales = scales[indices[:len(unique_Ns)]]
    unique_lacunarity = np.array(lacunarity_values)[indices[:len(unique_Ns)]]
    coeffs = np.polyfit(np.log(1 / unique_scales), np.log(unique_Ns), 1)
    return coeffs[0], np.mean(unique_lacunarity)

def convex_hull_volume(binary_image):
    """Calculate the volume of the convex hull of foreground pixels."""
    points = np.argwhere(binary_image)
    if len(points) < 4: return 0
    hull = ConvexHull(points)
    return hull.volume
