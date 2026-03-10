import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import tifffile as tif
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_fill_holes, distance_transform_edt
import sknw

from .utils import cupy_chunk_processing
from .skeletonization import (skeletonize_3d_parallel, 
                             prune_graph, _remove_mid_node, finalize_graph,
                             collect_border_vicinity_edges)
from .mesh_contraction import create_graph_contraction, reposition_graph_edges, contract_gpu
from .metrics import (compute_cross_sectional_areas, convex_hull_volume, 
                     compute_vessel_metrics, 
                     compute_junction_metrics, fractal_dimension_and_lacunarity)
from .visualization import visualize_vessel_network, graph2image

def construct_vessel_network(binary_image, global_metrics_df, skel_method="auto", contraction_timelim=5*60):
    """
    Construct a graph representation of the vessel network from a binary mask.

    Uses either skeletonization (thinning) or mesh contraction. Calculates global metrics
    like volume and coverage.

    Args:
        binary_image (np.ndarray): Binary vessel mask.
        global_metrics_df (pd.DataFrame): DataFrame to store global metrics.
        skel_method (str, optional): Method ('thinning', 'contraction', 'auto'). Defaults to "auto".
        contraction_timelim (int, optional): Time limit for contraction in seconds. Defaults to 300.

    Returns:
        tuple:
            - nx.Graph: The constructed vessel graph.
            - np.ndarray: 3D volume of cross-sectional areas at skeleton points.
    """

    global_metrics_df.loc[0, 'explant_volume'] = convex_hull_volume(binary_image) if np.any(binary_image) else 0
    global_metrics_df.loc[0, 'image_volume'] = np.prod(binary_image.shape)
    global_metrics_df.loc[0, 'total_vessel_volume'] = np.sum(binary_image)
    explant_vol = global_metrics_df.loc[0, 'explant_volume']
    vessel_vol = global_metrics_df.loc[0, 'total_vessel_volume']
    vessel_covergae = vessel_vol / explant_vol if explant_vol > 0 else 0

    global_metrics_df.loc[0, 'vessel_coverage_explant'] = vessel_covergae if explant_vol > 0 else 0
    global_metrics_df.loc[0, 'vessel_coverage_image'] = vessel_vol / global_metrics_df.loc[0, 'image_volume'] if global_metrics_df.loc[0, 'image_volume'] > 0 else 0

    if skel_method == "auto":
        if vessel_covergae > 0.3:
            skel_method = "thinning"
        else:
            skel_method = "contraction"
        print(f"Vessel coverage: {vessel_covergae}. Using {skel_method} method for skeletonization")

    global_metrics_df.loc[0, 'skeletonization_method'] = skel_method

    chunk_size = (binary_image.shape[0], 512, 512)

    binary_smoothed = cupy_chunk_processing(
        volume=binary_image.astype(np.float32),
        processing_func=gaussian_filter,
        sigma=3, 
        chunk_size=chunk_size 
        ) > 0.5

    binary_filled_holes = cupy_chunk_processing(
        volume=binary_smoothed, 
        processing_func=binary_fill_holes, 
        chunk_size=chunk_size
        ).astype(np.float32)

    binary_edt = cupy_chunk_processing(
        volume=binary_filled_holes, 
        processing_func=distance_transform_edt, 
        chunk_size=chunk_size
        )

    if skel_method == "thinning":
        skeleton_image = skeletonize_3d_parallel(
            binary_filled_holes,
            chunk_size=chunk_size,
        )

        area_image = compute_cross_sectional_areas(binary_image, skeleton_image, binary_edt)

        graph = sknw.build_sknw(skeleton_image)

        for n in list(graph.nodes()):
            graph.nodes[n]['pts'] = graph.nodes[n]['pts'][0]
            graph.nodes[n]['sprout'] = graph.degree(n) == 1

        pruned_graph = prune_graph(
            graph=graph, 
            area_3d=area_image, 
            edt_cutoff=0.20, 
            length_cutoff=25
           )
        
        G_final = _remove_mid_node(pruned_graph)

    elif skel_method == "contraction":
         from .mesh_contraction import create_graph_contraction # Local import
         G, *_ = create_graph_contraction(binary_edt, time_lim=contraction_timelim)

         G_repositioned = reposition_graph_edges(
            G, 
            binary_edt, 
            min_segment_length=10, 
            max_disp=10, 
            num_iterations=1000, 
            tol=1e-20,
            step_size=1,
            )

         G_final = finalize_graph(G_repositioned, binary_edt)
         skeleton_image = graph2image(G_final, binary_image.shape)
         area_image = compute_cross_sectional_areas(binary_image, skeleton_image, binary_edt)

    trimmed_graph = collect_border_vicinity_edges(G_final, binary_image.shape)
    isolated_nodes = [node for node in trimmed_graph.nodes() if trimmed_graph.degree[node] == 0]

    if isolated_nodes:
        trimmed_graph.remove_nodes_from(isolated_nodes)

    return trimmed_graph, area_image

def calculate_graph_metrics(graph, sample_name, area_image, global_metrics_df):
    """
    Calculate metrics from the vessel graph and area image.

    Aggregates vessel and junction metrics into global statistics (mean, std, median).

    Args:
        graph (nx.Graph): Vessel graph.
        sample_name (str): Name of the sample.
        area_image (np.ndarray): 3D area volume.
        global_metrics_df (pd.DataFrame): DataFrame to append global metrics to.

    Returns:
        tuple:
            - global_metrics_df (pd.DataFrame): Updated global metrics.
            - junction_metrics_df (pd.DataFrame): Detailed junction metrics.
            - vessel_metrics_df (pd.DataFrame): Detailed vessel metrics.
    """
    junction_metrics_df = pd.DataFrame()
    vessel_metrics_df = pd.DataFrame()

    if graph.number_of_edges() > 0:
        fd, lacunarity = fractal_dimension_and_lacunarity(area_image > 0)
        total_vessel_length = np.sum([np.linalg.norm(np.diff(graph[u][v]['pts'], axis=0), axis=1).sum() for u, v in graph.edges()])
        branchpoints_count = sum(1 for u in graph.nodes() if not graph.nodes[u]['sprout']) 
        vessels_count = sum(1 for u, v in graph.edges() if not graph.nodes[u]['sprout'] and not graph.nodes[v]['sprout'])
        sprouts_count = graph.number_of_edges() - vessels_count 

    else: 
        fd, lacunarity = np.nan, np.nan
        total_vessel_length = 0
        branchpoints_count = 0
        vessels_count = 0 
        sprouts_count = 0

    global_metrics_df.loc[0, 'total_vessel_length'] = total_vessel_length
    global_metrics_df.loc[0, 'total_number_of_sprouts'] = sprouts_count 
    global_metrics_df.loc[0, 'total_number_of_branches'] = vessels_count 
    global_metrics_df.loc[0, 'total_number_of_junctions'] = branchpoints_count 
    global_metrics_df.loc[0, 'fractal_dimension'] = fd
    global_metrics_df.loc[0, 'lacunarity'] = lacunarity

    if graph.number_of_nodes() > 0 and graph.number_of_edges() > 0:
        vessel_metrics_df = compute_vessel_metrics(graph, area_image, vessel_metrics_df)
        junction_metrics_df = compute_junction_metrics(graph, junction_metrics_df, distance_threshold=250) 

        if not vessel_metrics_df.empty:
            is_sprout_mask = vessel_metrics_df['is_sprout']
            for key, value in vessel_metrics_df.items():
                 if key not in ['z', 'y', 'x', 'is_sprout', 'mean_cs_area', 'median_cs_area', 'std_cs_area', 'node1_degree', 'node2_degree', 'orientation_z', 'orientation_y', 'orientation_x', 'sample']: 
                    if pd.api.types.is_numeric_dtype(value): 
                        global_metrics_df.loc[0, 'mean_branch_' + key] = np.nanmean(value[~is_sprout_mask])
                        global_metrics_df.loc[0, 'mean_sprout_' + key] = np.nanmean(value[is_sprout_mask])
                        global_metrics_df.loc[0, 'mean_sprout_and_branch_' + key] =  np.nanmean(value)

                        global_metrics_df.loc[0, 'std_branch_' + key] = np.nanstd(value[~is_sprout_mask])
                        global_metrics_df.loc[0, 'std_sprout_' + key] = np.nanstd(value[is_sprout_mask])
                        global_metrics_df.loc[0, 'std_sprout_and_branch_' + key] =  np.nanstd(value)

                        global_metrics_df.loc[0, 'median_branch_' + key] = np.nanmedian(value[~is_sprout_mask])
                        global_metrics_df.loc[0, 'median_sprout_' + key] = np.nanmedian(value[is_sprout_mask])
                        global_metrics_df.loc[0, 'median_sprout_and_branch_' + key] =  np.nanmedian(value)

        if not junction_metrics_df.empty:
            is_branch_point_mask = junction_metrics_df['is_junction']
            is_sprout_tip_mask = junction_metrics_df['is_sprout_tip']
            for key, value in junction_metrics_df.items():
                if key not in ['z', 'y', 'x', 'is_sprout_tip', 'is_junction', 'sample']: 
                     if pd.api.types.is_numeric_dtype(value): 
                        global_metrics_df.loc[0, 'mean_junction_' + key] = np.nanmean(value[is_branch_point_mask])
                        global_metrics_df.loc[0, 'mean_sprout_tip_' + key] = np.nanmean(value[is_sprout_tip_mask])
                        global_metrics_df.loc[0, 'mean_junction_and_sprout_tip_' + key] = np.nanmean(value) 

                        global_metrics_df.loc[0, 'std_junction_' + key] = np.nanstd(value[is_branch_point_mask])
                        global_metrics_df.loc[0, 'std_sprout_tip_' + key] = np.nanstd(value[is_sprout_tip_mask])
                        global_metrics_df.loc[0, 'std_junction_and_sprout_tip_' + key] = np.nanstd(value)

                        global_metrics_df.loc[0, 'median_junction_' + key] = np.nanmedian(value[is_branch_point_mask])
                        global_metrics_df.loc[0, 'median_sprout_tip_' + key] = np.nanmedian(value[is_sprout_tip_mask])
                        global_metrics_df.loc[0, 'median_junction_and_sprout_tip_' + key] = np.nanmedian(value)

    if not junction_metrics_df.empty:
        junction_metrics_df['sample'] = sample_name
    if not vessel_metrics_df.empty:
        vessel_metrics_df['sample'] = sample_name

    return global_metrics_df, junction_metrics_df, vessel_metrics_df

def process_directory(masks_dir_path, suffix, save_local_metrics, visualization=True, skel_method="thinning", contraction_timelim=5*60):
    """
    Process all masks in a directory.

    Args:
        masks_dir_path (str): Directory containing mask files.
        suffix (str): Suffix of mask files (e.g., '_mask').
        save_local_metrics (bool): Whether to save per-vessel/junction metrics.
        visualization (bool, optional): Whether to save visualization images. Defaults to True.
        skel_method (str, optional): Skeletonization method. Defaults to "thinning".
        contraction_timelim (int, optional): Time limit for contraction. Defaults to 300.
    """

    masks_dir = Path(masks_dir_path)
    masks_paths = list(masks_dir.glob(f'*{suffix}.tif'))
    global_csv_path = masks_dir / f'global_metrics_{skel_method}.csv'

    cumulative_global_metrics_df = pd.DataFrame()

    for mask_path in tqdm(masks_paths, desc='Processing masks'):
        sample_name = mask_path.stem
        try:
            binary_image = tif.imread(str(mask_path))
            
            if np.sum(binary_image) > 0:
                
                global_metrics_df = pd.DataFrame({'sample': [sample_name]})

                graph, area_image = construct_vessel_network(
                    binary_image, 
                    global_metrics_df, 
                    skel_method=skel_method, 
                    contraction_timelim=contraction_timelim
                    )

                global_metrics_df, junction_metrics_df, vessel_metrics_df = calculate_graph_metrics(
                    graph, sample_name, area_image, global_metrics_df)
                
                if save_local_metrics:
                    junction_csv_path = masks_dir / f'{sample_name}_junction_metrics_{skel_method}.h5'
                    vessel_csv_path = masks_dir / f'{sample_name}_vessel_metrics_{skel_method}.h5'

                    junction_metrics_df.to_hdf(junction_csv_path, index=False, key='df', mode='w')
                    vessel_metrics_df.to_hdf(vessel_csv_path, index=False, key='df', mode='w')
                
                if visualization:
                    visualize_vessel_network(binary_image, graph, area_image, sample_name=sample_name)
                    plt.savefig(masks_dir / f'{sample_name}_graph_{skel_method}.png', dpi=300)
                    plt.close('all')
            else: 
                global_metrics_df = pd.DataFrame({'sample' : [sample_name]})
            
            cumulative_global_metrics_df = pd.concat([cumulative_global_metrics_df, global_metrics_df], ignore_index=True)
            cumulative_global_metrics_df.to_csv(global_csv_path, index=False) 

        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute graph metrics on the vessel segmentation masks.")

    parser.add_argument("--masks_dir_path", type=str, required=True,
                        help="Path to the directory containing masks.")
    
    parser.add_argument("--file_suffix", type=str, default='_mask', # Changed type to str
                        help="Suffix of the mask files.")
    
    parser.add_argument("--save_local_metrics", action='store_true',
                        help="Save vessel and junction metrics for each mask, only save the global metrics otherwise.")

    parser.add_argument("--graph_visualization", action='store_true',
                        help="Save graph visualization.")
    
    parser.add_argument("--method", type=str, default="thinning",
                        help="Skeletonization method: thinning or contraction")

    args = parser.parse_args()

    process_directory(
        masks_dir_path=args.masks_dir_path,
        suffix=args.file_suffix,
        save_local_metrics=args.save_local_metrics, 
        visualization=args.graph_visualization,
        skel_method=args.method
        )
