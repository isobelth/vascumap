import numpy as np
import dask.array as da
from skimage.morphology import skeletonize as skeletonize_3d
from skimage import measure
from scipy.spatial.distance import cdist
from skimage.draw import line_nd
import networkx as nx
from utils import cupy_chunk_processing

def skeletonize_3d_parallel(binary_volume, chunk_size=(128, 128, 128), iter=1):
    """
    Perform 3D skeletonization in parallel using Dask.

    Splits the volume into chunks with overlap, skeletonizes each chunk, and stitches them back.

    Args:
        binary_volume (np.ndarray): Input binary 3D volume.
        chunk_size (tuple, optional): Size of chunks. Defaults to (128, 128, 128).
        iter (int, optional): Number of iterations for skeletonization (usually 1 is enough for full skeletonization). Defaults to 1.

    Returns:
        np.ndarray: 3D skeletonized volume.
    """
    dask_volume = da.from_array(binary_volume, chunks=chunk_size)
    overlap = (2, 2, 2)
    
    def skeletonize_chunk(chunk, iterations):
        result = chunk.copy()
        for _ in range(iterations):
            result = skeletonize_3d(result)
        return result
    
    skeleton = da.overlap.map_overlap(
        dask_volume,
        skeletonize_chunk,
        depth=overlap,
        boundary='none',
        dtype=bool,
        iterations=iter 
    )
    
    return skeleton.compute(scheduler='threads')

def measure_edge_length(coordinates):
    """
    Calculate the total length of a path defined by coordinates.

    Args:
        coordinates (np.ndarray): Array of coordinates (N, 3).

    Returns:
        float: Total length of the path.
    """
    differences = np.diff(coordinates, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return np.sum(segment_lengths)

def prune_graph(graph, area_3d, edt_cutoff=0.25, length_cutoff=25):
    """
    Prune short branches from the graph based on length and EDT criteria.

    Iteratively removes endpoint branches if they are short or have low EDT values 
    (indicating they might be artifacts).

    Args:
        graph (nx.Graph): NetworkX graph of the skeleton.
        area_3d (np.ndarray): 3D volume containing cross-sectional areas (or EDT).
        edt_cutoff (float, optional): Threshold for EDT-based pruning. Defaults to 0.25.
        length_cutoff (int, optional): Minimum length for a branch to be kept. Defaults to 25.

    Returns:
        nx.Graph: Pruned graph.
    """
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
                value = np.mean(part_oi)/(neighbor_edt+1e-6) 

                if value > edt_cutoff or edge_length <= length_cutoff:
                    graph.remove_node(node)
                    values.append(value)

        if len(values) == 0:
            break
    return graph

def _remove_mid_node(G):
    """
    Remove degree-2 nodes and merge the incident edges.

    This simplifies the graph by merging segments that are split by a simple node.

    Args:
        G (nx.Graph): Input graph.

    Returns:
        nx.Graph: Simplified graph.
    """
    while True:
        nodes_to_process = [n for n, d in G.degree() if d == 2]

        if not nodes_to_process:
            break 

        processed_in_iteration = False
        for i in nodes_to_process:
            if not G.has_node(i) or G.degree(i) != 2:
                continue

            neighbors = list(G.neighbors(i))
            if len(neighbors) != 2: 
                continue 

            n1, n2 = neighbors[0], neighbors[1]

            if n1 == n2 or G.has_edge(n1, n2):
                continue 

            edge1 = G.get_edge_data(i, n1)
            edge2 = G.get_edge_data(i, n2)
            
            pts1 = np.atleast_2d(edge1['pts'])
            pts2 = np.atleast_2d(edge2['pts'])
            node_coord = G.nodes[i]['pts'].astype(np.int32) 

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
            elif min_row == 1 and min_col == 0: 
                 combined_line = np.concatenate([pts1, [node_coord], pts2], axis=0) 

            new_weight = edge1.get('weight', 0) + edge2.get('weight', 0)
            G.add_edge(n1, n2, weight=new_weight, pts=combined_line) 
            G.remove_node(i)
            processed_in_iteration = True

        if not processed_in_iteration:
            break
    return G

def finalize_graph(G_repositioned, binary_edt):
    """
    Finalize the graph by removing degree-2 nodes, handling self-loops, and ensuring edge coordinates are valid.

    Args:
        G_repositioned (nx.Graph): Graph after repositioning nodes.
        binary_edt (np.ndarray): 3D EDT volume (used for shape and clipping).

    Returns:
        nx.Graph: Final cleaned graph.
    """
    G_final = G_repositioned.copy()
    while True:
        degree2_nodes = [n for n, d in G_final.degree() if d == 2]
        if not degree2_nodes:
            break

        node_to_process = degree2_nodes[0]
        
        try:
            neighbors = list(G_final.neighbors(node_to_process))
            if len(neighbors) != 2: 
                continue

            u, v = neighbors
            coord_u = np.asarray(G_final.nodes[u]['pts'])
            coord_mid = np.asarray(G_final.nodes[node_to_process]['pts'])
            coord_v = np.asarray(G_final.nodes[v]['pts'])

            edge_data_um = G_final.edges[u, node_to_process]
            if 'pts' in edge_data_um and len(edge_data_um['pts']) > 1:
                path_um = list(map(tuple, edge_data_um['pts'])) 
            else:
                path_um = [tuple(coord_u), tuple(coord_mid)]

            edge_data_mv = G_final.edges[node_to_process, v]
            if 'pts' in edge_data_mv and len(edge_data_mv['pts']) > 1:
                path_mv = list(map(tuple, edge_data_mv['pts']))
            else:
                path_mv = [tuple(coord_mid), tuple(coord_v)]

            combined_pts = None
            if np.allclose(path_um[-1], coord_mid) and np.allclose(path_mv[0], coord_mid):
                combined_pts = path_um[:-1] + path_mv
            elif np.allclose(path_um[0], coord_mid) and np.allclose(path_mv[0], coord_mid):
                combined_pts = path_um[::-1][:-1] + path_mv 
            elif np.allclose(path_um[-1], coord_mid) and np.allclose(path_mv[-1], coord_mid):
                 combined_pts = path_um[:-1] + path_mv[::-1] 
            elif np.allclose(path_um[0], coord_mid) and np.allclose(path_mv[-1], coord_mid):
                 combined_pts = path_um[::-1][:-1] + path_mv[::-1] 
            else:
                 G_final.remove_node(node_to_process) 
                 continue 

            G_final.remove_node(node_to_process)
            if u != v: 
                 combined_pts_array = np.array(combined_pts, dtype=float)
                 new_edge_attributes = {'pts': combined_pts_array}
                 G_final.add_edge(u, v, **new_edge_attributes)

        except Exception as e:
            if G_final.has_node(node_to_process):
                 G_final.remove_node(node_to_process)
            continue 

    for u, v, data in G_final.edges(data=True):
        if 'pts' not in data or not isinstance(data['pts'], np.ndarray) or data['pts'].shape[0] < 2:
            try:
                 coord_u = np.asarray(G_final.nodes[u]['pts'])
                 coord_v = np.asarray(G_final.nodes[v]['pts'])
                 pts_array = np.array([coord_u, coord_v], dtype=float)
                 G_final.edges[u, v]['pts'] = pts_array
            except KeyError:
                 pass 
            except Exception as e:
                 pass

    image_shape = binary_edt.shape
    for u, v, data in G_final.edges(data=True):
        int_coords = data['pts'].astype(np.int32)
        int_coords[:, 0] = np.clip(int_coords[:, 0], 0, image_shape[0]-1)
        int_coords[:, 1] = np.clip(int_coords[:, 1], 0, image_shape[1]-1)
        int_coords[:, 2] = np.clip(int_coords[:, 2], 0, image_shape[2]-1)
        G_final.edges[u, v]['pts'] = int_coords
        
        u_coords = G_final.nodes[u]['pts'].astype(np.int32)
        v_coords = G_final.nodes[v]['pts'].astype(np.int32)
        u_coords = np.clip(u_coords, 0, np.array(image_shape)-1)
        v_coords = np.clip(v_coords, 0, np.array(image_shape)-1)
        G_final.nodes[u]['pts'] = u_coords
        G_final.nodes[v]['pts'] = v_coords

    for node in G_final.nodes():
        G_final.nodes[node]['sprout'] = G_final.degree(node) == 1

    edges_to_remove = []
    for u, v, data in G_final.edges(data=True):
        if G_final.degree(u) == 1 and G_final.degree(v) == 1:
            edges_to_remove.append((u, v))
    
    for u, v in edges_to_remove:
        G_final.remove_edge(u, v)

    for u, v, data in G_final.edges(data=True):
        pts = data['pts']
        line_coords_list = []
        for i in range(len(pts) - 1):
            p1 = pts[i]
            p2 = pts[i+1]
            line_coords_list.append(np.array(line_nd(p1, p2, endpoint=False)).T)
        line_coords_list.append(pts[-1][None, ...])
        G_final.edges[u, v]['pts'] = np.concatenate(line_coords_list, axis=0)

    return G_final

def collect_border_vicinity_edges(graph, image_shape, vicinity_z=1, vicinity_xy=50):
    """
    Remove edges that are too close to the border of the volume.

    Args:
        graph (nx.Graph): Input graph.
        image_shape (tuple): Shape of the volume (Z, Y, X).
        vicinity_z (int, optional): Z-margin. Defaults to 1.
        vicinity_xy (int, optional): XY-margin. Defaults to 50.

    Returns:
        nx.Graph: Graph with border edges removed.
    """
    border_vicinity_edges = set()
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if any((pt[0] < vicinity_z or pt[0] > image_shape[0] - 1 - vicinity_z or
                    pt[1] < vicinity_xy or pt[1] > image_shape[1] - 1 - vicinity_xy or
                    pt[2] < vicinity_xy or pt[2] > image_shape[2] - 1 - vicinity_xy) for pt in pts):
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
