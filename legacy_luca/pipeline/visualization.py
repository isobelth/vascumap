import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from tqdm.auto import tqdm
from .utils import contrast

def visualize_vessel_network(binary_image, graph, area_image, sample_name="Vessel Network"):
    """
    Visualize the vessel network (graph edges and nodes) over a 2D projection of the original image.

    Edges are colored by estimated vessel diameter.
    Nodes are colored by type (sprout vs junction).

    Args:
        binary_image (np.ndarray): Binary vessel mask (used for background projection).
        graph (nx.Graph): Vessel graph.
        area_image (np.ndarray): 3D area volume (used to look up diameters).
        sample_name (str, optional): Title for the plot. Defaults to "Vessel Network".
    """
    
    cmap = cm.magma
    edge_areas = []
    for u, v in graph.edges():
        try:
            pts = graph[u][v]['pts']
            if len(pts) < 1: continue

            pts = pts.astype(np.uint32)
            valid_edge_areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            edge_areas.append(np.median(valid_edge_areas))
        except (KeyError, IndexError):
            continue

    if not edge_areas:
        min_diam, max_diam = 0, 1
    else:
        min_diam = np.percentile(edge_areas, 0) * 0.5
        max_diam = np.percentile(edge_areas, 90)

    norm = Normalize(vmin=min_diam, vmax=max_diam)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)

    mask_2d = contrast(np.sum(binary_image, axis=0), 0, 75)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(15, 20)) 

    ax.imshow(mask_2d, cmap='gray', alpha=0.35)

    for u, v in tqdm(graph.edges(), desc="Visualizing edges"):
        try:
            pts = graph[u][v]['pts']
            if len(pts) < 1: continue 

            pts = pts.astype(np.uint32)
            valid_edge_areas = area_image[pts[:, 0], pts[:, 1], pts[:, 2]]
            
            if len(valid_edge_areas) > 0:
                avg_diameter = np.median(valid_edge_areas)
            else:
                avg_diameter = 0

            y_coords = pts[:, 1]
            x_coords = pts[:, 2]

            ax.plot(x_coords, y_coords, color=scalar_mappable.to_rgba(avg_diameter), linewidth=1.5)

        except (KeyError, IndexError):
             continue

    node_positions_yx = []
    node_colors = []
    node_sizes = []
    node_degrees = dict(graph.degree()) 

    for node in graph.nodes():
        pos_zyx = graph.nodes[node]['pts']
        node_positions_yx.append([pos_zyx[2], pos_zyx[1]]) 

        degree = node_degrees.get(node, 0)
        node_data = graph.nodes[node] 
        
        if degree == 1:
            if node_data.get('sprout', False): 
                node_colors.append('lime') 
                node_sizes.append(15)      
            else:
                node_colors.append('orange') 
                node_sizes.append(15)       
        else :
            node_colors.append('white') 
            node_sizes.append(15)      

    if node_positions_yx: 
        node_positions_yx = np.array(node_positions_yx)
        ax.scatter(node_positions_yx[:, 0], node_positions_yx[:, 1], 
                    c=node_colors, 
                    s=node_sizes, 
                    alpha=0.5,
                    zorder=5) 

    ax.set_aspect('equal', adjustable='box') 
    plt.axis('off')
    plt.title(sample_name, fontsize=20) 
    plt.tight_layout()

def graph2image(graph, shape):
    """
    Convert a graph back into a binary skeleton image.

    Args:
        graph (nx.Graph): Input graph.
        shape (tuple): Shape of the output image.

    Returns:
        np.ndarray: Binary skeleton image.
    """

    pruned_skeleton_image = np.zeros(shape)

    for u, v in graph.edges():
        coords = graph.get_edge_data(u, v)['pts']

        clipped_coords = np.zeros_like(coords)
        clipped_coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
        clipped_coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)
        clipped_coords[:, 2] = np.clip(coords[:, 2], 0, shape[2] - 1)

        pruned_skeleton_image[clipped_coords[:, 0], clipped_coords[:, 1], clipped_coords[:, 2]] = 1
    
    return pruned_skeleton_image
