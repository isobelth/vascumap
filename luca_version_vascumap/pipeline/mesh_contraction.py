import time
import math
import numpy as np
import networkx as nx
from tqdm.auto import tqdm
from scipy.sparse import spdiags, dia_matrix
from cupyx.scipy.sparse import csr_matrix, vstack
from cupyx.scipy.sparse.linalg import lsmr as lsmr_gpu
import cupy as cp
from skeletor.utilities import make_trimesh
from skeletor.pre.utils import (laplacian_cotangent, getMeshVPos, laplacian_umbrella,
                    averageFaceArea, getOneRingAreas)
from scipy.ndimage import map_coordinates
from .utils import cupy_chunk_processing
from scipy.ndimage import gaussian_filter

# New imports for create_graph_contraction
from skimage import measure
import trimesh
import skeletor as sk
from pygel3d import hmesh, graph as pygel_graph

def subdivide_graph_edges(G, min_segment_length=5.0):
    """
    Subdivide graph edges into smaller segments.

    Ensures no edge is longer than `min_segment_length` by introducing intermediate nodes.

    Args:
        G (nx.Graph): Input graph where nodes have 'pts' attribute (coordinates).
        min_segment_length (float, optional): Maximum length of an edge segment. Defaults to 5.0.

    Returns:
        nx.Graph: Graph with subdivided edges.
    """
    if not isinstance(min_segment_length, (int, float)) or min_segment_length <= 0:
        raise ValueError("min_segment_length must be a positive number.")

    G_segments = nx.Graph()
    added_nodes = set() 

    max_existing_node_id = -1
    if G.nodes:
        numeric_nodes = [nid for nid in G.nodes() if isinstance(nid, (int, np.integer))]
        if numeric_nodes:
            max_existing_node_id = max(numeric_nodes)

    new_node_counter = max_existing_node_id + 1
    original_edges = list(G.edges(data=True)) 

    for u, v, edge_data in original_edges:
        try:
            if u not in G.nodes or v not in G.nodes:
                print(f"Warning: Node {u} or {v} not found. Skipping edge ({u}, {v}).")
                continue
            pts_u = np.asarray(G.nodes[u]['pts'], dtype=float)
            pts_v = np.asarray(G.nodes[v]['pts'], dtype=float)
        except Exception as e:
            print(f"Warning: Error accessing 'pts' for edge ({u}, {v}): {e}. Skipping edge.")
            continue

        if u not in added_nodes:
            G_segments.add_node(u, **G.nodes[u])
            added_nodes.add(u)
        if v not in added_nodes:
            G_segments.add_node(v, **G.nodes[v])
            added_nodes.add(v)

        edge_length = np.linalg.norm(pts_v - pts_u)

        if edge_length > 0: 
            num_subdivisions = int(math.ceil(edge_length / float(min_segment_length)))
            num_subdivisions = max(1, num_subdivisions) 
        else:
            num_subdivisions = 1 

        if num_subdivisions == 1:
            G_segments.add_edge(u, v, **edge_data)
        else:
            last_node_id_in_chain = u
            for i in range(1, num_subdivisions): 
                t = i / float(num_subdivisions)
                intermediate_pts = pts_u * (1.0 - t) + pts_v * t

                intermediate_node_id = new_node_counter
                new_node_counter += 1

                G_segments.add_node(intermediate_node_id, pts=intermediate_pts)
                G_segments.add_edge(last_node_id_in_chain, intermediate_node_id)

                last_node_id_in_chain = intermediate_node_id

            G_segments.add_edge(last_node_id_in_chain, v)

    return G_segments

def contract_gpu(mesh, epsilon=1e-06, iter_lim=100, time_lim=None, precision=1e-07,
                 SL=2, WH0=1, WL0='auto', operator='cotangent', progress=True,
                 validate=True):
    """
    Contract mesh using GPU-accelerated Laplacian smoothing (CuPy).

    Iteratively contracts the mesh to approximate a skeleton.

    Args:
        mesh (trimesh.Trimesh): Input mesh.
        epsilon (float, optional): Convergence threshold for area change. Defaults to 1e-06.
        iter_lim (int, optional): Maximum iterations. Defaults to 100.
        time_lim (int, optional): Time limit in seconds. Defaults to None.
        precision (float, optional): Solver precision. Defaults to 1e-07.
        SL (float, optional): Factor to increase contraction weight (WL) each step. Defaults to 2.
        WH0 (float, optional): Initial attraction weight. Defaults to 1.
        WL0 (str or float, optional): Initial contraction weight. 'auto' uses mesh statistics. Defaults to 'auto'.
        operator (str, optional): Laplacian operator ('cotangent' or 'umbrella'). Defaults to 'cotangent'.
        progress (bool, optional): Show progress bar. Defaults to True.
        validate (bool, optional): Validate mesh before processing. Defaults to True.

    Returns:
        trimesh.Trimesh: Contracted mesh.
    """
    assert operator in ('cotangent', 'umbrella')
    start = time.time()

    m = make_trimesh(mesh, validate=validate)
    n = len(m.vertices)

    zeros_np = np.zeros((n, 3)) 
    WH0_diag = np.zeros(n)
    WH0_diag.fill(WH0)
    WH0_sp = spdiags(WH0_diag, 0, WH0_diag.size, WH0_diag.size)
    WH_sp = dia_matrix(WH0_sp)

    if WL0 == 'auto':
        WL0 = 1e-03 * np.sqrt(averageFaceArea(m))
    WL_diag = np.zeros(n)
    WL_diag.fill(WL0)
    WL_sp = spdiags(WL_diag, 0, WL_diag.size, WL_diag.size)

    dm = m.copy()

    area_ratios = [1.0]
    originalRingAreas = getOneRingAreas(dm)
    goodvertices = dm.vertices.copy() 
    bar_format = ("{l_bar}{bar}| [{elapsed}<{remaining}, "
                  "{postfix[0]}/{postfix[1]}it, "
                  "{rate_fmt}, epsilon {postfix[2]:.2g}")
    with tqdm(total=100,
              bar_format=bar_format,
              disable=progress is False,
              postfix=[1, iter_lim, 1]) as pbar:
        for i in range(iter_lim):
            if operator == 'cotangent':
                L_sp = laplacian_cotangent(dm, normalized=True)
            else:
                L_sp = laplacian_umbrella(dm)

            L_gpu = csr_matrix(L_sp)
            WH_gpu = csr_matrix(WH_sp)
            WL_gpu = csr_matrix(WL_sp)

            V_gpu = cp.asarray(getMeshVPos(dm))
            zeros_gpu = cp.asarray(zeros_np)

            A_gpu = vstack([WL_gpu.dot(L_gpu), WH_gpu], format='csr')
            b_gpu = cp.vstack((zeros_gpu, WH_gpu.dot(V_gpu)))

            cpts = np.zeros((n, 3)) 
            for j in range(3):
                x0_gpu = cp.asarray(dm.vertices[:, j])
                r0_gpu = b_gpu[:, j] - A_gpu * x0_gpu
                dx_gpu = lsmr_gpu(A_gpu, r0_gpu,
                                 atol=precision, btol=precision,
                                 damp=1)[0]
                cpts_gpu_j = x0_gpu + dx_gpu
                cpts[:, j] = cp.asnumpy(cpts_gpu_j)

            dm.vertices = cpts

            if progress:
                pbar.postfix[0] = i + 1

            new_eps = dm.area / m.area
            if (new_eps > area_ratios[-1]):
                dm.vertices = goodvertices 
                if progress:
                    tqdm.write("Total face area increased from last iteration."
                               f" Contraction stopped prematurely after {i} "
                               f"iterations at epsilon {area_ratios[-1]:.2g}.")
                break
            area_ratios.append(new_eps)

            if progress:
                pbar.postfix[2] = area_ratios[-1]
                prog = round((area_ratios[-2] - area_ratios[-1]) / (1 - epsilon) * 100) if (1 - epsilon) > 1e-9 else 100
                pbar.update(min(prog, 100-pbar.n))

            goodvertices = cpts.copy() 

            WL_sp = dia_matrix(WL_sp.multiply(SL))

            changeinarea_np = np.sqrt(originalRingAreas / getOneRingAreas(dm))
            changeinarea_np = np.nan_to_num(changeinarea_np, nan=1.0, posinf=1.0, neginf=1.0)
            WH_sp = dia_matrix(WH0_sp.multiply(changeinarea_np))

            if (area_ratios[-1] <= epsilon):
                break

            if not isinstance(time_lim, (bool, type(None))):
                if (time.time() - start) >= time_lim:
                    if progress:
                        tqdm.write(f"Time limit ({time_lim}s) reached. Stopping after {i+1} iterations.")
                    break

        dm.epsilon = area_ratios[-1]

        return dm

def create_graph_contraction(binary_edt, time_lim=5*60):
    """
    Create a graph from a binary volume using mesh contraction.

    Steps:
    1. Create mesh from binary mask (Marching Cubes).
    2. Simplify mesh.
    3. Contract mesh (GPU).
    4. Simplify contracted mesh.
    5. Extract skeleton (PyGEL3D).
    6. Convert to NetworkX graph.

    Args:
        binary_edt (np.ndarray): Input binary volume (or EDT).
        time_lim (int, optional): Time limit for contraction. Defaults to 300s.

    Returns:
        tuple: (G, mesh, simple_mesh, simple_cont)
            - G: The resulting NetworkX graph.
            - mesh: Original mesh.
            - simple_mesh: Simplified initial mesh.
            - simple_cont: Simplified contracted mesh.
    """
    
    verts, faces, normals, values = measure.marching_cubes(binary_edt, level=0)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=normals, validate=True)

    simple_mesh = trimesh.Trimesh.simplify_quadric_decimation(mesh, 0.995, aggression=8)

    contracted = contract_gpu(simple_mesh, 	
                    epsilon=1e-6,
                    iter_lim=100,
                    time_lim=time_lim, 
                    precision=1e-09, 
                    SL=4, 
                    WH0=2, 
                    WL0='auto',
                    operator='cotangent',
                    progress=True,
                    validate=True
                    )

    simple_cont = trimesh.Trimesh.simplify_quadric_decimation(contracted, 0.99, aggression=7)
    simple_cont = sk.pre.fix_mesh(simple_cont, remove_disconnected=5, inplace=False)

    m = hmesh.Manifold().from_triangles(simple_cont.vertices, simple_cont.faces)
    g = pygel_graph.from_mesh(m)
    skel = pygel_graph.LS_skeleton(g)

    G = nx.Graph()
    positions_array = skel.positions()
    node_ids = list(skel.nodes()) 

    for i, node_id in enumerate(node_ids):
        pos_zyx = positions_array[i]
        G.add_node(node_id, pts=tuple(pos_zyx))

    for node_id in G.nodes(): 
        try:
            # PyGEL3D neighbor iteration might vary by version
            neighbors = list(skel.neighbors(node_id)) 
        except TypeError:
            continue
        except Exception:
            continue
        for neighbor_id in neighbors:
            if G.has_node(neighbor_id):
                G.add_edge(node_id, neighbor_id)

    nodes_to_remove = set()
    high_degree_nodes = [node for node, degree in G.degree() if degree > 6]
    for node_id in high_degree_nodes:
        for neighbor_id in list(G.neighbors(node_id)):
            if G.degree(neighbor_id) == 1:
                nodes_to_remove.add(neighbor_id)
    G.remove_nodes_from(list(nodes_to_remove))

    return G, mesh, simple_mesh, simple_cont

def reposition_graph_edges(
    G, 
    binary_edt, 
    min_segment_length=5.0, 
    max_disp=12.5, 
    step_size=1.0, 
    num_iterations=1000, 
    tol=1e-3
    ):
    """
    Reposition graph nodes towards the centerline of the vessels (using EDT gradient).

    Args:
        G (nx.Graph): Input graph.
        binary_edt (np.ndarray): 3D EDT volume.
        min_segment_length (float, optional): Minimum segment length for subdivision before repositioning. Defaults to 5.0.
        max_disp (float, optional): Maximum displacement allowed. Defaults to 12.5.
        step_size (float, optional): Gradient ascent step size. Defaults to 1.0.
        num_iterations (int, optional): Gradient ascent iterations. Defaults to 1000.
        tol (float, optional): Gradient tolerance for stopping. Defaults to 1e-3.

    Returns:
        nx.Graph: Repositioned graph.
    """
    
    from cupyx.scipy import ndimage as cupy_ndimage

    chunk_size = (binary_edt.shape[0], 512, 512)

    edt_smoothed = cupy_chunk_processing(
        volume=binary_edt.astype(np.float32),
        processing_func=cupy_ndimage.gaussian_filter,
        sigma=5, 
        chunk_size=chunk_size
    )

    grad_x, grad_y, grad_z = np.gradient(edt_smoothed) 

    G_segments = subdivide_graph_edges(G, min_segment_length)
    G_repositioned = G_segments.copy()

    for u in tqdm(G_repositioned.nodes(), desc="Repositioning temporary nodes"):
        pts = np.array(G_repositioned.nodes[u]['pts'], dtype=np.float64)
        disp_total = 0.0
        
        for _ in range(num_iterations):
            pos = pts.reshape(3, 1)
            gx = map_coordinates(grad_x, pos, order=1, mode='nearest')[0]
            gy = map_coordinates(grad_y, pos, order=1, mode='nearest')[0]
            gz = map_coordinates(grad_z, pos, order=1, mode='nearest')[0]
            
            grad_vec = np.array([gx, gy, gz])
            grad_norm = np.linalg.norm(grad_vec)
            if grad_norm < tol:
                break
            
            delta = (step_size * grad_vec) / grad_norm
            
            step_disp = np.linalg.norm(delta)
            if disp_total + step_disp > max_disp:
                remaining = max_disp - disp_total
                if step_disp > 0:
                    delta = delta * (remaining / step_disp)
                pts = pts + delta
                break  
            
            pts = pts + delta
            disp_total += step_disp
        
        G_repositioned.nodes[u]['pts'] = pts

    return G_repositioned
