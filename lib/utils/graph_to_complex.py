import sys, os

# working directory must be in folder test to execute below lines
ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join('..', '..', 'utils')))
sys.path.append(ROOT_DIR)

import graph_tool as gt
import graph_tool.topology as top
import numpy as np
import torch
import gudhi as gd
import itertools
import networkx as nx

from tqdm import tqdm
from lib.data.cochain import Cochain
from lib.data.complex import Complex
from typing import List, Dict, Optional, Union
from torch import Tensor
from torch_geometric.typing import Adj
from torch_scatter import scatter
from lib.utils.parallel import ProgressParallel
from joblib import delayed
from torch_geometric.data import Data


def pyg_to_simplex_tree(edge_index: Tensor, size: int):
    """
        Constructs a simplex tree from a PyG graph.

        Args:
            edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
            size: The number of nodes in the graph.
    """
    st = gd.SimplexTree()
    # Add vertices to the simplex.
    for v in range(size):
        st.insert([v])

    # Add the edges to the simplex.
    edges = edge_index.numpy()
    for e in range(edges.shape[1]):
        edge = [edges[0][e], edges[1][e]]
        st.insert(edge)

    return st


def get_simplex_boundaries(simplex):
    """
        Construct simplex boundary from simplex.

        Args:
            simplex: constructed from simplex_tree
    """
    boundaries = itertools.combinations(simplex, len(simplex) - 1)                  # in simplicial complex, get faces by n C (n-1)
    return [tuple(boundary) for boundary in boundaries]


def build_tables(simplex_tree, size):
    """
        Construct simplex tables and simplex-id mapping table from simplex_tree given total number of nodes.

        Args:
            simplex_tree: simplex tree from GUDHI
            size: total number of nodes

        Outputs:
            simplex_tables: list with length n-dimension containing lists of simplices
            id_maps: list with length n-dimension containing dictionary (simplex -> id). Id starts from 0 for every dim.
    """
    complex_dim = simplex_tree.dimension()
    # Each of these data structures has a separate entry per dimension.
    id_maps = [{} for _ in range(complex_dim+1)]                                    # list with length n-dimension containing dictionary (simplex -> id) 
    simplex_tables = [[] for _ in range(complex_dim+1)] # matrix of simplices       # list with length n-dimension containing lists of simplices

    simplex_tables[0] = [[v] for v in range(size)]                                  # first index is list of vertices
    id_maps[0] = {tuple([v]): v for v in range(size)}                               # first index is dictionary where key is tuple of a single vertex, value if the index of that vertex

    for simplex, _ in simplex_tree.get_simplices():
        dim = len(simplex) - 1
        if dim == 0:
            continue

        # Assign this simplex the next unused ID
        next_id = len(simplex_tables[dim])                                          # id starts from 0 for every dim
        id_maps[dim][tuple(simplex)] = next_id
        simplex_tables[dim].append(simplex)

    return simplex_tables, id_maps


def extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim: int):
    """
        Build two maps simplex -> its coboundaries and simplex -> its boundaries

        Args:
            simplex_tree: simplex tree from GUDHI
            id_maps: list with length n-dimension containing dictionary (simplex -> id). Id starts from 0 for every dim.
            complex_dim: dimension of the complex

        Outputs:
            boundaries_tables: list with length n-dimension containing list of boundary indices (use list because there may be duplicate boundary indices)
            boundaries: list with length (n+1)-dimension containing dict{simplex -> boundaries}
            coboundaries: list with length (n+1)-dimension containing dict{simplex -> coboundaries}
    """
    # The extra dimension is added just for convenience to avoid treating it as a special case.
    boundaries = [{} for _ in range(complex_dim+2)]                                 # list with length (n+1)-dimension containing dict{simplex -> boundaries}
    coboundaries = [{} for _ in range(complex_dim+2)]                               # list with length (n+1)-dimension containing dict{simplex -> coboundaries}
    boundaries_tables = [[] for _ in range(complex_dim+1)]                          # list with length n-dimension containing list of boundary indices (use list because there may be duplicate boundary indices)

    for simplex, _ in simplex_tree.get_simplices():
        # Extract the relevant boundary and coboundary maps
        simplex_dim = len(simplex) - 1
        level_coboundaries = coboundaries[simplex_dim]
        level_boundaries = boundaries[simplex_dim + 1]

        # Add the boundaries of the simplex to the boundaries table
        if simplex_dim > 0:     # id_maps match a simplex to its id
            boundaries_ids = [id_maps[simplex_dim-1][boundary] for boundary in get_simplex_boundaries(simplex)]
            boundaries_tables[simplex_dim].append(boundaries_ids)

        # This operation should be roughly be O(dim_complex*#-of-top-simplex), so that is very efficient for us (number of 2-simplices is not much)
        # For details see pages 6-7 https://hal.inria.fr/hal-00707901v1/document
        simplex_coboundaries = simplex_tree.get_cofaces(simplex, codimension=1)
        for coboundary, _ in simplex_coboundaries:
            assert len(coboundary) == len(simplex) + 1

            if tuple(simplex) not in level_coboundaries:
                level_coboundaries[tuple(simplex)] = list()
            level_coboundaries[tuple(simplex)].append(tuple(coboundary))

            if tuple(coboundary) not in level_boundaries:
                level_boundaries[tuple(coboundary)] = list()
            level_boundaries[tuple(coboundary)].append(tuple(simplex))

    return boundaries_tables, boundaries, coboundaries


def build_adj(boundaries: List[Dict], coboundaries: List[Dict], id_maps: List[Dict], complex_dim: int,
              include_down_adj: bool):
    """
    Builds the upper and lower adjacency data structures of the complex

    Args:
        boundaries: A list of dictionaries of the form
            boundaries[dim][simplex] -> List[simplex] (the boundaries)
        coboundaries: A list of dictionaries of the form
            coboundaries[dim][simplex] -> List[simplex] (the coboundaries)
        id_maps: A dictionary from simplex -> simplex_id
        complex_dim: dimension of complex
        include_down_adj: whether to include lower adjacent neighbours
    
    Outputs:
        all_shared_boundaries: a list of length n-dimension, each of element is a list whose index specifying the indices of the shared boundary for each lower adjacency
        all_shared_coboundaries: a list of length n-dimension, each of element is a list whose index specifying the indices of the shared coboundary for each upper adjacency
        lower_indexes: a list of length n-dimension, each of element has shape (num_lower_connections, 2)
        upper_indexes: a list of length n-dimension, each of element has shape (num_upper connection, 2)   
    """
    def initialise_structure():                     # create 2-d list with length of complex dimension
        return [[] for _ in range(complex_dim+1)]

    upper_indexes, lower_indexes = initialise_structure(), initialise_structure()
    all_shared_boundaries, all_shared_coboundaries = initialise_structure(), initialise_structure()

    # Go through all dimensions of the complex
    for dim in range(complex_dim+1):
        # Go through all the simplices at that dimension
        for simplex, id in id_maps[dim].items():
            # Add the upper adjacent neighbours from the level below
            if dim > 0:
                for boundary1, boundary2 in itertools.combinations(boundaries[dim][simplex], 2):        # get two boundaries of the current simplex
                    id1, id2 = id_maps[dim - 1][boundary1], id_maps[dim - 1][boundary2]                 # get id of boundaries of a simplex
                    upper_indexes[dim - 1].extend([[id1, id2], [id2, id1]])                             # equivalent with append twice
                    all_shared_coboundaries[dim - 1].extend([id, id])                                   # record which simplices that two boundaries belong to

            # Add the lower adjacent neighbours from the level above
            if include_down_adj and dim < complex_dim and simplex in coboundaries[dim]:                 # make sure simplex is in coboundaries
                for coboundary1, coboundary2 in itertools.combinations(coboundaries[dim][simplex], 2):  # get two co-boundaries of the current simplex
                    id1, id2 = id_maps[dim + 1][coboundary1], id_maps[dim + 1][coboundary2]             # get id of coboundaries of a simplex
                    lower_indexes[dim + 1].extend([[id1, id2], [id2, id1]])                             # equivalent with append twice
                    all_shared_boundaries[dim + 1].extend([id, id])                                     # record which simplices that two coboundaries belong to

    return all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes


def construct_features(vx: Tensor, cell_tables, init_method: str) -> List:
    """
    Combines the features of the component vertices to initialise the cell features

    Args:
        vx: a Tensor storing feature of a vertex
        cell_tables: list with length n-dimension containing lists of simplices
        init_method: either sum or mean (default is sum)
    
    Outputs:
        features: list of features at every dim
    """
    features = [vx]
    for dim in range(1, len(cell_tables)):                                                              # Go through every dim >= 1
        aux_1 = []                                                                                      # storing index of every vertex in cell
        aux_0 = []                                                                                      # storing cell
        for c, cell in enumerate(cell_tables[dim]):
            aux_1 += [c for _ in range(len(cell))]                                                      # extend cell index
            aux_0 += cell                                                                               # extend vertices belong to cell
        node_cell_index = torch.LongTensor([aux_0, aux_1])                                              # shape (2, #-of-cells * (dim+1))
        in_features = vx.index_select(0, node_cell_index[0])                                            # vx has shape (#nodes, features), so select features of necessary indices only
        features.append(scatter(in_features, node_cell_index[1], dim=0,                                 # feature of a simplex/cell is summation of feature from nodes
                                dim_size=len(cell_tables[dim]), reduce=init_method))

    return features


def extract_labels(y, size):
    """
    Extract labels.

    Args:
        y: label of the graph/labels of vertices
        size: number of nodes
    
    Outputs:
        v_y: labels of vertices. None for graph classification/regression tasks.
        complex_y: label of complex. Same with graph label. None for node classification/regression tasks.
    """
    v_y, complex_y = None, None
    if y is None:
        return v_y, complex_y

    y_shape = list(y.size())

    if y_shape[0] == 1:
        # This is a label for the whole graph (for graph classification).
        # We will use it for the complex.
        complex_y = y
    else:
        # This is a label for the vertices of the complex.
        assert y_shape[0] == size
        v_y = y

    return v_y, complex_y


def generate_cochain(dim, x, all_upper_index, all_lower_index,
                   all_shared_boundaries, all_shared_coboundaries, cell_tables, boundaries_tables,
                   complex_dim, y=None):
    """
    Builds a Cochain given all the adjacency data extracted from the complex.
    
    Args:
        dim: simplex dimension
        x: feature of simplex
        all_upper_index: upper adjacent neighbors (get from build_adj())
        all_lower_index: lower adjacent neighbors (get from build_adj())
        all_shared_boundaries: a list of length n-dimension, each of element is a list whose index specifying 
                                the indices of the shared boundary for each lower adjacency (get from build_adj())
        all_shared_coboundaries: a list of length n-dimension, each of element is a list whose index specifying 
                                the indices of the shared coboundary for each upper adjacency (get from build_adj())
        cell_tables: tables of cells, for every dim storing simplices (get from build_tables())
        boundaries_tables: tables of boundaries, for every dim storing index of boundaries (get from extract_boundaries_and_coboundaries_from_simplex_tree())
        complex_dim: complex dimension
        y: label (default is None)

    Outputs:
        Cochain()
    """
    if dim == 0:
        assert len(all_lower_index[dim]) == 0
        assert len(all_shared_boundaries[dim]) == 0

    num_cells_down = len(cell_tables[dim-1]) if dim > 0 else None
    num_cells_up = len(cell_tables[dim+1]) if dim < complex_dim else 0

    up_index = (torch.tensor(all_upper_index[dim], dtype=torch.long).t()
                if len(all_upper_index[dim]) > 0 else None)
    down_index = (torch.tensor(all_lower_index[dim], dtype=torch.long).t()
                  if len(all_lower_index[dim]) > 0 else None)
    shared_coboundaries = (torch.tensor(all_shared_coboundaries[dim], dtype=torch.long)
                      if len(all_shared_coboundaries[dim]) > 0 else None)
    shared_boundaries = (torch.tensor(all_shared_boundaries[dim], dtype=torch.long)
                    if len(all_shared_boundaries[dim]) > 0 else None)
    
    # Construct boundary index from boundaries_tables
    boundary_index = None
    if len(boundaries_tables[dim]) > 0:
        boundary_index = [list(), list()]
        for s, cell in enumerate(boundaries_tables[dim]):
            for boundary in cell:
                boundary_index[1].append(s)
                boundary_index[0].append(boundary)
        boundary_index = torch.LongTensor(boundary_index)
        
    if num_cells_down is None:
        assert shared_boundaries is None
    if num_cells_up == 0:
        assert shared_coboundaries is None

    if up_index is not None:
        assert up_index.size(1) == shared_coboundaries.size(0)
        assert num_cells_up == shared_coboundaries.max() + 1
    if down_index is not None:
        assert down_index.size(1) == shared_boundaries.size(0)
        assert num_cells_down >= shared_boundaries.max() + 1

    return Cochain(dim=dim, x=x, upper_index=up_index,
                 lower_index=down_index, shared_coboundaries=shared_coboundaries,
                 shared_boundaries=shared_boundaries, y=y, num_cells_down=num_cells_down,
                 num_cells_up=num_cells_up, boundary_index=boundary_index)


def compute_clique_complex_with_gudhi(x: Tensor, edge_index: Adj, size: int,
                                      expansion_dim: int = 2, y: Tensor = None,
                                      include_down_adj=True,
                                      init_method: str = 'sum') -> Complex:
    """Generates a clique complex of a pyG graph via gudhi.

    Args:
        x: The feature matrix for the nodes of the graph
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        size: The number of nodes in the graph
        expansion_dim: The dimension to expand the simplex to.
        y: Labels for the graph nodes or a label for the whole graph.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, Tensor)  # Support only tensor edge_index for now

    # Creates the gudhi-based simplicial complex
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    simplex_tree.expansion(expansion_dim)  # Computes the clique complex up to the desired dim.
    complex_dim = simplex_tree.dimension()  # See what is the dimension of the complex now.

    # Builds tables of the simplicial complexes at each level and their IDs
    simplex_tables, id_maps = build_tables(simplex_tree, size)

    # Extracts the boundaries and coboundaries of each simplex in the complex
    boundaries_tables, boundaries, co_boundaries = (
        extract_boundaries_and_coboundaries_from_simplex_tree(simplex_tree, id_maps, complex_dim))

    # Computes the adjacencies between all the simplexes in the complex
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    # TODO: Make this handle edge features as well and add alternative options to compute this.
    xs = construct_features(x, simplex_tables, init_method)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim+1):
        y = v_y if i == 0 else None                 # TODO: this code will assign vertex label to k-cochain for k>= 1. Consider change it if working with node task.
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               simplex_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_gudhi(dataset, expansion_dim: int, include_down_adj=True,
                                     init_method: str = 'sum'):
    # TODO(Cris): Add parallelism to this code like in the cell complex conversion code.
    max_dimension = -1
    complexes = []
    num_features = [None for _ in range(expansion_dim+1)]                           # Indeed never used when evaluated on benchmarks. Only for testing.

    for data in tqdm(dataset):
        complex = compute_clique_complex_with_gudhi(data.x, data.edge_index, data.num_nodes,        # **not computing data.edge_attr. instead, aggregating vertices**
            expansion_dim=expansion_dim, y=data.y, include_down_adj=include_down_adj,
            init_method=init_method)                                                # get complex
        if complex.dimension > max_dimension:
            max_dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features              # assign cochains[dim].num_features to num_features[dim]
            else:
                assert num_features[dim] == complex.cochains[dim].num_features
        complexes.append(complex)

    return complexes, max_dimension, num_features[:max_dimension+1]






####################################################################
######## Support for rings as cells ################################
####################################################################


def get_rings(edge_index, max_k=7):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()

    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)
    # We represent rings with their original node ordering
    # so that we can easily read out the boundaries
    # The use of the `sorted_rings` set allows to discard
    # different isomorphisms which are however associated
    # to the same original ring – this happens due to the intrinsic
    # symmetries of cycles
    rings = set()
    sorted_rings = set()
    for k in range(3, max_k+1):
        pattern = nx.cycle_graph(k)
        pattern_edge_list = list(pattern.edges)
        pattern_gt = gt.Graph(directed=False)
        pattern_gt.add_edge_list(pattern_edge_list)
        sub_isos = top.subgraph_isomorphism(pattern_gt, graph_gt, induced=True, subgraph=True,
                                           generator=True)              # induced ring only
        sub_iso_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_isos)
        for iso in sub_iso_sets:
            if tuple(sorted(iso)) not in sorted_rings:      # sorted_rings to make sure no duplicate rings
                rings.add(iso)
                sorted_rings.add(tuple(sorted(iso)))
    rings = list(rings)
    return rings


def build_tables_with_rings(edge_index, simplex_tree, size, max_k):
    
    # Build simplex tables and id_maps up to edges by conveniently
    # invoking the code for simplicial complexes
    cell_tables, id_maps = build_tables(simplex_tree, size)
    
    # Find rings in the graph
    rings = get_rings(edge_index, max_k=max_k)
    
    if len(rings) > 0:
        # Extend the tables with rings as 2-cells
        id_maps += [{}]
        cell_tables += [[]]
        assert len(cell_tables) == 3, cell_tables
        for cell in rings:
            next_id = len(cell_tables[2])
            id_maps[2][cell] = next_id
            cell_tables[2].append(list(cell))

    return cell_tables, id_maps


def get_ring_boundaries(ring):
    boundaries = list()
    for n in range(len(ring)):
        a = n
        if n + 1 == len(ring):
            b = 0
        else:
            b = n + 1
        # We represent the boundaries in lexicographic order
        # so to be compatible with 0- and 1- dim cells
        # extracted as simplices with gudhi
        boundaries.append(tuple(sorted([ring[a], ring[b]])))
    return sorted(boundaries)


def extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps):
    """Build two maps: cell -> its coboundaries and cell -> its boundaries"""

    # Find boundaries and coboundaries up to edges by conveniently
    # invoking the code for simplicial complexes
    assert simplex_tree.dimension() <= 1
    boundaries_tables, boundaries, coboundaries = extract_boundaries_and_coboundaries_from_simplex_tree(
                                            simplex_tree, id_maps, simplex_tree.dimension())
    
    assert len(id_maps) <= 3
    if len(id_maps) == 3:
        # Extend tables with boundary and coboundary information of rings
        boundaries += [{}]
        coboundaries += [{}]
        boundaries_tables += [[]]
        for cell in id_maps[2]:
            cell_boundaries = get_ring_boundaries(cell)
            boundaries[2][cell] = list()
            boundaries_tables[2].append([])
            for boundary in cell_boundaries:
                assert boundary in id_maps[1], boundary
                boundaries[2][cell].append(boundary)
                if boundary not in coboundaries[1]:
                    coboundaries[1][boundary] = list()
                coboundaries[1][boundary].append(cell)
                boundaries_tables[2][-1].append(id_maps[1][boundary])
    
    return boundaries_tables, boundaries, coboundaries


def compute_ring_2complex(x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          size: int, y: Optional[Union[Tensor, np.ndarray]] = None, max_k: int = 7,
                          include_down_adj=True, init_method: str = 'sum',
                          init_edges=True, init_rings=False) -> Complex:
    """Generates a ring 2-complex of a pyG graph via graph-tool.

    Args:
        x: The feature matrix for the nodes of the graph (shape [num_vertices, num_v_feats])
        edge_index: The edge_index of the graph (a tensor of shape [2, num_edges])
        edge_attr: The feature matrix for the edges of the graph (shape [num_edges, num_e_feats])
        size: The number of nodes in the graph
        y: Labels for the graph nodes or a label for the whole graph.
        max_k: maximum length of rings to look for.
        include_down_adj: Whether to add down adj in the complex or not
        init_method: How to initialise features at higher levels.
    """
    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    # For parallel processing with joblib we need to pass numpy arrays as inputs
    # Therefore, we convert here everything back to a tensor.
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # Creates the gudhi-based simplicial complex up to edges
    simplex_tree = pyg_to_simplex_tree(edge_index, size)
    assert simplex_tree.dimension() <= 1
    if simplex_tree.dimension() == 0:
        assert edge_index.size(1) == 0

    # Builds tables of the cellular complexes at each level and their IDs
    cell_tables, id_maps = build_tables_with_rings(edge_index, simplex_tree, size, max_k)
    assert len(id_maps) <= 3
    complex_dim = len(id_maps)-1

    # Extracts the boundaries and coboundaries of each cell in the complex
    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_with_rings(simplex_tree, id_maps)

    # Computes the adjacencies between all the cells in the complex;
    # here we force complex dimension to be 2
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)
    
    # Construct features for the higher dimensions
    xs = [x, None, None]
    constructed_features = construct_features(x, cell_tables, init_method)
    if simplex_tree.dimension() == 0:
        assert len(constructed_features) == 1
    if init_rings and len(constructed_features) > 2:
        xs[2] = constructed_features[2]
    
    if init_edges and simplex_tree.dimension() >= 1:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        # If we have edge-features we simply use them for 1-cells
        else:
            # If edge_attr is a list of scalar features, make it a matrix
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # Retrieve feats and check edge features are undirected
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats

            # Build edge feature matrix
            max_id = max(ex.keys())
            edge_feats = []
            assert len(cell_tables[1]) == max_id + 1
            for id in range(max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            assert xs[1].dim() == 2
            assert xs[1].size(0) == len(id_maps[1])
            assert xs[1].size(1) == edge_attr.size(1)

    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, size)

    cochains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               cell_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_rings(dataset, max_ring_size=7, include_down_adj=False,
                                     init_method: str = 'sum', init_edges=True, init_rings=False,
                                     n_jobs=1):
    dimension = -1
    num_features = [None, None, None]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x

    # Process the dataset in parallel
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    # It is important we supply a numpy array here. tensors seem to slow joblib down significantly.
    complexes = parallel(delayed(compute_ring_2complex)(
        maybe_convert_to_numpy(data.x), maybe_convert_to_numpy(data.edge_index),
        maybe_convert_to_numpy(data.edge_attr),
        data.num_nodes, y=maybe_convert_to_numpy(data.y), max_k=max_ring_size,
        include_down_adj=include_down_adj, init_method=init_method,
        init_edges=init_edges, init_rings=init_rings) for data in dataset)

    # NB: here we perform additional checks to verify the order of complexes
    # corresponds to that of input graphs after _parallel_ conversion
    for c, complex in enumerate(complexes):

        # Handle dimension and number of features
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features

        # Validate against graph
        graph = dataset[c]
        if complex.y is None:
            assert graph.y is None
        else:
            assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.cochains[0].x, graph.x)
        if complex.dimension >= 1:
            assert complex.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    return complexes, dimension, num_features[:dimension+1]



####################################################################
######## Support for Path Complexes ################################
####################################################################

def get_paths(edge_index, max_k = 3):
    '''
        Get all paths having length [2, max_k].

        Args:
            edge_index: edge index matrix similar to PyG
            max_k: max path length
        
        Returns:
            paths: list of all possible paths
    '''
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.numpy()
    
    edge_list = edge_index.T
    graph_gt = gt.Graph(directed=False)
    graph_gt.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph_gt)
    gt.stats.remove_parallel_edges(graph_gt)

    paths = set()
    complex_dim = 0
    for k in range(2, max_k + 2):
        path_pattern = nx.path_graph(k)         # only simple path
        path_pattern_edge_list = list(path_pattern.edges)
        path_pattern_gt = gt.Graph(directed = False)
        path_pattern_gt.add_edge_list(path_pattern_edge_list)
        sub_path_isos = top.subgraph_isomorphism(path_pattern_gt, graph_gt, induced = False, subgraph = True, generator = True)
        
        sub_path_isos_sets = map(lambda isomorphism: tuple(isomorphism.a), sub_path_isos)         # assign vertex array to iso
        for iso in sub_path_isos_sets:
            if (tuple(reversed(iso)) not in paths and iso[0] < iso[-1]):        # add constraint for ease of debug
                paths.add(iso)
                complex_dim = k-1
    paths = list(paths)
    return paths, complex_dim

def build_tables_with_paths(edge_index, num_nodes, max_k):
    '''
        Build tables id_maps and path_tables.
        
        Args:
            edge_index: edge_index matrix similar to PyG
            num_nodes: number of nodes
            max_k: maximum dimension of p-path
        
        Outputs:
            path_tables: table that map an id to a path (as list) in that dim
            id_maps: table to map a path (as tuple) to an id in that dim
    '''
    paths, complex_dim = get_paths(edge_index, max_k = max_k)                # list of paths

    id_maps = [{} for _ in range(complex_dim + 1)]
    path_tables = [[] for _ in range(complex_dim + 1)]

    

    path_tables[0] = [[v] for v in range(num_nodes)]
    id_maps[0] = {tuple([v]): v for v in range(num_nodes)}

    for path in paths:
        dim = len(path) - 1
        if dim == 0:                            # Already initialized
            continue
        
        next_id = len(path_tables[dim])         # get next id of current dim
        id_maps[dim][tuple(path)] = next_id
        path_tables[dim].append(list(path))

    return path_tables, id_maps

def get_path_boundaries(path, path_tables):
    '''
        Extract path boundaries

        Args:
            path: current path that has boundaries extracted.
            path_tables: path tables that contains all paths of different dims
        
        Returns:
            allowed_boundaries: boundaries of path (as list of tuples)
    '''
    # we need path_tables to make sure boundaries belong to path table
    path_length = len(path) - 1
    num_vertices = len(path)
    allowed_boundaries = list()
    for i in range(num_vertices):
        boundary = list(path[0 : i] + path[(i + 1) : num_vertices])
        if (boundary in path_tables[path_length - 1]):
            allowed_boundaries.append(tuple(boundary))
        elif (list(reversed(boundary)) in path_tables[path_length - 1]):
            allowed_boundaries.append(tuple(reversed(boundary)))
        else:   # boundary is not allowed
            continue
    return allowed_boundaries

def extract_boundaries_and_coboundaries_of_paths(id_maps, path_tables):
    '''
        Extract boundaries and coboundaries of paths

        Args:
            id_maps: table to map a path to an id in that dim
            path_tables: table that map an id to a path in that dim
            complex_dim: dimension of path complex
        
        Returns:
            boundaries: list of dictionary mapping tuple --> list, where each element in list corresponds to dim,
                while dictionary maps path --> boundaries as list.
            coboundaries: list of dictionary mapping tuple --> list, where each element in list corresponds to dim,
                while dictionary maps boundary --> cofaces as list.
            boundaries_tables: list of list mapping int --> list, where each element in list corresponds to dim,
                while each element in inner list corresponds to list of boundary indices.
    '''
    # Initialize boundaries and coboundaries and tables.
    complex_dim = len(id_maps) - 1
    boundaries = [{} for _ in range(complex_dim + 1)]
    coboundaries = [{} for _ in range(complex_dim + 1)]
    boundaries_tables = [[] for _ in range(complex_dim + 1)]

    # Find boundaries
    for path_dim in range(len(id_maps)):
        for path in id_maps[path_dim]:
            if path_dim > 0:
                path_boundaries = get_path_boundaries(path, path_tables)
                boundaries_tables[path_dim].append(list())      # initialize empty list for current path index

                boundaries[path_dim][path] = list()
                for boundary in path_boundaries:
                    boundaries[path_dim][path].append(boundary)
                    if boundary not in coboundaries[path_dim - 1]:
                        coboundaries[path_dim - 1][boundary] = list()
                    coboundaries[path_dim - 1][boundary].append(path)

                    # get path id and boundary id
                    path_id = id_maps[path_dim][path]
                    boundary_id =  id_maps[path_dim - 1][boundary]                  
                    boundaries_tables[path_dim][path_id].append(boundary_id)

                # sort boundaries tables for ease of debugging
                boundaries_tables[path_dim][path_id] = sorted(boundaries_tables[path_dim][path_id])
            else:       # path_dim == 0, so there is no boundary, and coboundary already updated when iterating through dim = 1
                continue
    
    return boundaries_tables, boundaries, coboundaries

def compute_path_complex(x: Union[Tensor, np.ndarray], edge_index: Union[Tensor, np.ndarray],
                          edge_attr: Optional[Union[Tensor, np.ndarray]],
                          num_nodes: int, y: Optional[Union[Tensor, np.ndarray]] = None, max_k: int = 3,
                          include_down_adj=True, init_method: str = 'sum',
                          init_edges = True, init_high_order_paths = False) -> Complex:
    '''
        Generate path complex
    '''
    assert x is not None
    assert isinstance(edge_index, np.ndarray) or isinstance(edge_index, Tensor)

    # For parallel processing with joblib we need to pass numpy arrays as inputs
    # Therefore, we convert here everything back to a tensor.
    if isinstance(x, np.ndarray):
        x = torch.tensor(x)
    if isinstance(edge_index, np.ndarray):
        edge_index = torch.tensor(edge_index)
    if isinstance(edge_attr, np.ndarray):
        edge_attr = torch.tensor(edge_attr)
    if isinstance(y, np.ndarray):
        y = torch.tensor(y)

    # Build tables of the path complexes
    path_tables, id_maps = build_tables_with_paths(edge_index, num_nodes, max_k)

    # Extract boundaries and coboundaries
    boundaries_tables, boundaries, co_boundaries = extract_boundaries_and_coboundaries_of_paths(id_maps, path_tables)

    # Computes adjacencies
    complex_dim = len(id_maps) - 1
    shared_boundaries, shared_coboundaries, lower_idx, upper_idx = build_adj(boundaries, co_boundaries, id_maps,
                                                                   complex_dim, include_down_adj)

    # Construct features for the higher dimensions
    xs = [None for _ in range(complex_dim + 1)]
    xs[0] = x
    if init_edges or init_high_order_paths:
        constructed_features = construct_features(x, path_tables, init_method)
    
    if complex_dim >= 1 and init_edges:
        if edge_attr is None:
            xs[1] = constructed_features[1]
        else:
            # If edge_attr is a list of scalar features, make it a matrix
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            # Retrieve feats and check edge features are undirected
            ex = dict()
            for e, edge in enumerate(edge_index.numpy().T):
                canon_edge = tuple(sorted(edge))
                edge_id = id_maps[1][canon_edge]
                edge_feats = edge_attr[e]
                if edge_id in ex:
                    assert torch.equal(ex[edge_id], edge_feats)
                else:
                    ex[edge_id] = edge_feats
            # Build edge feature matrix
            max_id = max(ex.keys())
            edge_feats = []
            assert len(path_tables[1]) == max_id + 1
            for id in range(max_id + 1):
                edge_feats.append(ex[id])
            xs[1] = torch.stack(edge_feats, dim=0)
            assert xs[1].dim() == 2
            assert xs[1].size(0) == len(id_maps[1])
            assert xs[1].size(1) == edge_attr.size(1)
    
    # features for dim >= 2
    if complex_dim >= 2 and init_high_order_paths:
        for i in range(2, complex_dim + 1):
            xs[i] = constructed_features[i]
    
    # Initialise the node / complex labels
    v_y, complex_y = extract_labels(y, num_nodes)

    # generate cochains
    cochains = []
    for i in range(complex_dim + 1):
        y = v_y if i == 0 else None
        cochain = generate_cochain(i, xs[i], upper_idx, lower_idx, shared_boundaries, shared_coboundaries,
                               path_tables, boundaries_tables, complex_dim=complex_dim, y=y)
        cochains.append(cochain)

    return Complex(*cochains, y=complex_y, dimension=complex_dim)


def convert_graph_dataset_with_paths(dataset, max_k = 3, include_down_adj=False, init_edges = True, init_high_order_paths = False, 
                                     init_method: str = 'sum', n_jobs=1):
    dimension = -1
    num_features = [None for _ in range(max_k + 1)]

    def maybe_convert_to_numpy(x):
        if isinstance(x, Tensor):
            return x.numpy()
        return x
    

    # Process the dataset in parallel
    parallel = ProgressParallel(n_jobs=n_jobs, use_tqdm=True, total=len(dataset))
    # It is important we supply a numpy array here. tensors seem to slow joblib down significantly.
    complexes = parallel(
            delayed(compute_path_complex)(
                    maybe_convert_to_numpy(data.x), maybe_convert_to_numpy(data.edge_index),
                    maybe_convert_to_numpy(data.edge_attr),
                    data.num_nodes, y=maybe_convert_to_numpy(data.y), max_k= max_k,
                    include_down_adj=include_down_adj, init_method=init_method,
                    init_edges = init_edges, init_high_order_paths = init_high_order_paths
                ) for data in dataset)

    # NB: here we perform additional checks to verify the order of complexes
    # corresponds to that of input graphs after _parallel_ conversion
    for c, complex in enumerate(complexes):

        # Handle dimension and number of features
        if complex.dimension > dimension:
            dimension = complex.dimension
        for dim in range(complex.dimension + 1):
            if num_features[dim] is None:
                num_features[dim] = complex.cochains[dim].num_features
            else:
                assert num_features[dim] == complex.cochains[dim].num_features

        # Validate against graph
        graph = dataset[c]
        if complex.y is None:
            assert graph.y is None
        else:
            assert torch.equal(complex.y, graph.y)
        assert torch.equal(complex.cochains[0].x, graph.x)
        if complex.dimension >= 1 and init_edges:
            assert complex.cochains[1].x.size(0) == (graph.edge_index.size(1) // 2)

    return complexes, dimension, num_features[:dimension+1]
    



if __name__ == "__main__":
    graph = '''
      4
     / \\
    3---2
    |   |
    0---1
    '''
    complex_dim = 2
    edge_index = np.array(
        [
            [0, 1, 0, 3, 1, 2, 2, 3, 2, 4, 3, 4],
            [1, 0, 3, 0, 2, 1, 3, 2, 4, 2, 4, 3]
        ]
    )
    print("Graph")
    print(graph)

    path_tables, id_maps = build_tables_with_paths(edge_index, num_nodes = 5, max_k = complex_dim)
    print("Path Tables:")
    for dim in range(len(path_tables)):
        print("Dim ", dim, ":")
        print(path_tables[dim])
    print("---------------------------------")
    print("ID maps")
    for dim in range(len(id_maps)):
        print("Dim ", dim, ":")
        print(id_maps[dim])
    print("---------------------------------")
    boundaries = get_path_boundaries((2,1,0,3), path_tables)
    print("Boundary of (2,1,0,3):")
    for boundary in boundaries:
        print(boundary)
    print("---------------------------------")
    boundaries_tables, boundaries, coboundaries,  = extract_boundaries_and_coboundaries_of_paths(id_maps, path_tables)
    print("Boundaries")
    for dim in range(len(boundaries)):
        print("Dim ", dim, ":")
        print(boundaries[dim])
    print("---------------------------------")
    print("Coboundaries")
    for dim in range(len(coboundaries)):
        print("Dim ", dim, ":")
        print(coboundaries[dim])
    print("---------------------------------")
    print("Boundaries Tables")
    for dim in range(len(boundaries_tables)):
        print("Dim ", dim, ":")
        print(boundaries_tables[dim])
    print("---------------------------------")

    # st = pyg_to_simplex_tree(torch.Tensor(edge_index), 5)
    # st.expansion(2)  # Computes the clique complex up to the desired dim.
    # simplex_tables, id_maps = build_tables(st, 5)
    # print(simplex_tables)
    # print(id_maps)
    # print(extract_boundaries_and_coboundaries_from_simplex_tree(st, id_maps, 2)[0])

    all_shared_boundaries, all_shared_coboundaries, lower_indexes, upper_indexes = build_adj(boundaries, coboundaries, id_maps, len(id_maps) - 1, include_down_adj = True)
    print("Lower Indexes")
    for dim in range(len(lower_indexes)):
        print("Dim ", dim, ":")
        print(np.array(lower_indexes[dim]).T.shape)
        print(np.array(lower_indexes[dim]).T)
    print("---------------------------------")
    print("Upper Indexes")
    for dim in range(len(upper_indexes)):
        print("Dim ", dim, ":")
        print(np.array(upper_indexes[dim]).T.shape)
        print(np.array(upper_indexes[dim]).T)
    print("---------------------------------")
    print("All Shared Boundaries")
    for dim in range(len(all_shared_boundaries)):
        print("Dim ", dim, ":")
        print(np.array(all_shared_boundaries[dim]).T.shape)
        print(np.array(all_shared_boundaries[dim]).T)
    print("---------------------------------")
    print("All Shared Coboundaries")
    for dim in range(len(all_shared_coboundaries)):
        print("Dim ", dim, ":")
        print(np.array(all_shared_coboundaries[dim]).T.shape)
        print(np.array(all_shared_coboundaries[dim]).T)
    print("---------------------------------")
    x = torch.Tensor([1,2,3,4,5])
    complex = compute_path_complex(x, edge_index, None, num_nodes=5, y = None, max_k = complex_dim)
    complexes, _, _ = convert_graph_dataset_with_paths([Data(x = x, edge_index = torch.tensor(edge_index, dtype=torch.long), y = None), 
                                                        Data(x, torch.tensor(edge_index, dtype=torch.long), y = None)], complex_dim)
    print(complexes[0].cochains)
    