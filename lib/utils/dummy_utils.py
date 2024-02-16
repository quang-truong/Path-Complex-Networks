import torch

from torch_geometric.data import Data
from lib.utils.dummy_cell_complexes import *


# TODO: make the features for these dummy complexes disjoint to stress tests even more
def convert_to_graph(complex):
    """Extracts the underlying graph of a cochain complex."""
    assert 0 in complex.cochains
    assert complex.cochains[0].num_cells > 0
    cochain = complex.cochains[0]
    x = cochain.x
    y = complex.y
    edge_attr = None
    if cochain.upper_index is None:
        edge_index = torch.LongTensor([[], []])
    else:
        edge_index = cochain.upper_index
        if 1 in complex.cochains and complex.cochains[1].x is not None and cochain.shared_coboundaries is not None:
            edge_attr = torch.index_select(complex.cochains[1].x, 0, cochain.shared_coboundaries)
    if edge_attr is None:
        edge_attr = torch.FloatTensor([[]])
    graph = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    return graph

def get_testing_cell_complex_list():
    """Returns a list of cell complexes used for testing. The list contains many edge cases."""
    return [get_fullstop_complex(), get_pyramid_complex(), get_house_complex(), get_kite_complex(), get_square_complex(),
            get_square_dot_complex(), get_square_complex(), get_fullstop_complex(), get_house_complex(),
            get_kite_complex(), get_pyramid_complex(), get_bridged_complex(), get_square_dot_complex(), get_colon_complex(),
            get_filled_square_complex(), get_molecular_complex(), get_fullstop_complex(), get_colon_complex(),
            get_bridged_complex(), get_colon_complex(), get_fullstop_complex(), get_fullstop_complex(), get_colon_complex()]


def get_mol_testing_cell_complex_list():
    """Returns a list of cell complexes used for testing. The list contains many edge cases."""
    return [get_house_complex(), get_kite_complex(), get_square_complex(), get_fullstop_complex(), get_bridged_complex(),
            get_square_dot_complex(), get_square_complex(), get_filled_square_complex(), get_colon_complex(), get_bridged_complex(),
            get_kite_complex(), get_square_dot_complex(), get_colon_complex(), get_molecular_complex(), get_bridged_complex(),
            get_filled_square_complex(), get_molecular_complex(), get_fullstop_complex(), get_colon_complex()]