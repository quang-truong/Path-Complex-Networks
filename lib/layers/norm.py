from torch.nn import BatchNorm1d as BN, LayerNorm as LN, Identity

def get_graph_norm(norm):
    if norm == 'bn':
        return BN
    elif norm == 'ln':
        return LN
    elif norm == 'id':
        return Identity
    else:
        raise ValueError(f'Graph Normalisation {norm} not currently supported')