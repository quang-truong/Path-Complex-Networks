import torch
from torch_geometric.nn import global_mean_pool, global_add_pool

def get_pooling_fn(readout):
    if readout == 'sum':
        return global_add_pool
    elif readout == 'mean':
        return global_mean_pool
    else:
        raise NotImplementedError('Readout {} is not currently supported.'.format(readout))
    

def pool_complex(xs, data, max_dim, readout_type):
    pooling_fn = get_pooling_fn(readout_type)
    # All complexes have nodes so we can extract the batch size from cochains[0]
    batch_size = data.cochains[0].batch.max() + 1
    # The MP output is of shape [message_passing_dim, batch_size, feature_dim]
    pooled_xs = torch.zeros(max_dim+1, batch_size, xs[0].size(-1),
                            device=batch_size.device)
    for i in range(len(xs)):
        # It's very important that size is supplied.
        pooled_xs[i, :, :] = pooling_fn(xs[i], data.cochains[i].batch, size=batch_size)
    return pooled_xs