import torch

from typing import Callable, Optional
from torch import Tensor
from lib.message_passing.cochain_mp import CochainMessagePassing
from lib.message_passing.cochain_mp_params import CochainMessagePassingParams
from torch_geometric.nn.inits import reset
from torch.nn import Linear, Sequential, BatchNorm1d as BN
from .catter import Catter


class SparseCINCochainConv(CochainMessagePassing):
    """This is a CIN Cochain layer that operates of boundaries and upper adjacent cells."""
    def __init__(self, dim: int,
                 up_msg_size: int,
                 down_msg_size: int,
                 boundary_msg_size: Optional[int],
                 use_coboundaries: bool,
                 use_boundaries: bool,
                 include_down_adj: bool,
                 msg_up_nn: Callable,
                 msg_down_nn: Optional[Callable],
                 msg_boundaries_nn: Callable,
                 update_up_nn: Callable,
                 update_down_nn: Optional[Callable],
                 update_boundaries_nn: Callable,
                 combine_nn: Callable,
                 eps: float = 0.,
                 train_eps: bool = False):
        super(SparseCINCochainConv, self).__init__(up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                                                 use_down_msg=False)
        self.dim = dim
        self.include_down_adj = include_down_adj
        self.msg_up_nn = msg_up_nn
        self.msg_down_nn = msg_down_nn
        self.msg_boundaries_nn = msg_boundaries_nn
        self.update_up_nn = update_up_nn                            # mlp for updates from upper adjacent neighbors (CochainMessagePassing only handles messages)
        self.update_down_nn = update_down_nn
        self.update_boundaries_nn = update_boundaries_nn            # mlp for updates from boundaries (CochainMessagePassing only handles messages)
        self.combine_nn = combine_nn                                # mlp to combine out_up and out_boundaries (CochainMessagePassing only handles messages)
        self.use_coboundaries = use_coboundaries
        self.use_boundaries = use_boundaries
        self.initial_eps = eps
        if train_eps:
            self.eps1 = torch.nn.Parameter(torch.Tensor([eps]))
            self.eps2 = torch.nn.Parameter(torch.Tensor([eps]))
            if include_down_adj:
                self.eps3 = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps1', torch.Tensor([eps]))
            self.register_buffer('eps2', torch.Tensor([eps]))
            if include_down_adj:
                self.register_buffer('eps3', torch.Tensor([eps]))
        self.reset_parameters()

    def forward(self, cochain: CochainMessagePassingParams):
        if self.include_down_adj:
            out_up, out_down, out_boundaries = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              up_attr=cochain.kwargs['up_attr'],
                                              down_attr=cochain.kwargs['down_attr'],
                                              boundary_attr=cochain.kwargs['boundary_attr'])
        else:
            out_up, _, out_boundaries = self.propagate(cochain.up_index, cochain.down_index,
                                              cochain.boundary_index, x=cochain.x,
                                              up_attr=cochain.kwargs['up_attr'],
                                              boundary_attr=cochain.kwargs['boundary_attr'])

        # As in GIN, we can learn an injective update function for each multi-set                   # Update aggregated messages
        out_up += (1 + self.eps1) * cochain.x
        out_boundaries += (1 + self.eps2) * cochain.x
        if self.include_down_adj:
            out_down += (1 + self.eps3) * cochain.x
        out_up = self.update_up_nn(out_up)
        out_boundaries = self.update_boundaries_nn(out_boundaries)
        if self.include_down_adj:
            out_down = self.update_down_nn(out_down)

        # We need to combine the two such that the output is injective
        # Because the cross product of countable spaces is countable, then such a function exists.
        # And we can learn it with another MLP.
        if self.include_down_adj:
            return self.combine_nn(torch.cat([out_up, out_down, out_boundaries], dim=-1))
        else:
            return self.combine_nn(torch.cat([out_up, out_boundaries], dim=-1))

    def reset_parameters(self):
        reset(self.msg_up_nn)
        reset(self.msg_boundaries_nn)
        reset(self.update_up_nn)
        reset(self.update_boundaries_nn)
        reset(self.combine_nn)
        self.eps1.data.fill_(self.initial_eps)
        self.eps2.data.fill_(self.initial_eps)
        if self.include_down_adj:
            reset(self.msg_down_nn)
            reset(self.update_down_nn)
            self.eps3.data.fill_(self.initial_eps)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:            # override message_up method (message from upper adjacenct neighbors)
        if self.use_coboundaries:
            return self.msg_up_nn((up_x_j, up_attr))                            # up_attr is formed from cochain[dim+1] for every connection, up_x_j is formed from boundary (from init_conv)
        else:
            return self.msg_up_nn(up_x_j)
        
    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:      # override message_down method (message from upper adjacenct neighbors)
        if self.use_boundaries:
            return self.msg_down_nn((down_x_j, down_attr))
        else:
            return self.msg_down_nn(down_x_j)
    
    def message_boundary(self, boundary_x_j: Tensor) -> Tensor:                 # override message_boundary method (message from boundaries)
        return self.msg_boundaries_nn(boundary_x_j)                             # boundary_x_j is extracted automatically based on boundary_attr
    

class SparseCINConv(torch.nn.Module):
    """A cellular version of GIN which performs message passing from  cellular upper
    neighbors and boundaries, but not from lower neighbors (hence why "Sparse")
    """

    # TODO: Refactor the way we pass networks externally to allow for different networks per dim.
    def __init__(self, up_msg_size: int, down_msg_size: int, boundary_msg_size: Optional[int],
                 passed_msg_up_nn: Optional[Callable],
                 passed_msg_down_nn: Optional[Callable],
                 passed_msg_boundaries_nn: Optional[Callable], 
                 passed_update_up_nn: Optional[Callable],
                 passed_update_down_nn: Optional[Callable],  
                 passed_update_boundaries_nn: Optional[Callable], 
                 eps: float = 0., 
                 train_eps: bool = False, 
                 max_dim: int = 2,
                 graph_norm=BN, 
                 use_coboundaries=False,
                 use_boundaries=False,
                 include_down_adj=False,
                 num_layers_update: int = 2,
                 num_layers_combine: int = 1,
                 conv_type: str = "B",
                 **kwargs):
        super(SparseCINConv, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            msg_up_nn = passed_msg_up_nn
            if msg_up_nn is None:
                if use_coboundaries:                                                # ZINC dataset needs use_coboundaries
                    msg_up_nn = Sequential(
                            Catter(),
                            Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),   # Linear: (up_x_j, up_attr) --> message_attr
                            kwargs['act_module']())
                else:
                    msg_up_nn = torch.nn.Identity()
            
            if include_down_adj:
                msg_down_nn = passed_msg_down_nn
                if msg_down_nn is None:
                    if use_boundaries:                                                # ZINC dataset needs use_coboundaries
                        msg_down_nn = Sequential(
                                Catter(),
                                Linear(kwargs['layer_dim'] * 2, kwargs['layer_dim']),   # Linear: (up_x_j, up_attr) --> message_attr
                                kwargs['act_module']())
                    else:
                        msg_down_nn = torch.nn.Identity()
            else:
                msg_down_nn = None

            msg_boundaries_nn = passed_msg_boundaries_nn
            if msg_boundaries_nn is None:
                msg_boundaries_nn = torch.nn.Identity()

            # Update from Upper-adjacent Neighbors NN
            update_up_nn = passed_update_up_nn
            if update_up_nn is None:
                update_up_nn = []
                for i in range(num_layers_update):
                    layer_dim = kwargs['layer_dim'] if i == 0 else kwargs['hidden']
                    if conv_type == "A" or conv_type == "C":
                        update_up_nn.extend(
                            [Linear(layer_dim, kwargs['hidden'])]
                        )
                    elif conv_type == "B" or conv_type == "D":
                        update_up_nn.extend(
                            [Linear(layer_dim, kwargs['hidden']),
                             graph_norm(kwargs['hidden']),
                             kwargs['act_module']()]
                        )
                update_up_nn = Sequential(*update_up_nn)
            
            # Update from Lower-adjacent Neighbors NN
            if include_down_adj:
                update_down_nn = passed_update_down_nn
                if update_down_nn is None:
                    update_down_nn = []
                    for i in range(num_layers_update):
                        layer_dim = kwargs['layer_dim'] if i == 0 else kwargs['hidden']
                        if conv_type == "A" or conv_type == "C":
                            update_down_nn.extend(
                                [Linear(layer_dim, kwargs['hidden'])]
                            )
                        elif conv_type == "B" or conv_type == "D":
                            update_down_nn.extend(
                                [Linear(layer_dim, kwargs['hidden']),
                                graph_norm(kwargs['hidden']),
                                kwargs['act_module']()]
                            )
                    update_down_nn = Sequential(*update_down_nn)
            else:
                update_down_nn = None

            # Update from Boundaries NN
            update_boundaries_nn = passed_update_boundaries_nn
            if update_boundaries_nn is None:
                update_boundaries_nn = []
                for i in range(num_layers_update):
                    layer_dim = kwargs['layer_dim'] if i == 0 else kwargs['hidden']
                    if conv_type == "A" or conv_type == "C":
                        update_boundaries_nn.extend(
                            [Linear(layer_dim, kwargs['hidden'])]
                        )
                    elif conv_type == "B" or conv_type == "D":
                        update_boundaries_nn.extend(
                            [Linear(layer_dim, kwargs['hidden']),
                             graph_norm(kwargs['hidden']),
                             kwargs['act_module']()]
                        )
                update_boundaries_nn = Sequential(*update_boundaries_nn)
            
            # Combine NN
            combine_nn = []
            for i in range(num_layers_combine):
                if i == 0:
                    if include_down_adj:
                        layer_dim = kwargs['hidden']*3
                    else:
                        layer_dim = kwargs['hidden']*2
                else:
                    layer_dim = kwargs['hidden']
                if conv_type == "A" or conv_type == "D":
                    combine_nn.extend(
                            [Linear(layer_dim, kwargs['hidden']),
                            graph_norm(kwargs['hidden'])]
                    )
                elif conv_type == "B" or conv_type == "C":
                    combine_nn.extend(
                            [Linear(layer_dim, kwargs['hidden']),
                             graph_norm(kwargs['hidden']),
                             kwargs['act_module']()]
                    )
            combine_nn = Sequential(*combine_nn)

            mp = SparseCINCochainConv(dim, up_msg_size, down_msg_size, boundary_msg_size=boundary_msg_size,
                use_coboundaries=use_coboundaries, use_boundaries=use_boundaries, 
                include_down_adj=include_down_adj, msg_up_nn=msg_up_nn, msg_down_nn=msg_down_nn,
                msg_boundaries_nn=msg_boundaries_nn, update_up_nn=update_up_nn, update_down_nn=update_down_nn,
                update_boundaries_nn=update_boundaries_nn, combine_nn=combine_nn, eps=eps,
                train_eps=train_eps)
            self.mp_levels.append(mp)

    def forward(self, *cochain_params: CochainMessagePassingParams, start_to_process=0):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            if dim < start_to_process:
                out.append(cochain_params[dim].x)
            else:
                out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out