import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import reset
from torch.nn import Linear
from lib.layers.cin_conv import SparseCINConv
from lib.data.complex import ComplexBatch
from lib.layers.non_linear import get_nonlinearity
from lib.layers.norm import get_graph_norm
from lib.layers.pooling import pool_complex

from torch_geometric.nn import JumpingKnowledge

class SparseCIN(torch.nn.Module):
    """
    A cellular version of GIN.

    This model is based on
    https://github.com/rusty1s/pytorch_geometric/blob/master/benchmark/kernel/gin.py
    """

    def __init__(self, 
                 num_input_features, 
                 num_node_labels, 
                 num_classes,
                 num_layers,
                 hidden,
                 dropout_rate: float = 0.5,
                 in_dropout_rate = 0.0,
                 max_dim: int = 2, 
                 jump_mode=None, 
                 nonlinearity='relu', 
                 readout='sum',
                 train_eps=False, 
                 final_hidden_multiplier: int = 2, 
                 final_readout='sum', 
                 apply_dropout_before='lin2',
                 use_coboundaries=False,
                 use_boundaries=False,
                 include_down_adj=False,
                 init_reduce='sum',
                 graph_norm='bn',
                 disable_graph_norm=False,
                 readout_dims=(0, 1, 2, 4, 5),
                 num_layers_update = 2,
                 num_layers_combine = 1,
                 num_fc_layers=2,
                 conv_type="B"):
        super(SparseCIN, self).__init__()

        self.max_dim = max_dim
        self.jump_mode = None if jump_mode == 'None' else jump_mode

        # Find Readout Dims
        self.readout_dims = tuple([dim for dim in readout_dims if dim <= max_dim])

        # Initialize Readout
        self.readout = readout
        self.final_readout = final_readout


        # Initialize Dropout
        self.dropout_rate = dropout_rate
        self.apply_dropout_before = apply_dropout_before

        # Initialize convs, non-linear act. function, and graph norm
        self.convs = torch.nn.ModuleList()
        self.nonlinearity = nonlinearity
        self.graph_norm = get_graph_norm(graph_norm)
        act_module = get_nonlinearity(nonlinearity, return_module=True)

        ## Add SparseCINConv
        for i in range(num_layers):
            layer_dim = num_input_features if i == 0 else hidden
            self.convs.append(
                SparseCINConv(up_msg_size=layer_dim, down_msg_size=layer_dim,
                    boundary_msg_size=layer_dim, passed_msg_boundaries_nn=None,
                    passed_msg_up_nn=None, passed_msg_down_nn=None, passed_update_up_nn=None, passed_update_down_nn=None,
                    passed_update_boundaries_nn=None, train_eps=train_eps, max_dim=self.max_dim,
                    hidden=hidden, act_module=act_module, layer_dim=layer_dim,
                    graph_norm=self.graph_norm, use_coboundaries=use_coboundaries, use_boundaries=use_boundaries,
                    include_down_adj=include_down_adj, num_layers_update=num_layers_update, num_layers_combine=num_layers_combine,
                    conv_type=conv_type)
            )

        ## Jumping Knowledge
        self.jump = JumpingKnowledge(self.jump_mode) if self.jump_mode is not None else None

        ## Disable graph norm
        tmp_graph_norm = get_graph_norm('id') if disable_graph_norm else self.graph_norm

        ## First Linear Layer and Final Readout FC
        self.lin1s = torch.nn.ModuleList()
        for _ in range(max_dim + 1):
            if self.jump_mode == 'cat':
                self.lin1s.append(torch.nn.Sequential
                    (
                        Linear(num_layers * hidden, final_hidden_multiplier * hidden, bias=False),
                        tmp_graph_norm(final_hidden_multiplier * hidden),
                        act_module(),
                    )
                )
            else:
                self.lin1s.append(torch.nn.Sequential
                    (
                        Linear(hidden, final_hidden_multiplier * hidden),
                        tmp_graph_norm(final_hidden_multiplier * hidden),
                        act_module(),
                    )
                )

        if self.final_readout == 'concat':
            self.readout_fc = torch.nn.Sequential(
                    Linear(final_hidden_multiplier * hidden * len(self.readout_dims), final_hidden_multiplier * hidden),
                    tmp_graph_norm(final_hidden_multiplier * hidden),
                    act_module(),
                )

        if num_fc_layers == 2:
            self.lin2s = torch.nn.Sequential(
                Linear(final_hidden_multiplier * hidden, num_classes)
            )
        elif num_fc_layers == 3:
            self.lin2s = torch.nn.Sequential(
                Linear(final_hidden_multiplier * hidden, hidden),
                act_module(),
                Linear(hidden, num_classes)
            )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump_mode is not None:
            self.jump.reset_parameters()
        reset(self.lin1s)
        reset(self.lin2s)


    def jump_complex(self, jump_xs):
        # Perform JumpingKnowledge at each level of the complex
        xs = []
        for jumpx in jump_xs:
            xs += [self.jump(jumpx)]
        return xs

    def forward(self, data: ComplexBatch, include_partial=False):
        xs = None
        jump_xs = None
        res = {}

        for c, conv in enumerate(self.convs):
            params = data.get_all_cochain_params(max_dim=self.max_dim, include_down_features=False)
            xs = conv(*params, start_to_process=0)
            data.set_xs(xs)

            if include_partial:
                for k in range(len(xs)):
                    res[f"layer{c}_{k}"] = xs[k]
            
            if self.jump_mode is not None:
                if jump_xs is None:
                    jump_xs = [[] for _ in xs]
                for i, x in enumerate(xs):
                    jump_xs[i] += [x]

        if self.jump_mode is not None:
            xs = self.jump_complex(jump_xs)

        xs = pool_complex(xs, data, self.max_dim, self.readout)
        # Select the dimensions we want at the end.
        xs = [xs[i] for i in self.readout_dims]

        if include_partial:
            for k in range(len(xs)):
                res[f"pool_{k}"] = xs[k]
        
        new_xs = []
        for i, x in enumerate(xs):
            if self.apply_dropout_before == 'lin1':
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            new_xs.append(self.lin1s[self.readout_dims[i]](x))

        x = torch.stack(new_xs, dim=0)
        
        if self.apply_dropout_before == 'final_readout':
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        if self.final_readout == 'mean':
            x = x.mean(0)
        elif self.final_readout == 'sum':
            x = x.sum(0)
        elif self.final_readout == 'concat':
            x = x.transpose(0,1)
            x = torch.flatten(x, 1, -1)
            x = self.readout_fc(x)
        else:
            raise NotImplementedError
        if self.apply_dropout_before not in ['lin1', 'final_readout']:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.lin2s(x)

        if include_partial:
            res['out'] = x
            return x, res
        return x

    def __repr__(self):
        return self.__class__.__name__
