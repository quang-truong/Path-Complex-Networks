import torch

from torch import Tensor
from lib.message_passing.cochain_mp import CochainMessagePassing
from lib.message_passing.cochain_mp_params import CochainMessagePassingParams


class DummyCochainMessagePassing(CochainMessagePassing):
    """This is a dummy parameter-free message passing model used for testing."""
    def __init__(self, up_msg_size, down_msg_size, boundary_msg_size=None,
                 use_boundary_msg=False, use_down_msg=True):
        super(DummyCochainMessagePassing, self).__init__(up_msg_size, down_msg_size,
                                                       boundary_msg_size=boundary_msg_size,
                                                       use_boundary_msg=use_boundary_msg,
                                                       use_down_msg=use_down_msg)

    def message_up(self, up_x_j: Tensor, up_attr: Tensor) -> Tensor:
        # (num_up_adj, x_feature_dim) + (num_up_adj, up_feat_dim)
        # We assume the feature dim is the same across al levels
        return up_x_j + up_attr

    def message_down(self, down_x_j: Tensor, down_attr: Tensor) -> Tensor:
        # (num_down_adj, x_feature_dim) + (num_down_adj, down_feat_dim)
        # We assume the feature dim is the same across al levels
        return down_x_j + down_attr

    def forward(self, cochain: CochainMessagePassingParams):
        up_out, down_out, boundary_out = self.propagate(cochain.up_index, cochain.down_index,
                                                    cochain.boundary_index, x=cochain.x,
                                                    up_attr=cochain.kwargs['up_attr'],
                                                    down_attr=cochain.kwargs['down_attr'],
                                                    boundary_attr=cochain.kwargs['boundary_attr'])
        # down or boundary will be zero if one of them is not used.
        return cochain.x + up_out + down_out + boundary_out


class DummyCellularMessagePassing(torch.nn.Module):
    def __init__(self, input_dim=1, max_dim: int = 2, use_boundary_msg=False, use_down_msg=True):
        super(DummyCellularMessagePassing, self).__init__()
        self.max_dim = max_dim
        self.mp_levels = torch.nn.ModuleList()
        for dim in range(max_dim+1):
            mp = DummyCochainMessagePassing(input_dim, input_dim, boundary_msg_size=input_dim,
                                          use_boundary_msg=use_boundary_msg, use_down_msg=use_down_msg)
            self.mp_levels.append(mp)
    
    def forward(self, *cochain_params: CochainMessagePassingParams):
        assert len(cochain_params) <= self.max_dim+1

        out = []
        for dim in range(len(cochain_params)):
            out.append(self.mp_levels[dim].forward(cochain_params[dim]))
        return out